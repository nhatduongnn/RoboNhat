import sys
import os
import numpy as np
import pysam
import matplotlib.pyplot as plt
import math
import pandas
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects.packages import importr
import rpy2.robjects.vectors as vectors
import rpy2.robjects as ro
import io
from contextlib import redirect_stdout
import inspect
import seaborn as sns
from Bio import SeqIO
from Bio.SeqFeature import SeqFeature, FeatureLocation
sys.path.insert(0, '../pkg/')
sys.path.insert(0, '/home/rapiduser/programs/RoboCOP/pkg/')
#from robocop.utils.parameters import computeMNaseTFPhisMus

import numpy as np
from collections import defaultdict
from Bio.Seq import reverse_complement

def computeMNaseTFPhisMus(bamFile, csvFile, tmpDir, fragRange, filename, offset=0):
    """
    Negative binomial distribution parameters for DMS-seq data at TF binding sites.
    Returns parameters for each TF, strand, and base position.
    """
    
    # Initialize R fitdistrplus
    fitdist = importr('fitdistrplus')
    
    # Load BAM file
    samfile = pysam.AlignmentFile(bamFile, "rb")
    
    # Load reference genome - TODO: Make this configurable instead of hardcoded
    fasta_file = {}
    for seq_record in SeqIO.parse("/home/rapiduser/programs/RoboCOP/analysis/inputs/SacCer3.fa", "fasta"):
        fasta_file[seq_record.id] = seq_record.seq
    
    # Load TF binding sites
    tfs = pandas.read_csv(csvFile, sep='\t', header=None)
    tfs = tfs.rename(columns={0: 'chr', 1: 'start', 2: 'end', 3: 'tf_name', 4: 'score', 5: 'strand'})
    
    # Filter TFs by count threshold
    tf_counts = tfs.groupby('tf_name')['chr'].count()
    ind_tfs = list(tf_counts.loc[tf_counts >= 50].index)
    
    # Initialize the nested dictionary structure: mu/phi -> watson/crick -> TF -> ACGT
    params_all = {
        'mu': {'watson': {}, 'crick': {}},
        'phi': {'watson': {}, 'crick': {}}
    }
    
    # Process each TF individually
    for tf_name in ind_tfs:
        tf_params = compute_individual_DMSTFPhisMus(
            samfile, tfs, tf_name, fasta_file, fitdist, offset
        )
        # Restructure the data into the desired format
        for strand in ['watson', 'crick']:
            params_all['mu'][strand][tf_name] = tf_params['mu'][strand]
            params_all['phi'][strand][tf_name] = tf_params['phi'][strand]

        plot_mu_phi_heatmaps(params_all['mu']['watson'][tf_name], params_all['phi']['watson'][tf_name], strand_label="Watson", tf_name=tf_name)
        plot_mu_phi_heatmaps(params_all['mu']['crick'][tf_name], params_all['phi']['crick'][tf_name], strand_label="Crick", tf_name=tf_name)
    
    # Handle TFs with < 50 sites as a combined group if needed
    tf_counts_low = tf_counts.loc[tf_counts < 50]
    if len(tf_counts_low) > 0:
        # Combine all low-count TFs into one group
        combined_tfs = list(tf_counts_low.index)
        combined_params = compute_combined_DMSTFPhisMus(
            samfile, tfs, combined_tfs, fasta_file, fitdist, offset
        )
        for strand in ['watson', 'crick']:
            params_all['mu'][strand]['combined_low_count'] = combined_params['mu'][strand]
            params_all['phi'][strand]['combined_low_count'] = combined_params['phi'][strand]
    
    samfile.close()


    return params_all


def compute_individual_DMSTFPhisMus(samfile, tfs_df, tf_name, fasta_file, fitdist, offset=0):
    """
    Compute DMS-seq parameters for a single TF.
    """
    # Initialize count arrays for each strand and base
    base_names = ['A', 'C', 'G', 'T']
    strand_names = ['watson', 'crick']
    
    tf_counts = {strand: {base: [] for base in base_names} for strand in strand_names}
    
    # Get all sites for this TF (only negative strand as per original logic)
    one_tf_df = tfs_df.loc[(tfs_df['tf_name'] == tf_name) & (tfs_df['strand'] == '-')]
    
    if len(one_tf_df) == 0:
        return create_default_params()
    
    # Get TF length (assuming consistent length for this TF)
    tf_len = one_tf_df.iloc[0]['end'] - one_tf_df.iloc[0]['start'] + 1
    
    for i1, r1 in one_tf_df.iterrows():
        chrm = r1['chr']
        
        # Initialize counts for this TF site
        site_counts = {
            strand: {base: [1] * tf_len for base in base_names} 
            for strand in strand_names
        }
        
        # Process reads overlapping this TF site
        region = samfile.fetch(chrm, r1['start'] - 1, r1['end'] + 1)
        
        for read in region:
            if read.template_length == 0:
                continue
                
            if read.template_length > 0:  # Watson strand
                frag_start = read.reference_start + 1 - 1  # 5' methylation site
                if r1['start'] <= frag_start <= r1['end']:
                    nucleotide = SeqFeature(FeatureLocation(frag_start-1, frag_start)).extract(
                        fasta_file[chrm]
                    )
                    pos = frag_start - r1['start']
                    if str(nucleotide) in base_names:
                        site_counts['watson'][str(nucleotide)][pos] += 1
                        
            elif read.template_length < 0:  # Crick strand
                frag_end = read.reference_end + 1 - 1 + 1  # 3' methylation site
                if r1['start'] <= frag_end <= r1['end']:
                    nucleotide = SeqFeature(FeatureLocation(frag_end-1, frag_end)).extract(
                        fasta_file[chrm]
                    )
                    pos = frag_end - r1['start']
                    # Reverse complement mapping for crick strand
                    complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
                    if str(nucleotide) in complement_map:
                        site_counts['crick'][complement_map[str(nucleotide)]][pos] += 1
        
        # Accumulate counts across all sites for this TF
        for strand in strand_names:
            for base in base_names:
                tf_counts[strand][base].extend(site_counts[strand][base])
    
    # Fit negative binomial parameters
    return fit_nb_parameters(tf_counts, tf_len, len(one_tf_df), fitdist)


def compute_combined_DMSTFPhisMus(samfile, tfs_df, tf_names_list, fasta_file, fitdist, offset=0):
    """
    Compute combined parameters for multiple low-count TFs.
    """
    # Similar logic to individual but combining multiple TF types
    base_names = ['A', 'C', 'G', 'T']
    strand_names = ['watson', 'crick']
    
    tf_counts = {strand: {base: [] for base in base_names} for strand in strand_names}
    
    combined_df = tfs_df.loc[
        (tfs_df['tf_name'].isin(tf_names_list)) & (tfs_df['strand'] == '-')
    ]
    
    if len(combined_df) == 0:
        return create_default_params()
    
    # Use a standard length or the most common length
    tf_lengths = combined_df['end'] - combined_df['start'] + 1
    tf_len = int(tf_lengths.mode().iloc[0])  # Most common length
    
    # Process each site (similar to individual function)
    for i1, r1 in combined_df.iterrows():
        # Skip if this site has different length than standard
        if (r1['end'] - r1['start'] + 1) != tf_len:
            continue
            
        # Same processing logic as in compute_individual_DMSTFPhisMus
        chrm = r1['chr']
        
        site_counts = {
            strand: {base: [1] * tf_len for base in base_names} 
            for strand in strand_names
        }
        
        region = samfile.fetch(chrm, r1['start'] - 1, r1['end'] + 1)
        
        for read in region:
            if read.template_length == 0:
                continue
                
            if read.template_length > 0:  # Watson strand
                frag_start = read.reference_start + 1 - 1
                if r1['start'] <= frag_start <= r1['end']:
                    nucleotide = SeqFeature(FeatureLocation(frag_start-1, frag_start)).extract(
                        fasta_file[chrm]
                    )
                    pos = frag_start - r1['start']
                    if str(nucleotide) in base_names:
                        site_counts['watson'][str(nucleotide)][pos] += 1
                        
            elif read.template_length < 0:  # Crick strand
                frag_end = read.reference_end + 1 - 1 + 1
                if r1['start'] <= frag_end <= r1['end']:
                    nucleotide = SeqFeature(FeatureLocation(frag_end-1, frag_end)).extract(
                        fasta_file[chrm]
                    )
                    pos = frag_end - r1['start']
                    complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
                    if str(nucleotide) in complement_map:
                        site_counts['crick'][complement_map[str(nucleotide)]][pos] += 1
        
        for strand in strand_names:
            for base in base_names:
                tf_counts[strand][base].extend(site_counts[strand][base])
    
    return fit_nb_parameters(tf_counts, tf_len, len(combined_df), fitdist)


def fit_nb_parameters(tf_counts, tf_len, num_sites, fitdist):
    """
    Fit negative binomial parameters to the count data.
    Returns structure: strand -> base -> position_array
    """
    import numpy as np
    from contextlib import redirect_stdout
    import io
    import rpy2.robjects.vectors as vectors
    
    base_names = ['A', 'C', 'G', 'T']
    strand_names = ['watson', 'crick']
    
    params = {
        'mu': {strand: {base: np.zeros(tf_len) for base in base_names} for strand in strand_names},
        'phi': {strand: {base: np.zeros(tf_len) for base in base_names} for strand in strand_names}
    }
    
    try:
        f = io.StringIO()
        with redirect_stdout(f):
            for strand in strand_names:
                for base in base_names:
                    # Reshape counts to (num_sites, tf_len)
                    counts_array = np.array(tf_counts[strand][base]).reshape(num_sites, tf_len)
                    
                    for pos in range(tf_len):
                        pos_counts = counts_array[:, pos]
                        
                        # Fit negative binomial
                        nb_fit = fitdist.fitdist(
                            vectors.IntVector(pos_counts), 'nbinom', method="mle"
                        )
                        estimates = nb_fit.rx2("estimate")
                        
                        params['mu'][strand][base][pos] = estimates.rx2("mu")[0]
                        params['phi'][strand][base][pos] = estimates.rx2("size")[0]
                        
    except Exception as e:
        print(f"Error fitting parameters: {e}")
        return create_default_params_individual()
    
    return params


def create_default_params():
    """
    Create default parameters when fitting fails.
    Returns structure: mu/phi -> watson/crick -> TF -> ACGT
    """
    base_names = ['A', 'C', 'G', 'T']
    strand_names = ['watson', 'crick']
    tf_len = 14  # Default length
    
    return {
        'mu': {strand: {base: np.full(tf_len, 0.002) for base in base_names} for strand in strand_names},
        'phi': {strand: {base: np.full(tf_len, 100) for base in base_names} for strand in strand_names}
    }


def create_default_params_individual():
    """
    Create default parameters for individual TF fitting.
    Returns structure: strand -> base -> position_array
    """
    base_names = ['A', 'C', 'G', 'T']
    strand_names = ['watson', 'crick']
    tf_len = 14  # Default length
    
    return {
        'mu': {strand: {base: np.full(tf_len, 0.002) for base in base_names} for strand in strand_names},
        'phi': {strand: {base: np.full(tf_len, 100) for base in base_names} for strand in strand_names}
    }



def plot_mu_phi_heatmaps(mu_dict, phi_dict, strand_label, tf_name):
    """
    Plot heatmaps of mu and phi for a given strand.

    Args:
        mu_dict: dict of {base: np.array of mu values}
        phi_dict: dict of {base: np.array of phi values}
        strand_label: "Watson" or "Crick"
        tf_name: string name of the transcription factor
    """
    bases = ["A", "C", "G", "T"]
    motif_len = len(next(iter(mu_dict.values())))
    num_bases = len(bases)

    # Stack into 2D arrays (base x position)
    mu_matrix = np.vstack([mu_dict[base] for base in bases])
    phi_matrix = np.vstack([phi_dict[base] for base in bases])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    im0 = axes[0].imshow(mu_matrix, aspect="auto", cmap="Reds")
    axes[0].set_title(f"{tf_name} - {strand_label} - Mu")
    axes[0].set_yticks(range(num_bases))
    axes[0].set_yticklabels(bases)
    axes[0].set_xlabel("Motif Position")
    axes[0].set_ylabel("Base")
    fig.colorbar(im0, ax=axes[0], label="Mu")

    im1 = axes[1].imshow(phi_matrix, aspect="auto", cmap="Blues")
    axes[1].set_title(f"{tf_name} - {strand_label} - Phi")
    axes[1].set_yticks(range(num_bases))
    axes[1].set_yticklabels(bases)
    axes[1].set_xlabel("Motif Position")
    fig.colorbar(im1, ax=axes[1], label="Phi")

    plt.tight_layout()
    plt.show()

    # plt.imshow(full_cell_phi[0:4,:], cmap='Blues')
    # plt.title('Phi (overdispersion) values for Watson strand')
    # plt.colorbar()  # Add a colorbar to show the intensity scale
    # plt.show()




a = computeMNaseTFPhisMus("/home/rapiduser/projects/DMS-seq/DM1664/DM1664_trim_3prime_18bp_remaining_name_change_sorted.bam",\
                          "/home/rapiduser/programs/RoboCOP/analysis/inputs/MacIsaac_sacCer3_liftOver_Abf1_Reb1.bed",\
                            "/home/rapiduser/programs/RoboCOP/analysis/robocop_train/tmpDir",\
                            (0, 80),\
                                None,\
                                    0)

# Okay maybe this is it
print(a)