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

from collections import defaultdict
from Bio.Seq import reverse_complement


def computeMNaseTFPhisMus(bamFile, csvFile, tmpDir, fragRange, filename, offset=0):
    """
    Compute negative binomial distribution parameters (mu and phi) for DMS-seq methylation 
    data at transcription factor binding sites. Processes each TF individually if it has 
    ≥50 binding sites, otherwise combines low-count TFs. Automatically calculates parameters 
    for both Watson/Crick motif orientations and signal strands.
    
    Parameters:
    -----------
    bamFile : str
        Path to the BAM file containing aligned sequencing reads with fragment information
    csvFile : str  
        Path to tab-separated file containing TF binding sites with columns:
        chr, start, end, tf_name, score, strand
    tmpDir : str
        Path to temporary directory (currently unused but kept for compatibility)
    fragRange : tuple or list
        Fragment size range filter (currently unused but kept for compatibility)  
    filename : str
        Output filename prefix (currently unused but kept for compatibility)
    offset : int, optional
        Position offset adjustment (default: 0, currently unused)
        
    Returns:
    --------
    dict
        Nested dictionary with structure:
        {
            'mu': {
                'TF_name': {
                    'Watson Motif': {
                        'Watson Signal': {'A': array, 'C': array, 'G': array, 'T': array},
                        'Crick Signal': {'A': array, 'C': array, 'G': array, 'T': array}
                    },
                    'Crick Motif': {
                        'Watson Signal': {'A': array, 'C': array, 'G': array, 'T': array},
                        'Crick Signal': {'A': array, 'C': array, 'G': array, 'T': array}
                    }
                }
            },
            'phi': { ... same structure as 'mu' ... }
        }
        
        Arrays contain position-wise parameters (length = TF motif length)
        TFs with <50 sites are grouped under 'combined_low_count' key
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
    
    # Initialize the nested dictionary structure: mu/phi -> TF -> motif_strand -> signal_strand -> ACGT
    params_all = {
        'mu': {},
        'phi': {}
    }
    
    # Process each TF individually
    for tf_name in ind_tfs:
        params_all['mu'][tf_name] = {}
        params_all['phi'][tf_name] = {}
        
        # Process both Watson and Crick motif orientations
        for motif_strand in ['+', '-']:
            motif_name = 'Watson Motif' if motif_strand == '+' else 'Crick Motif'
            
            tf_params = compute_individual_DMSTFPhisMus(
                samfile, tfs, tf_name, motif_strand, fasta_file, fitdist, offset
            )
            
            # Structure: TF -> motif_strand -> signal_strand -> base
            params_all['mu'][tf_name][motif_name] = {
                'Watson Signal': tf_params['mu']['watson'],
                'Crick Signal': tf_params['mu']['crick']
            }
            params_all['phi'][tf_name][motif_name] = {
                'Watson Signal': tf_params['phi']['watson'],
                'Crick Signal': tf_params['phi']['crick']
            }

        # plot_mu_phi_heatmaps(params_all['mu'][tf_name]['Watson Motif']['Watson Signal'], params_all['phi'][tf_name]['Watson Motif']['Watson Signal'], strand_label="Watson", tf_name=tf_name)
        # plot_mu_phi_heatmaps(params_all['mu'][tf_name]['Watson Motif']['Crick Signal'], params_all['phi'][tf_name]['Watson Motif']['Crick Signal'], strand_label="Crick", tf_name=tf_name)
        # plot_mu_phi_heatmaps(params_all['mu'][tf_name]['Crick Motif']['Watson Signal'], params_all['phi'][tf_name]['Crick Motif']['Watson Signal'], strand_label="Watson", tf_name=tf_name)
        # plot_mu_phi_heatmaps(params_all['mu'][tf_name]['Crick Motif']['Crick Signal'], params_all['phi'][tf_name]['Crick Motif']['Crick Signal'], strand_label="Crick", tf_name=tf_name)
    
    
    # Handle TFs with < 50 sites as a combined group if needed
    tf_counts_low = tf_counts.loc[tf_counts < 50]
    if len(tf_counts_low) > 0:
        # Combine all low-count TFs into one group
        combined_tfs = list(tf_counts_low.index)
        params_all['mu']['combined_low_count'] = {}
        params_all['phi']['combined_low_count'] = {}
        
        for motif_strand in ['+', '-']:
            motif_name = 'Watson Motif' if motif_strand == '+' else 'Crick Motif'
            
            combined_params = compute_combined_DMSTFPhisMus(
                samfile, tfs, combined_tfs, motif_strand, fasta_file, fitdist, offset
            )
            
            params_all['mu']['combined_low_count'][motif_name] = {
                'Watson Signal': combined_params['mu']['watson'],
                'Crick Signal': combined_params['mu']['crick']
            }
            params_all['phi']['combined_low_count'][motif_name] = {
                'Watson Signal': combined_params['phi']['watson'],
                'Crick Signal': combined_params['phi']['crick']
            }
    
    samfile.close()
    return params_all


def compute_individual_DMSTFPhisMus(samfile, tfs_df, tf_name, motif_strand, fasta_file, fitdist, offset=0):
    """
    Compute negative binomial parameters for a single transcription factor on one motif strand.
    Extracts methylation fragment counts at each nucleotide position, distinguishes Watson/Crick
    signal strands, and fits negative binomial distributions across all binding sites.
    
    Parameters:
    -----------
    samfile : pysam.AlignmentFile
        Opened BAM file object for reading sequencing data
    tfs_df : pandas.DataFrame
        DataFrame containing TF binding site information with columns:
        chr, start, end, tf_name, score, strand
    tf_name : str
        Name of the specific transcription factor to process
    motif_strand : str
        Motif orientation: '+' for Watson Motif, '-' for Crick Motif
    fasta_file : dict
        Dictionary mapping chromosome names to Bio.SeqRecord.seq objects
        containing reference genome sequences
    fitdist : rpy2 R package
        R fitdistrplus package imported via rpy2 for negative binomial fitting
    offset : int, optional
        Position offset adjustment (default: 0, currently unused)
        
    Returns:
    --------
    dict
        Dictionary with structure:
        {
            'mu': {
                'watson': {'A': array, 'C': array, 'G': array, 'T': array},
                'crick': {'A': array, 'C': array, 'G': array, 'T': array}
            },
            'phi': {
                'watson': {'A': array, 'C': array, 'G': array, 'T': array},
                'crick': {'A': array, 'C': array, 'G': array, 'T': array}
            }
        }
        
        Arrays contain fitted parameters for each position in the TF motif
    """
    # Initialize count arrays for each signal strand and base
    base_names = ['A', 'C', 'G', 'T']
    signal_strand_names = ['watson', 'crick']
    
    tf_counts = {strand: {base: [] for base in base_names} for strand in signal_strand_names}
    
    # Get all sites for this TF on the specified motif strand
    one_tf_df = tfs_df.loc[(tfs_df['tf_name'] == tf_name) & (tfs_df['strand'] == motif_strand)]
    
    if len(one_tf_df) == 0:
        return create_default_params_individual()
    
    # Get TF length (assuming consistent length for this TF)
    tf_len = one_tf_df.iloc[0]['end'] - one_tf_df.iloc[0]['start'] + 1
    
    for i1, r1 in one_tf_df.iterrows():
        chrm = r1['chr']
        
        # Initialize counts for this TF site
        site_counts = {
            strand: {base: [1] * tf_len for base in base_names} 
            for strand in signal_strand_names
        }
        
        # Process reads overlapping this TF site
        region = samfile.fetch(chrm, r1['start'] - 1, r1['end'] + 1)
        
        for read in region:
            if read.template_length == 0:
                continue
                
            if read.template_length > 0:  # Watson signal strand
                frag_start = read.reference_start + 1 - 1  # 5' methylation site
                if r1['start'] <= frag_start <= r1['end']:
                    nucleotide = SeqFeature(FeatureLocation(frag_start-1, frag_start)).extract(
                        fasta_file[chrm]
                    )
                    pos = frag_start - r1['start']
                    if str(nucleotide) in base_names:
                        site_counts['watson'][str(nucleotide)][pos] += 1
                        
            elif read.template_length < 0:  # Crick signal strand
                frag_end = read.reference_end + 1 - 1 + 1  # 3' methylation site
                if r1['start'] <= frag_end <= r1['end']:
                    nucleotide = SeqFeature(FeatureLocation(frag_end-1, frag_end)).extract(
                        fasta_file[chrm]
                    )
                    pos = frag_end - r1['start']
                    # Reverse complement mapping for crick signal strand
                    complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
                    if str(nucleotide) in complement_map:
                        site_counts['crick'][complement_map[str(nucleotide)]][pos] += 1
        
        # Accumulate counts across all sites for this TF
        for strand in signal_strand_names:
            for base in base_names:
                tf_counts[strand][base].extend(site_counts[strand][base])
    
    # Fit negative binomial parameters
    return fit_nb_parameters(tf_counts, tf_len, len(one_tf_df), fitdist)


def compute_combined_DMSTFPhisMus(samfile, tfs_df, tf_names_list, motif_strand, fasta_file, fitdist, offset=0):
    """
    Compute combined negative binomial parameters for multiple low-count transcription factors.
    Handles TFs with <50 binding sites by combining them into a single group for parameter 
    estimation. Standardizes to the most common motif length and excludes sites with different lengths.
    
    Parameters:
    -----------
    samfile : pysam.AlignmentFile
        Opened BAM file object for reading sequencing data
    tfs_df : pandas.DataFrame
        DataFrame containing TF binding site information with columns:
        chr, start, end, tf_name, score, strand
    tf_names_list : list
        List of TF names to combine (typically those with <50 binding sites)
    motif_strand : str
        Motif orientation: '+' for Watson Motif, '-' for Crick Motif
    fasta_file : dict
        Dictionary mapping chromosome names to Bio.SeqRecord.seq objects
        containing reference genome sequences
    fitdist : rpy2 R package
        R fitdistrplus package imported via rpy2 for negative binomial fitting
    offset : int, optional
        Position offset adjustment (default: 0, currently unused)
        
    Returns:
    --------
    dict
        Dictionary with same structure as compute_individual_DMSTFPhisMus:
        {
            'mu': {
                'watson': {'A': array, 'C': array, 'G': array, 'T': array},
                'crick': {'A': array, 'C': array, 'G': array, 'T': array}
            },
            'phi': {
                'watson': {'A': array, 'C': array, 'G': array, 'T': array},
                'crick': {'A': array, 'C': array, 'G': array, 'T': array}
            }
        }
        
        Arrays contain fitted parameters for each position using combined data
        Sites with lengths different from the mode are excluded
    """
    # Similar logic to individual but combining multiple TF types
    base_names = ['A', 'C', 'G', 'T']
    signal_strand_names = ['watson', 'crick']
    
    tf_counts = {strand: {base: [] for base in base_names} for strand in signal_strand_names}
    
    combined_df = tfs_df.loc[
        (tfs_df['tf_name'].isin(tf_names_list)) & (tfs_df['strand'] == motif_strand)
    ]
    
    if len(combined_df) == 0:
        return create_default_params_individual()
    
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
            for strand in signal_strand_names
        }
        
        region = samfile.fetch(chrm, r1['start'] - 1, r1['end'] + 1)
        
        for read in region:
            if read.template_length == 0:
                continue
                
            if read.template_length > 0:  # Watson signal strand
                frag_start = read.reference_start + 1 - 1
                if r1['start'] <= frag_start <= r1['end']:
                    nucleotide = SeqFeature(FeatureLocation(frag_start-1, frag_start)).extract(
                        fasta_file[chrm]
                    )
                    pos = frag_start - r1['start']
                    if str(nucleotide) in base_names:
                        site_counts['watson'][str(nucleotide)][pos] += 1
                        
            elif read.template_length < 0:  # Crick signal strand
                frag_end = read.reference_end + 1 - 1 + 1
                if r1['start'] <= frag_end <= r1['end']:
                    nucleotide = SeqFeature(FeatureLocation(frag_end-1, frag_end)).extract(
                        fasta_file[chrm]
                    )
                    pos = frag_end - r1['start']
                    complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
                    if str(nucleotide) in complement_map:
                        site_counts['crick'][complement_map[str(nucleotide)]][pos] += 1
        
        for strand in signal_strand_names:
            for base in base_names:
                tf_counts[strand][base].extend(site_counts[strand][base])
    
    return fit_nb_parameters(tf_counts, tf_len, len(combined_df), fitdist)


def fit_nb_parameters(tf_counts, tf_len, num_sites, fitdist):
    """
    Fit negative binomial distribution parameters to methylation count data.
    Takes accumulated count data organized by signal strand and nucleotide base, reshapes 
    to matrices, and fits position-wise distributions using R's fitdistrplus package.
    
    Parameters:
    -----------
    tf_counts : dict
        Nested dictionary containing count data with structure:
        {
            'watson': {'A': list, 'C': list, 'G': list, 'T': list},
            'crick': {'A': list, 'C': list, 'G': list, 'T': list}
        }
        Each list contains counts flattened across all sites and positions
    tf_len : int
        Length of the transcription factor motif (number of base positions)
    num_sites : int
        Number of TF binding sites used in the analysis
    fitdist : rpy2 R package
        R fitdistrplus package imported via rpy2 for negative binomial MLE fitting
        
    Returns:
    --------
    dict
        Dictionary containing fitted parameters with structure:
        {
            'mu': {
                'watson': {'A': array, 'C': array, 'G': array, 'T': array},
                'crick': {'A': array, 'C': array, 'G': array, 'T': array}
            },
            'phi': {
                'watson': {'A': array, 'C': array, 'G': array, 'T': array},
                'crick': {'A': array, 'C': array, 'G': array, 'T': array}
            }
        }
        
        Arrays have length tf_len with fitted mu (mean) and phi (size/dispersion) parameters
        Returns default parameters if fitting fails
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
    Create default negative binomial parameters when fitting fails or no data is available.
    Provides conservative estimates: mu=0.002 (low methylation) and phi=100 (high dispersion).
    
    Parameters:
    -----------
    None
        
    Returns:
    --------
    dict
        Dictionary with default parameters using structure:
        {
            'mu': {
                'watson': {'A': array, 'C': array, 'G': array, 'T': array},
                'crick': {'A': array, 'C': array, 'G': array, 'T': array}
            },
            'phi': {
                'watson': {'A': array, 'C': array, 'G': array, 'T': array},
                'crick': {'A': array, 'C': array, 'G': array, 'T': array}
            }
        }
        
        All arrays have length 14 (default TF motif length) filled with default values
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
    Create default negative binomial parameters for individual TF parameter fitting.
    Provides fallback values when individual TF fitting fails, using same conservative 
    estimates as create_default_params().
    
    Parameters:
    -----------
    None
        
    Returns:
    --------
    dict
        Dictionary with default parameters using structure:
        {
            'mu': {
                'watson': {'A': array, 'C': array, 'G': array, 'T': array},
                'crick': {'A': array, 'C': array, 'G': array, 'T': array}
            },
            'phi': {
                'watson': {'A': array, 'C': array, 'G': array, 'T': array},
                'crick': {'A': array, 'C': array, 'G': array, 'T': array}
            }
        }
        
        All arrays have length 14 (default TF motif length) with:
        - mu values: 0.002 (low methylation rate)
        - phi values: 100 (high dispersion for biological variability)
    """
    import numpy as np
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

        # plot_mu_phi_heatmaps(params_all['mu'][tf_name]['Watson Motif']['Watson Signal'], params_all['phi'][tf_name]['Watson Motif']['Watson Signal'], strand_label="Watson", tf_name=tf_name)
        # plot_mu_phi_heatmaps(params_all['mu'][tf_name]['Watson Motif']['Crick Signal'], params_all['phi'][tf_name]['Watson Motif']['Crick Signal'], strand_label="Crick", tf_name=tf_name)
        # plot_mu_phi_heatmaps(params_all['mu'][tf_name]['Crick Motif']['Watson Signal'], params_all['phi'][tf_name]['Crick Motif']['Watson Signal'], strand_label="Watson", tf_name=tf_name)
        # plot_mu_phi_heatmaps(params_all['mu'][tf_name]['Crick Motif']['Crick Signal'], params_all['phi'][tf_name]['Crick Motif']['Crick Signal'], strand_label="Crick", tf_name=tf_name)
    


