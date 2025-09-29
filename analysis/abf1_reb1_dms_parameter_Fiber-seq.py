import sys
import os
import numpy as np
import pysam
import matplotlib.pyplot as plt
import math
import pandas as pd
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




def computeMNaseTFPhisMus(tech, bamFile, modkitFile, csvFile, tmpDir, fragRange, filename, offset=0):
    fitdist = importr('fitdistrplus')
    if tech != "Fiber_seq":
        samfile = pysam.AlignmentFile(bamFile, "rb")
    fasta_file = {}
    for seq_record in SeqIO.parse("/home/rapiduser/programs/RoboCOP/analysis/inputs/SacCer3.fa", "fasta"):
        fasta_file[seq_record.id] = seq_record.seq
    tfs = pd.read_csv(csvFile, sep='\t')
    tfs = tfs.rename(columns={'TF': 'tf_name'})
    tf_counts = tfs.groupby('tf_name')['chr'].count()
    ind_tfs = list(tf_counts.loc[tf_counts >= 50].index)
    params_all = {'mu': {}, 'phi': {}}

    # Load Modkit file only once outside loop
    modified_bases_df = pd.read_csv(modkitFile, sep='\t', header=None)
    # Split the 9th column into multiple columns (if following previous code pattern)
    split_columns = modified_bases_df[9].str.split(' ', expand=True)
    split_columns.columns = [i for i in range(9,9+split_columns.shape[1])]
    modified_bases_df = pd.concat([modified_bases_df.drop(columns=[9]), split_columns], axis=1)

    for tf_name in ind_tfs:
        if tech != "Fiber_seq":
            watson_counts = compute_individual_DMSTFPhisMus(
                samfile, tfs, tf_name, '+', fasta_file, fitdist, offset
            )
            crick_counts = compute_individual_DMSTFPhisMus(
                samfile, tfs, tf_name, '-', fasta_file, fitdist, offset
            )
        else:
            # Use the modified_bases_df here instead of modkitFile path
            watson_counts = compute_individual_Fiber_seq_TFPhisMus(
                modified_bases_df, tfs, tf_name, '+', fasta_file, fitdist, offset
            )
            crick_counts = compute_individual_Fiber_seq_TFPhisMus(
                modified_bases_df, tfs, tf_name, '-', fasta_file, fitdist, offset
            )

        # plot_mu_phi_heatmaps(watson_counts['watson_signal'], watson_counts['crick_signal'], strand_label="Watson", tf_name=tf_name)
        # plot_mu_phi_heatmaps(crick_counts['watson_signal'], crick_counts['crick_signal'], strand_label="Crick", tf_name=tf_name)
        combined_counts = combine_motif_counts(watson_counts, crick_counts)
        if combined_counts['num_sites'] == 0:
            params = create_default_params_individual()
        else:
            params = fit_nb_parameters(combined_counts, combined_counts['tf_len'], combined_counts['num_sites'], fitdist)
        params_all['mu'][tf_name] = {
            'Watson Signal': params['mu']['watson'],
            'Crick Signal': params['mu']['crick']
        }
        params_all['phi'][tf_name] = {
            'Watson Signal': params['phi']['watson'],
            'Crick Signal': params['phi']['crick']
        }
        plot_mu_phi_heatmaps(params_all['mu'][tf_name]['Watson Signal'], params_all['phi'][tf_name]['Watson Signal'], strand_label="Watson", tf_name=tf_name)
        plot_mu_phi_heatmaps(params_all['mu'][tf_name]['Crick Signal'], params_all['phi'][tf_name]['Crick Signal'], strand_label="Crick", tf_name=tf_name)
    
    tf_counts_low = tf_counts.loc[tf_counts < 50]
    if len(tf_counts_low) > 0:
        combined_tfs = list(tf_counts_low.index)
        if tech != "Fiber_seq":
            watson_counts = compute_combined_DMSTFPhisMus(
                samfile, tfs, combined_tfs, '+', fasta_file, fitdist, offset
            )
            crick_counts = compute_combined_DMSTFPhisMus(
                samfile, tfs, combined_tfs, '-', fasta_file, fitdist, offset
            )
        else:
            # Use the modified_bases_df instead of file path
            watson_counts = compute_combined_Fiber_seq_TFPhisMus(
                modified_bases_df, tfs, combined_tfs, '+', fasta_file, fitdist, offset
            )
            crick_counts = compute_combined_Fiber_seq_TFPhisMus(
                modified_bases_df, tfs, combined_tfs, '-', fasta_file, fitdist, offset
            )
        combined_counts = combine_motif_counts(watson_counts, crick_counts)
        if combined_counts['num_sites'] == 0:
            params = create_default_params_individual()
        else:
            params = fit_nb_parameters(combined_counts, combined_counts['tf_len'], combined_counts['num_sites'], fitdist)
        params_all['mu']['combined_low_count'] = {
            'Watson Signal': params['mu']['watson'],
            'Crick Signal': params['mu']['crick']
        }
        params_all['phi']['combined_low_count'] = {
            'Watson Signal': params['phi']['watson'],
            'Crick Signal': params['phi']['crick']
        }
    else:
        # No low-count TFs; still populate default combined_low_count params
        params = create_default_params_individual()
        params_all['mu']['combined_low_count'] = {
            'Watson Signal': params['mu']['watson'],
            'Crick Signal': params['mu']['crick']
        }
        params_all['phi']['combined_low_count'] = {
            'Watson Signal': params['phi']['watson'],
            'Crick Signal': params['phi']['crick']
        }

    if tech != "Fiber_seq":
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
    tfs_df : pd.DataFrame
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
    # Since the input is a bed file, end - start, no need to add 1 to calculate the TF length
    tf_len = one_tf_df.iloc[0]['end'] - one_tf_df.iloc[0]['start']
    
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
    return {
        'watson_signal': tf_counts['watson'],
        'crick_signal': tf_counts['crick'],
        'tf_len': tf_len,
        'num_sites': len(one_tf_df)
    }




def compute_individual_Fiber_seq_TFPhisMus(modified_bases_df, tfs_df, tf_name, motif_strand, fasta_file, fitdist, offset=0):
    """
    ...
    Parameters:
    -----------
    modified_bases_df : pd.DataFrame
        DataFrame of modkit file with columns already split, including modification counts.
    ...
    """
    base_names = ['A', 'C', 'G', 'T']
    signal_strand_names = ['watson', 'crick']
    tf_counts = {strand: {base: [] for base in base_names} for strand in signal_strand_names}
    one_tf_df = tfs_df.loc[(tfs_df['tf_name'] == tf_name) & (tfs_df['strand'] == motif_strand)]
    if len(one_tf_df) == 0:
        return create_default_params_individual()
    tf_len = one_tf_df.iloc[0]['end'] - one_tf_df.iloc[0]['start']

    # No file reading or splitting here, use modified_bases_df directly

    for i1, r1 in one_tf_df.iterrows():
        chrm = r1['chr']
        site_counts = {
            strand: {base: [1] * tf_len for base in base_names}
            for strand in signal_strand_names
        }
        relevant_rows = modified_bases_df[
            (modified_bases_df[0] == chrm) &
            (modified_bases_df[1] < r1['end']) &
            (modified_bases_df[2] > r1['start'])
        ]
        for _, row in relevant_rows.iterrows():
            modified_base = row[3].upper()
            strand_info = row[5]
            count = int(row[11])
            pos = row[1] - r1['start']
            if strand_info == '+':
                if modified_base in base_names:
                    site_counts['watson'][modified_base][pos] += count
            elif strand_info == '-':
                if modified_base in base_names:
                    site_counts['crick'][modified_base][pos] += count
        for strand in signal_strand_names:
            for base in base_names:
                tf_counts[strand][base].extend(site_counts[strand][base])
    return {
        'watson_signal': tf_counts['watson'],
        'crick_signal': tf_counts['crick'],
        'tf_len': tf_len,
        'num_sites': len(one_tf_df)
    }

def compute_combined_Fiber_seq_TFPhisMus(modified_bases_df, tfs_df, tf_names_list, motif_strand, fasta_file, fitdist, offset=0):
    """
    ...
    Parameters:
    -----------
    modified_bases_df : pd.DataFrame
        DataFrame of modkit file with columns already split, including modification counts
    ...
    """
    base_names = ['A', 'C', 'G', 'T']
    signal_strand_names = ['watson', 'crick']
    tf_counts = {strand: {base: [] for base in base_names} for strand in signal_strand_names}
    combined_df = tfs_df.loc[(tfs_df['tf_name'].isin(tf_names_list)) & (tfs_df['strand'] == motif_strand)]
    if len(combined_df) == 0:
        return create_default_params_individual()
    tf_lengths = combined_df['end'] - combined_df['start']
    tf_len = int(tf_lengths.mode().iloc[0])

    # No file reading or splitting here, use modified_bases_df directly

    for _, r1 in combined_df.iterrows():
        if (r1['end'] - r1['start']) != tf_len:
            continue
        chrm = r1['chr']
        start = r1['start']
        end = r1['end']
        site_counts = {strand: {base: [1] * tf_len for base in base_names} for strand in signal_strand_names}
        relevant_rows = modified_bases_df[
            (modified_bases_df[0] == chrm) &
            (modified_bases_df[1] < end) &
            (modified_bases_df[2] > start)
        ]
        for _, row in relevant_rows.iterrows():
            modified_base = str(row[3]).upper()
            strand_info = row[5]
            count = int(row[11])
            pos = int(row[1]) - start
            if pos < 0 or pos >= tf_len:
                continue
            if strand_info == '+':
                if modified_base in base_names:
                    site_counts['watson'][modified_base][pos] += count
            elif strand_info == '-':
                if modified_base in base_names:
                    site_counts['crick'][modified_base][pos] += count
        for strand in signal_strand_names:
            for base in base_names:
                tf_counts[strand][base].extend(site_counts[strand][base])

    return {
        'watson_signal': tf_counts['watson'],
        'crick_signal': tf_counts['crick'],
        'tf_len': tf_len,
        'num_sites': combined_df.shape[0]
    }


# def compute_combined_Fiber_seq_TFPhisMus(modkitFile, tfs_df, tf_names_list, motif_strand, fasta_file, fitdist, offset=0):
#     """
#     Compute combined negative binomial parameters for multiple low-count transcription factors in Fiber_seq data.
#     Combines counts from multiple TFs, standardizes to the most common TF motif length, and aggregates methylation counts
#     for each base (A,C,G,T) and signal strand (Watson/Crick) across all sites.

#     Parameters:
#     -----------
#     modkitFile : str
#         Path to the Modkit pileup file with fiberseq methylation data
#     tfs_df : pd.DataFrame
#         DataFrame of TF binding sites with columns chr, start, end, tf_name, score, strand
#     tf_names_list : list of str
#         List of TF names to combine
#     motif_strand : str
#         '+' for Watson motif strand, '-' for Crick motif strand
#     fasta_file : dict
#         Dictionary mapping chromosome names to Bio.SeqRecord.seq objects with reference genome
#     fitdist : rpy2 R package import
#         Imported fitdistrplus R package via rpy2 (not used directly here, but for consistency)
#     offset : int, optional
#         Position offset adjustment (default=0, currently unused)

#     Returns:
#     --------
#     dict
#         Dictionary with keys:
#           - 'watson_signal': dict of base -> list of counts aggregated over all sites
#           - 'crick_signal': dict of base -> list of counts aggregated over all sites
#           - 'tf_len': int motif length (most common length across TFs)
#           - 'num_sites': int total number of binding sites combined
#     """
#     base_names = ['A', 'C', 'G', 'T']
#     signal_strand_names = ['watson', 'crick']

#     # Initialize empty aggregated counts lists (flattened across all sites)
#     tf_counts = {strand: {base: [] for base in base_names} for strand in signal_strand_names}

#     # Filter TF sites for the combined TFs and given motif strand
#     combined_df = tfs_df.loc[(tfs_df['tf_name'].isin(tf_names_list)) & (tfs_df['strand'] == motif_strand)]

#     if len(combined_df) == 0:
#         from copy import deepcopy
#         # Return default params with default TF motif length 13 if no sites found
#         return create_default_params_individual()

#     # Determine the most common motif length among all TFs in combined list
#     tf_lengths = combined_df['end'] - combined_df['start']
#     tf_len = int(tf_lengths.mode().iloc[0])  # Most common length

#     # Load Modkit file only once outside loop
#     modified_bases_df = pd.read_csv(modkitFile, sep='\t', header=None)
#     # Split the 9th column into multiple columns (if following previous code pattern)
#     split_columns = modified_bases_df[9].str.split(' ', expand=True)
#     split_columns.columns = [i for i in range(9,9+split_columns.shape[1])]
#     modified_bases_df = pd.concat([modified_bases_df.drop(columns=[9]), split_columns], axis=1)

#     # Iterate over each binding site (TF site)
#     for _, r1 in combined_df.iterrows():
#         # Skip sites with different length than most common TF length
#         if (r1['end'] - r1['start']) != tf_len:
#             continue

#         chrm = r1['chr']
#         start = r1['start']
#         end = r1['end']

#         # Initialize site count arrays with pseudocount=1 to avoid zeros
#         site_counts = {strand: {base: [1]*tf_len for base in base_names} for strand in signal_strand_names}

#         # Filter relevant modified base rows overlapping with this TF site
#         relevant_rows = modified_bases_df[
#             (modified_bases_df[0] == chrm) &
#             (modified_bases_df[1] < end) &
#             (modified_bases_df[2] > start)
#         ]

#         for _, row in relevant_rows.iterrows():
#             modified_base = str(row[3]).upper()
#             strand_info = row[5]
#             count = int(row[11])
#             pos = int(row[1]) - start  # position relative to TF start (0-based)

#             if pos < 0 or pos >= tf_len:
#                 continue  # just a safety check (This is potentially here because combined TFs have different lengths)

#             if strand_info == '+':
#                 if modified_base in base_names:
#                     site_counts['watson'][modified_base][pos] += count
#             elif strand_info == '-':
#                 if modified_base in base_names:
#                     site_counts['crick'][modified_base][pos] += count

#         # Append these site counts flattened by base and strand into aggregate lists
#         for strand in signal_strand_names:
#             for base in base_names:
#                 tf_counts[strand][base].extend(site_counts[strand][base])

#     # Return with total number of sites and motif length
#     return {
#         'watson_signal': tf_counts['watson'],
#         'crick_signal': tf_counts['crick'],
#         'tf_len': tf_len,
#         'num_sites': combined_df.shape[0]
#     }

def combine_motif_counts(watson_counts, crick_counts):
    base_names = ['A', 'C', 'G', 'T']
    tf_len = watson_counts['tf_len']
    num_sites_watson = watson_counts['num_sites']
    num_sites_crick = crick_counts['num_sites']

    combined = {'Watson Signal': {b: None for b in base_names},
                'Crick Signal': {b: None for b in base_names},
                'tf_len': tf_len,
                'num_sites': num_sites_watson + num_sites_crick}

    # Helper to reshape counts into (num_sites, tf_len)
    def reshape_counts(counts, num_sites):
        reshaped = {'Watson Signal': {}, 'Crick Signal': {}}
        for signal in ['watson_signal', 'crick_signal']:
            key = 'Watson Signal' if signal == 'watson_signal' else 'Crick Signal'
            for base in base_names:
                arr = np.array(counts[signal][base])
                if num_sites == 0:
                    arr_reshaped = np.empty((0, tf_len), dtype=int)
                else:
                    arr_reshaped = arr.reshape((num_sites, tf_len))
                reshaped[key][base] = arr_reshaped
        return reshaped

    watson_reshaped = reshape_counts(watson_counts, num_sites_watson)
    crick_reshaped = reshape_counts(crick_counts, num_sites_crick)


    # Initialize combined arrays
    for base in base_names:
        # Combine Watson motif Watson signal with reversed Crick motif Crick signal
        combined['Watson Signal'][base] = np.vstack((
            watson_reshaped['Watson Signal'][base],
            crick_reshaped['Crick Signal'][base][:, ::-1]
        )) if (num_sites_watson + num_sites_crick) > 0 else np.empty((0, tf_len), dtype=int)

        # Combine Watson motif Crick signal with reversed Crick motif Watson signal
        combined['Crick Signal'][base] = np.vstack((
            watson_reshaped['Crick Signal'][base],
            crick_reshaped['Watson Signal'][base][:, ::-1]
        )) if (num_sites_watson + num_sites_crick) > 0 else np.empty((0, tf_len), dtype=int)

    return combined

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

    
    # base_names = ['A', 'C', 'G', 'T']
    # strand_names = ['watson', 'crick']
    
    # params = {
    #     'mu': {strand: {base: np.zeros(tf_len) for base in base_names} for strand in strand_names},
    #     'phi': {strand: {base: np.zeros(tf_len) for base in base_names} for strand in strand_names}
    # }
    
    # try:
    #     f = io.StringIO()
    #     with redirect_stdout(f):
    #         for strand in strand_names:
    #             for base in base_names:
    #                 # Reshape counts to (num_sites, tf_len)
    #                 counts_array = np.array(tf_counts[strand][base]).reshape(num_sites, tf_len)
                    
    #                 for pos in range(tf_len):
    #                     pos_counts = counts_array[:, pos]
                        
    #                     # Fit negative binomial
    #                     nb_fit = fitdist.fitdist(
    #                         vectors.IntVector(pos_counts), 'nbinom', method="mle"
    #                     )
    #                     estimates = nb_fit.rx2("estimate")
                        
    #                     params['mu'][strand][base][pos] = estimates.rx2("mu")[0]
    #                     params['phi'][strand][base][pos] = estimates.rx2("size")[0]
                        
    # except Exception as e:
    #     print(f"Error fitting parameters: {e}")
    #     return create_default_params_individual()
    
    # return params
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
                signal_key = 'Watson Signal' if strand == 'watson' else 'Crick Signal'
                for base in base_names:
                    # **tf_counts[signal_key][base] is now 2D (num_sites x tf_len) numpy array**
                    counts_array = tf_counts[signal_key][base]
                    for pos in range(tf_len):
                        pos_counts = counts_array[:, pos]
                        # Fit negative binomial using R's fitdistrplus
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
    tf_len = 13  # Default length
    
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
    tf_len = 13  # Default length
    
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

    # Add text annotations to mu heatmap
    for i in range(num_bases):        # rows (bases)
        for j in range(motif_len):   # cols (positions)
            val = mu_matrix[i, j]
            axes[0].text(j, i, f"{val:.3f}", ha="center", va="center", color="black", fontsize=6)

    im1 = axes[1].imshow(phi_matrix, aspect="auto", cmap="Blues")
    axes[1].set_title(f"{tf_name} - {strand_label} - Phi")
    axes[1].set_yticks(range(num_bases))
    axes[1].set_yticklabels(bases)
    axes[1].set_xlabel("Motif Position")
    fig.colorbar(im1, ax=axes[1], label="Phi")

    # Add text annotations to phi heatmap
    for i in range(num_bases):
        for j in range(motif_len):
            val = phi_matrix[i, j]
            axes[1].text(j, i, f"{val:.3f}", ha="center", va="center", color="black", fontsize=6)

    plt.tight_layout()
    plt.show()

    # plt.imshow(full_cell_phi[0:4,:], cmap='Blues')
    # plt.title('Phi (overdispersion) values for Watson strand')
    # plt.colorbar()  # Add a colorbar to show the intensity scale
    # plt.show()


import pickle

a = computeMNaseTFPhisMus("Fiber_seq",\
                          "/home/rapiduser/projects/DMS-seq/DM1664/DM1664_trim_3prime_18bp_remaining_name_change_sorted.bam",\
                          "/home/rapiduser/projects/Fiber_seq/03202025_barcode01_sup_model_sorted_pileup_all_chr",\
                          "/home/rapiduser/programs/RoboCOP/analysis/inputs/rossi_peak_w_strand_conformed_to_PWM_abf1_reb1.bed",\
                            "/home/rapiduser/programs/RoboCOP/analysis/robocop_train/tmpDir",\
                            (0, 80),\
                                None,\
                                    0)

# Okay maybe this is it
print(a)
abf1_reb1_params = a

# To load the dictionary later, you can use:

# Save the dictionary to a file
with open('inputs/abf1_reb1_params.pkl', 'wb') as f:
    pickle.dump(abf1_reb1_params, f)

# # Load the dictionary from the file
# with open('inputs/abf1_reb1_params.pkl', 'rb') as f:
#     loaded_params = pickle.load(f)





# def computeMNaseTFPhisMus(tech, bamFile, modkitFile, csvFile, tmpDir, fragRange, filename, offset=0):
#     """
#     Compute negative binomial distribution parameters (mu and phi) for DMS-seq methylation 
#     data at transcription factor binding sites. Processes each TF individually if it has 
#     ≥50 binding sites, otherwise combines low-count TFs. Automatically calculates parameters 
#     for both Watson/Crick motif orientations and signal strands.
    
#     Parameters:
#     -----------
#     tech : str
#         Specify whether or not the input data is DMS_seq or Fiber_seq
#     bamFile : str
#         Path to the BAM file containing aligned sequencing reads with fragment information
#     modkitFile : str
#         Path to the Modkit file containing pileup information of fiberseq-reads
#     csvFile : str  
#         Path to tab-separated file containing TF binding sites with columns:
#         chr, start, end, tf_name, score, strand
#     tmpDir : str
#         Path to temporary directory (currently unused but kept for compatibility)
#     fragRange : tuple or list
#         Fragment size range filter (currently unused but kept for compatibility)  
#     filename : str
#         Output filename prefix (currently unused but kept for compatibility)
#     offset : int, optional
#         Position offset adjustment (default: 0, currently unused)
        
#     Returns:
#     --------
#     dict
#         Nested dictionary with structure:
#         {
#             'mu': {
#                 'TF_name': {
#                     'Watson Motif': {
#                         'Watson Signal': {'A': array, 'C': array, 'G': array, 'T': array},
#                         'Crick Signal': {'A': array, 'C': array, 'G': array, 'T': array}
#                     },
#                     'Crick Motif': {
#                         'Watson Signal': {'A': array, 'C': array, 'G': array, 'T': array},
#                         'Crick Signal': {'A': array, 'C': array, 'G': array, 'T': array}
#                     }
#                 }
#             },
#             'phi': { ... same structure as 'mu' ... }
#         }
        
#         Arrays contain position-wise parameters (length = TF motif length)
#         TFs with <50 sites are grouped under 'combined_low_count' key
#     """
    
#     # Initialize R fitdistrplus
#     fitdist = importr('fitdistrplus')
    
#     if tech != "Fiber_seq":
#         # Load BAM file
#         samfile = pysam.AlignmentFile(bamFile, "rb")
    
#     # Load reference genome - TODO: Make this configurable instead of hardcoded
#     fasta_file = {}
#     for seq_record in SeqIO.parse("/home/rapiduser/programs/RoboCOP/analysis/inputs/SacCer3.fa", "fasta"):
#         fasta_file[seq_record.id] = seq_record.seq
    
#     # Load TF binding sites
#     tfs = pd.read_csv(csvFile, sep='\t', header=None)
#     tfs = tfs.rename(columns={0: 'chr', 1: 'start', 2: 'end', 3: 'tf_name', 4: 'score', 5: 'strand'})
    
#     # Filter TFs by count threshold
#     tf_counts = tfs.groupby('tf_name')['chr'].count()
#     ind_tfs = list(tf_counts.loc[tf_counts >= 50].index)
    
#     # Initialize the nested dictionary structure: mu/phi -> TF -> motif_strand -> signal_strand -> ACGT
#     params_all = {
#         'mu': {},
#         'phi': {}
#     }
    
#     # Process each TF individually
#     for tf_name in ind_tfs:
#         params_all['mu'][tf_name] = {}
#         params_all['phi'][tf_name] = {}
        
#         # Process both Watson and Crick motif orientations
#         for motif_strand in ['+', '-']:
#             motif_name = 'Watson Motif' if motif_strand == '+' else 'Crick Motif'
            
#             if tech != "Fiber_seq":
#                 tf_params = compute_individual_DMSTFPhisMus(
#                     samfile, tfs, tf_name, motif_strand, fasta_file, fitdist, offset
#                 )
#             elif tech == "Fiber_seq":
#                 # For Fiber-seq, we need to handle the pileup data differently
#                 tf_params = compute_individual_Fiber_seq_TFPhisMus(
#                     modkitFile, tfs, tf_name, motif_strand, fasta_file, fitdist, offset
#                 )
            
#             # Structure: TF -> motif_strand -> signal_strand -> base
#             params_all['mu'][tf_name][motif_name] = {
#                 'Watson Signal': tf_params['mu']['watson'],
#                 'Crick Signal': tf_params['mu']['crick']
#             }
#             params_all['phi'][tf_name][motif_name] = {
#                 'Watson Signal': tf_params['phi']['watson'],
#                 'Crick Signal': tf_params['phi']['crick']
#             }

#         plot_mu_phi_heatmaps(params_all['mu'][tf_name]['Watson Motif']['Watson Signal'], params_all['phi'][tf_name]['Watson Motif']['Watson Signal'], strand_label="Watson", tf_name=tf_name)
#         plot_mu_phi_heatmaps(params_all['mu'][tf_name]['Watson Motif']['Crick Signal'], params_all['phi'][tf_name]['Watson Motif']['Crick Signal'], strand_label="Crick", tf_name=tf_name)
#         plot_mu_phi_heatmaps(params_all['mu'][tf_name]['Crick Motif']['Watson Signal'], params_all['phi'][tf_name]['Crick Motif']['Watson Signal'], strand_label="Watson", tf_name=tf_name)
#         plot_mu_phi_heatmaps(params_all['mu'][tf_name]['Crick Motif']['Crick Signal'], params_all['phi'][tf_name]['Crick Motif']['Crick Signal'], strand_label="Crick", tf_name=tf_name)
    
    
#     # Handle TFs with < 50 sites as a combined group if needed
#     tf_counts_low = tf_counts.loc[tf_counts < 50]
#     if len(tf_counts_low) > 0:
#         # Combine all low-count TFs into one group
#         combined_tfs = list(tf_counts_low.index)
#         params_all['mu']['combined_low_count'] = {}
#         params_all['phi']['combined_low_count'] = {}
        
#         for motif_strand in ['+', '-']:
#             motif_name = 'Watson Motif' if motif_strand == '+' else 'Crick Motif'

#             if tech != "Fiber_seq":
#                 combined_params = compute_combined_DMSTFPhisMus(
#                     samfile, tfs, combined_tfs, motif_strand, fasta_file, fitdist, offset
#                 )
#             elif tech == "Fiber_seq":
#                 # For Fiber-seq, we need to handle the pileup data differently
#                 combined_params = compute_individual_Fiber_seq_TFPhisMus(
#                     modkitFile, tfs, combined_tfs, motif_strand, fasta_file, fitdist, offset
#                 )
            
            
#             params_all['mu']['combined_low_count'][motif_name] = {
#                 'Watson Signal': combined_params['mu']['watson'],
#                 'Crick Signal': combined_params['mu']['crick']
#             }
#             params_all['phi']['combined_low_count'][motif_name] = {
#                 'Watson Signal': combined_params['phi']['watson'],
#                 'Crick Signal': combined_params['phi']['crick']
#             }
    
#     if tech == "DMS_seq":
#         samfile.close()
#     return params_all

# def computeMNaseTFPhisMus(tech, bamFile, modkitFile, csvFile, tmpDir, fragRange, filename, offset=0):
#     fitdist = importr('fitdistrplus')
#     if tech != "Fiber_seq":
#         samfile = pysam.AlignmentFile(bamFile, "rb")
#     fasta_file = {}
#     for seq_record in SeqIO.parse("/home/rapiduser/programs/RoboCOP/analysis/inputs/SacCer3.fa", "fasta"):
#         fasta_file[seq_record.id] = seq_record.seq
#     tfs = pd.read_csv(csvFile, sep='\t', header=None)
#     tfs = tfs.rename(columns={0: 'chr', 1: 'start', 2: 'end', 3: 'tf_name', 4: 'score', 5: 'strand'})
#     tf_counts = tfs.groupby('tf_name')['chr'].count()
#     ind_tfs = list(tf_counts.loc[tf_counts >= 50].index)

#     params_all = {
#         'mu': {},
#         'phi': {}
#     }

#     for tf_name in ind_tfs:
#         # For Fiber_seq or DMS_seq, get counts for both Watson and Crick motif strands separately
#         if tech != "Fiber_seq":
#             watson_counts = compute_individual_DMSTFPhisMus(
#                 samfile, tfs, tf_name, '+', fasta_file, fitdist, offset
#             )
#             crick_counts = compute_individual_DMSTFPhisMus(
#                 samfile, tfs, tf_name, '-', fasta_file, fitdist, offset
#             )
#         else:
#             watson_counts = compute_individual_Fiber_seq_TFPhisMus(
#                 modkitFile, tfs, tf_name, '+', fasta_file, fitdist, offset
#             )
#             crick_counts = compute_individual_Fiber_seq_TFPhisMus(
#                 modkitFile, tfs, tf_name, '-', fasta_file, fitdist, offset
#             )

#         combined_counts = combine_motif_counts(watson_counts, crick_counts)

#         if combined_counts['num_sites'] == 0:
#             params = create_default_params_individual()
#         else:
#             params = fit_nb_parameters(combined_counts, combined_counts['tf_len'], combined_counts['num_sites'], fitdist)

#         params_all['mu'][tf_name] = {
#             'Watson Signal': params['mu']['watson'],
#             'Crick Signal': params['mu']['crick']
#         }
#         params_all['phi'][tf_name] = {
#             'Watson Signal': params['phi']['watson'],
#             'Crick Signal': params['phi']['crick']
#         }

#         # Optionally, plot heatmaps here
#         plot_mu_phi_heatmaps(params_all['mu'][tf_name]['Watson Signal'], params_all['phi'][tf_name]['Watson Signal'], strand_label="Watson", tf_name=tf_name)
#         plot_mu_phi_heatmaps(params_all['mu'][tf_name]['Crick Signal'], params_all['phi'][tf_name]['Crick Signal'], strand_label="Crick", tf_name=tf_name)

#     ### Handle combined low count TFs similarly with modification ###

#     tf_counts_low = tf_counts.loc[tf_counts < 50]
#     if len(tf_counts_low) > 0:
#         combined_tfs = list(tf_counts_low.index)
#         if tech != "Fiber_seq":
#             watson_counts = compute_combined_DMSTFPhisMus(  # you may need to modify this function similarly to return raw counts.
#                 samfile, tfs, combined_tfs, '+', fasta_file, fitdist, offset
#             )
#             crick_counts = compute_combined_DMSTFPhisMus(
#                 samfile, tfs, combined_tfs, '-', fasta_file, fitdist, offset
#             )
#         else:
#             # Use the new combined Fiber_seq function
#             watson_counts = compute_combined_Fiber_seq_TFPhisMus(
#                 modkitFile, tfs, combined_tfs, '+', fasta_file, fitdist, offset
#             )
#             crick_counts = compute_combined_Fiber_seq_TFPhisMus(
#                 modkitFile, tfs, combined_tfs, '-', fasta_file, fitdist, offset
#             )

#         combined_counts = combine_motif_counts(watson_counts, crick_counts)

#         if combined_counts['num_sites'] == 0:
#             params = create_default_params_individual()
#         else:
#             params = fit_nb_parameters(combined_counts, combined_counts['tf_len'], combined_counts['num_sites'], fitdist)

#         params_all['mu']['combined_low_count'] = {
#             'Watson Signal': params['mu']['watson'],
#             'Crick Signal': params['mu']['crick']
#         }
#         params_all['phi']['combined_low_count'] = {
#             'Watson Signal': params['phi']['watson'],
#             'Crick Signal': params['phi']['crick']
#         }

#     if tech == "DMS_seq":
#         samfile.close()

#     return params_all


# def compute_combined_DMSTFPhisMus(samfile, tfs_df, tf_names_list, motif_strand, fasta_file, fitdist, offset=0):
#     """
#     Compute combined counts for multiple low-count TFs (DMS_seq/Fiber_seq-like BAM input),
#     aggregating methylation fragment counts at each position and for each base on both 
#     signal strands, without fitting distributions inside this function.

#     Parameters:
#     -----------
#     samfile : pysam.AlignmentFile
#         Opened BAM file object for reading sequencing data
#     tfs_df : pd.DataFrame
#         DataFrame containing TF binding sites (columns: chr, start, end, tf_name, score, strand)
#     tf_names_list : list of str
#         List of TF names to combine (usually low-count TFs)
#     motif_strand : str
#         '+' for Watson motif strand, '-' for Crick motif strand
#     fasta_file : dict
#         Dictionary mapping chromosome names to Bio.SeqRecord.seq objects with reference genome
#     fitdist : rpy2 R package import
#         Imported fitdistrplus R package via rpy2 (not used here, but kept for consistency)
#     offset : int, optional
#         Position offset adjustment (default 0)

#     Returns:
#     --------
#     dict
#         Dictionary with structure:
#         {
#           'watson_signal': {base: list of counts},
#           'crick_signal': {base: list of counts},
#           'tf_len': int motif length (mode length across combined TFs),
#           'num_sites': int number of sites used
#         }
#     """
#     base_names = ['A', 'C', 'G', 'T']
#     signal_strand_names = ['watson', 'crick']

#     # Initialize aggregated counts as empty lists
#     tf_counts = {strand: {base: [] for base in base_names} for strand in signal_strand_names}

#     combined_df = tfs_df.loc[(tfs_df['tf_name'].isin(tf_names_list)) & (tfs_df['strand'] == motif_strand)]

#     if len(combined_df) == 0:
#         return create_default_params_individual()

#     # Determine most common TF length (mode)
#     tf_lengths = combined_df['end'] - combined_df['start']
#     tf_len = int(tf_lengths.mode().iloc[0])  # Most common length

#     # Iterate over each binding site, skip if length not equal to mode tf_len
#     for _, r1 in combined_df.iterrows():
#         site_len = r1['end'] - r1['start']
#         if site_len != tf_len:
#             continue

#         chrm = r1['chr']

#         # Initialize site counts with pseudocount 1 to avoid zeros
#         site_counts = {
#             strand: {base: [1]*tf_len for base in base_names} 
#             for strand in signal_strand_names
#         }

#         # Fetch reads overlapping this TF site
#         region = samfile.fetch(chrm, r1['start'] - 1, r1['end'] + 1)

#         for read in region:
#             if read.template_length == 0:
#                 continue

#             if read.template_length > 0:  # Watson signal strand
#                 frag_start = read.reference_start  # 0-based start of alignment
#                 if r1['start'] <= frag_start + 1 <= r1['end']:
#                     nucleotide = SeqFeature(FeatureLocation(frag_start, frag_start + 1)).extract(fasta_file[chrm])
#                     pos = (frag_start + 1) - r1['start']  # position relative to TF start
#                     if str(nucleotide) in base_names:
#                         site_counts['watson'][str(nucleotide)][pos] += 1

#             elif read.template_length < 0:  # Crick signal strand
#                 frag_end = read.reference_end  # 0-based exclusive end of alignment
#                 # Reference end is exclusive, so frag_end actually one past last ref covered base
#                 # Adjusting to 1-based coordinate internally consistent with TF site coordinates
#                 adjusted_frag_end = frag_end
#                 if r1['start'] <= adjusted_frag_end <= r1['end']:
#                     nucleotide = SeqFeature(FeatureLocation(adjusted_frag_end -1, adjusted_frag_end)).extract(fasta_file[chrm])
#                     pos = adjusted_frag_end - r1['start']  # position relative to TF start
#                     complement_map = {'A':'T', 'C':'G', 'G':'C', 'T':'A'}
#                     if str(nucleotide) in complement_map:
#                         site_counts['crick'][complement_map[str(nucleotide)]][pos] += 1

#         # Append site counts flattened by base and strand into aggregate lists
#         for strand in signal_strand_names:
#             for base in base_names:
#                 tf_counts[strand][base].extend(site_counts[strand][base])

#     return {
#         'watson_signal': tf_counts['watson'],
#         'crick_signal': tf_counts['crick'],
#         'tf_len': tf_len,
#         'num_sites': len(combined_df)
#     }

# def compute_individual_Fiber_seq_TFPhisMus(modkitFile, tfs_df, tf_name, motif_strand, fasta_file, fitdist, offset=0):
#     """
#     Compute negative binomial parameters for a single transcription factor on one motif strand.
#     Extracts methylation fragment counts at each nucleotide position, distinguishes Watson/Crick
#     signal strands, and fits negative binomial distributions across all binding sites.
    
#     Parameters:
#     -----------
#     modkitFile : modkit pileup file
#         dataframe of modkit file
#     tfs_df : pd.DataFrame
#         DataFrame containing TF binding site information with columns:
#         chr, start, end, tf_name, score, strand
#     tf_name : str
#         Name of the specific transcription factor to process
#     motif_strand : str
#         Motif orientation: '+' for Watson Motif, '-' for Crick Motif
#     fasta_file : dict
#         Dictionary mapping chromosome names to Bio.SeqRecord.seq objects
#         containing reference genome sequences
#     fitdist : rpy2 R package
#         R fitdistrplus package imported via rpy2 for negative binomial fitting
#     offset : int, optional
#         Position offset adjustment (default: 0, currently unused)
        
#     Returns:
#     --------
#     dict
#         Dictionary with structure:
#         {
#             'mu': {
#                 'watson': {'A': array, 'C': array, 'G': array, 'T': array},
#                 'crick': {'A': array, 'C': array, 'G': array, 'T': array}
#             },
#             'phi': {
#                 'watson': {'A': array, 'C': array, 'G': array, 'T': array},
#                 'crick': {'A': array, 'C': array, 'G': array, 'T': array}
#             }
#         }
        
#         Arrays contain fitted parameters for each position in the TF motif
#     """
#     # Initialize count arrays for each signal strand and base
#     base_names = ['A', 'C', 'G', 'T']
#     signal_strand_names = ['watson', 'crick']
    
#     tf_counts = {strand: {base: [] for base in base_names} for strand in signal_strand_names}
    
#     # Get all sites for this TF on the specified motif strand
#     one_tf_df = tfs_df.loc[(tfs_df['tf_name'] == tf_name) & (tfs_df['strand'] == motif_strand)]
    
#     if len(one_tf_df) == 0:
#         return create_default_params_individual()
    
#     # Get TF length (assuming consistent length for this TF)
#     tf_len = one_tf_df.iloc[0]['end'] - one_tf_df.iloc[0]['start']

#     modified_bases_df = pd.read_csv(modkitFile, sep='\t', header=None)
#     # Split the 9th column into multiple columns
#     split_columns = modified_bases_df[9].str.split(' ', expand=True)
#     split_columns.columns = [i for i in range(9,9+split_columns.shape[1])]

#     # Drop the original 9th column and concatenate the new columns back to the original DataFrame
#     modified_bases_df = pd.concat([modified_bases_df.drop(columns=[9]), split_columns], axis=1)
    
#     for i1, r1 in one_tf_df.iterrows():
#         chrm = r1['chr']
        
#         # Initialize counts for this TF site
#         site_counts = {
#             strand: {base: [1] * tf_len for base in base_names} 
#             for strand in signal_strand_names
#         }


#         # Filter for relevant rows based on chromosome and position
#         relevant_rows = modified_bases_df[
#             (modified_bases_df[0] == chrm) &
#             (modified_bases_df[1] < r1['end']) &
#             (modified_bases_df[2] > r1['start'])
#         ]

#         for _, row in relevant_rows.iterrows():
#             modified_base = row[3].upper()
#             strand_info = row[5]
#             count = int(row[11])
#             ## row[1] is bed file start coordinate which is 0 indexed. 
#             ##  r1['start'] is also from a bed file and is also 0 indexed
#             pos = row[1] - r1['start']

#             if strand_info == '+':
#                 if modified_base in base_names:
#                     site_counts['watson'][modified_base][pos] += count
#             elif strand_info == '-':
#                 if modified_base in base_names:
#                     site_counts['crick'][modified_base][pos] += count
        
#         # Accumulate counts across all sites for this TF
#         for strand in signal_strand_names:
#             for base in base_names:
#                 tf_counts[strand][base].extend(site_counts[strand][base])
    
#     # Fit negative binomial parameters
#     return {
#         'watson_signal': tf_counts['watson'],
#         'crick_signal': tf_counts['crick'],
#         'tf_len': tf_len,
#         'num_sites': len(one_tf_df)
#     }