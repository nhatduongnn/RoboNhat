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
import pickle
import pyranges as pr
sys.path.insert(0, '../pkg/')
sys.path.insert(0, '/home/rapiduser/programs/RoboCOP/pkg/')
#from robocop.utils.parameters import computeMNaseTFPhisMus

from collections import defaultdict
from Bio.Seq import reverse_complement


def compute_Fiber_seq_TFPhisMus(
    tech,
    bamFile,
    modkitFile,
    csvFile,
    tmpDir,
    fragRange,
    filename,
    offset=0,
    dist="nbinom",           # "nbinom" (default) or "binomial"
    successes_col=11,        # Only used when dist="binomial" for Fiber_seq
    trials_col=9            # Only used when dist="binomial" for Fiber_seq
):
    fitdist = importr('fitdistrplus')
    if tech != "Fiber_seq":
        samfile = pysam.AlignmentFile(bamFile, "rb")

    fasta_file = {}
    for seq_record in SeqIO.parse("/home/rapiduser/programs/RoboCOP/analysis/inputs/SacCer3.fa", "fasta"):
        fasta_file[seq_record.id] = seq_record.seq

    tfs = pd.read_csv(csvFile, sep='\t')
    tfs = tfs.rename(columns={'TF': 'tf_name'})
    tf_counts = tfs.groupby('tf_name')['chr'].count()

    ## Get a list of individual TFs with at least 50 sites, to estimate parameters for them
    ind_tfs = list(tf_counts.loc[tf_counts >= 50].index)

    # Params container supports both NB and binomial
    params_all = {'mu': {}, 'phi': {}, 'p': {}}

    # Load Modkit file once for Fiber_seq
    modified_bases_df = None
    if tech == "Fiber_seq":
        modified_bases_df = pd.read_csv(modkitFile, sep='\t', header=None)
        split_columns = modified_bases_df[9].str.split(' ', expand=True)
        split_columns.columns = [i for i in range(9, 9 + split_columns.shape[1])]
        modified_bases_df = pd.concat([modified_bases_df.drop(columns=[9]), split_columns], axis=1)

    for tf_name in ind_tfs:
        if tech != "Fiber_seq":
            # DMS path (NB only)
            watson_counts = compute_individual_DMSTFPhisMus(
                samfile, tfs, tf_name, '+', fasta_file, fitdist, offset
            )
            crick_counts = compute_individual_DMSTFPhisMus(
                samfile, tfs, tf_name, '-', fasta_file, fitdist, offset
            )
            combined_counts = combine_motif_counts(watson_counts, crick_counts)
            if combined_counts['num_sites'] == 0:
                params = create_default_params_individual()
            else:
                params = fit_nb_parameters(combined_counts, combined_counts['tf_len'], combined_counts['num_sites'], fitdist)
            params_all['mu'][tf_name] = {
                'watson_signal': params['mu']['watson_signal'],
                'crick_signal': params['mu']['crick_signal']
            }
            params_all['phi'][tf_name] = {
                'watson_signal': params['phi']['watson_signal'],
                'crick_signal': params['phi']['crick_signal']
            }
            # Optional plotting
            # plot_mu_phi_heatmaps(params_all['mu'][tf_name]['Watson Signal'], params_all['phi'][tf_name]['Watson Signal'], strand_label="Watson", tf_name=tf_name)
            # plot_mu_phi_heatmaps(params_all['mu'][tf_name]['Crick Signal'], params_all['phi'][tf_name]['Crick Signal'], strand_label="Crick", tf_name=tf_name)
        else:
            # Fiber_seq path: choose distribution
            if dist == "nbinom":
                watson_counts = compute_individual_Fiber_seq_TFPhisMus(
                    modified_bases_df, tfs, tf_name, '+', fasta_file, fitdist, offset
                )
                crick_counts = compute_individual_Fiber_seq_TFPhisMus(
                    modified_bases_df, tfs, tf_name, '-', fasta_file, fitdist, offset
                )
                combined_counts = combine_motif_counts(watson_counts, crick_counts)
                if combined_counts['num_sites'] == 0:
                    params = create_default_params_individual()
                else:
                    params = fit_nb_parameters(combined_counts, combined_counts['tf_len'], combined_counts['num_sites'], fitdist)
                params_all['mu'][tf_name] = {
                    'watson_signal': params['mu']['watson_signal'],
                    'crick_signal': params['mu']['crick_signal']
                }
                params_all['phi'][tf_name] = {
                    'watson_signal': params['phi']['watson_signal'],
                    'crick_signal': params['phi']['crick_signal']
                }
                # Optional plotting
                plot_mu_phi_heatmaps(params_all['mu'][tf_name]['watson_signal'], params_all['phi'][tf_name]['watson_signal'], strand_label="Watson", tf_name=tf_name)
                plot_mu_phi_heatmaps(params_all['mu'][tf_name]['crick_signal'], params_all['phi'][tf_name]['crick_signal'], strand_label="Crick", tf_name=tf_name)
            elif dist == "binomial":
                watson_counts = compute_individual_Fiber_seq_TF_binom(
                    modified_bases_df, tfs, tf_name, '+', fasta_file, offset, successes_col=successes_col, trials_col=trials_col
                )
                crick_counts = compute_individual_Fiber_seq_TF_binom(
                    modified_bases_df, tfs, tf_name, '-', fasta_file, offset, successes_col=successes_col, trials_col=trials_col
                )
                combined_counts = combine_motif_counts_binom(watson_counts, crick_counts)
                if combined_counts['num_sites'] == 0:
                    params = create_default_params_binomial()
                else:
                    params = fit_binomial_parameters(combined_counts, combined_counts['tf_len'], combined_counts['num_sites'])
                params_all['p'][tf_name] = {
                    'watson_signal': params['p']['watson_signal'],
                    'crick_signal': params['p']['crick_signal']
                }
                # Optional plotting
                plot_p_heatmaps(params_all['p'][tf_name]['watson_signal'], strand_label="Watson", tf_name=tf_name)
                plot_p_heatmaps(params_all['p'][tf_name]['crick_signal'], strand_label="Crick", tf_name=tf_name)
            else:
                raise ValueError(f"Unsupported dist: {dist}. Use 'nbinom' or 'binomial'.")

    # Combined low-count TFs
    tf_counts_low = tf_counts.loc[tf_counts < 50]
    if len(tf_counts_low) > 0:
        combined_tfs = list(tf_counts_low.index)
        if tech != "Fiber_seq":
            # DMS path (NB only)
            watson_counts = compute_combined_DMSTFPhisMus(
                samfile, tfs, combined_tfs, '+', fasta_file, fitdist, offset
            )
            crick_counts = compute_combined_DMSTFPhisMus(
                samfile, tfs, combined_tfs, '-', fasta_file, fitdist, offset
            )
            combined_counts = combine_motif_counts(watson_counts, crick_counts)
            if combined_counts['num_sites'] == 0:
                params = create_default_params_individual()
            else:
                params = fit_nb_parameters(combined_counts, combined_counts['tf_len'], combined_counts['num_sites'], fitdist)
            params_all['mu']['combined_low_count'] = {
                'watson_signal': params['mu']['watson_signal'],
                'crick_signal': params['mu']['crick_signal']
            }
            params_all['phi']['combined_low_count'] = {
                'watson_signal': params['phi']['watson_signal'],
                'crick_signal': params['phi']['crick_signal']
            }
        else:
            if dist == "nbinom":
                watson_counts = compute_combined_Fiber_seq_TFPhisMus(
                    modkitFile, tfs, combined_tfs, '+', fasta_file, fitdist, offset
                )
                crick_counts = compute_combined_Fiber_seq_TFPhisMus(
                    modkitFile, tfs, combined_tfs, '-', fasta_file, fitdist, offset
                )
                combined_counts = combine_motif_counts(watson_counts, crick_counts)
                if combined_counts['num_sites'] == 0:
                    params = create_default_params_individual()
                else:
                    params = fit_nb_parameters(combined_counts, combined_counts['tf_len'], combined_counts['num_sites'], fitdist)
                params_all['mu']['combined_low_count'] = {
                    'watson_signal': params['mu']['watson_signal'],
                    'crick_signal': params['mu']['crick_signal']
                }
                params_all['phi']['combined_low_count'] = {
                    'watson_signal': params['phi']['watson_signal'],
                    'crick_signal': params['phi']['crick_signal']
                }
            elif dist == "binomial":
                # Use DF already loaded for Fiber_seq binomial combined path
                watson_counts = compute_combined_Fiber_seq_TF_binom(
                    modified_bases_df, tfs, combined_tfs, '+', fasta_file, offset, successes_col=successes_col, trials_col=trials_col
                )
                crick_counts = compute_combined_Fiber_seq_TF_binom(
                    modified_bases_df, tfs, combined_tfs, '-', fasta_file, offset, successes_col=successes_col, trials_col=trials_col
                )
                combined_counts = combine_motif_counts_binom(watson_counts, crick_counts)
                if combined_counts['num_sites'] == 0:
                    params = create_default_params_binomial()
                else:
                    params = fit_binomial_parameters(combined_counts, combined_counts['tf_len'], combined_counts['num_sites'])
                params_all['p']['combined_low_count'] = {
                    'watson_signal': params['p']['watson_signal'],
                    'crick_signal': params['p']['crick_signal']
                }
    else:
        # No low-count TFs; still populate defaults
        if dist == "nbinom":
            params = create_default_params_individual()
            params_all['mu']['combined_low_count'] = {
                'watson_signal': params['mu']['watson_signal'],
                'crick_signal': params['mu']['crick_signal']
            }
            params_all['phi']['combined_low_count'] = {
                'watson_signal': params['phi']['watson_signal'],
                'crick_signal': params['phi']['crick_signal']
            }
        elif dist == "binomial":
            params = create_default_params_binomial()
            params_all['p']['combined_low_count'] = {
                'watson_signal': params['p']['watson_signal'],
                'crick_signal': params['p']['crick_signal']
            }

    if tech != "Fiber_seq":
        samfile.close()
    return params_all


def compute_individual_DMSTFPhisMus(samfile, tfs_df, tf_name, motif_strand, fasta_file, fitdist, offset=0):
    base_names = ['A', 'C', 'G', 'T']
    signal_strand_names = ['watson', 'crick']
    tf_counts = {strand: {base: [] for base in base_names} for strand in signal_strand_names}

    # FIXED comparisons
    one_tf_df = tfs_df.loc[(tfs_df['tf_name'] == tf_name) & (tfs_df['strand'] == motif_strand)]
    if len(one_tf_df) == 0:
        return create_default_params_individual()

    tf_len = one_tf_df.iloc[0]['end'] - one_tf_df.iloc[0]['start']

    for _, r1 in one_tf_df.iterrows():
        chrm = r1['chr']
        site_counts = {
            strand: {base: [1] * tf_len for base in base_names}
            for strand in signal_strand_names
        }
        region = samfile.fetch(chrm, r1['start'] - 1, r1['end'] + 1)
        for read in region:
            if read.template_length == 0:
                continue
            if read.template_length > 0:  # Watson signal
                frag_start = read.reference_start  # 0-based
                frag_pos_1based = frag_start + 1
                if r1['start'] <= frag_pos_1based <= r1['end']:
                    nucleotide = SeqFeature(FeatureLocation(frag_start, frag_start + 1)).extract(fasta_file[chrm])
                    pos = frag_pos_1based - r1['start']
                    if 0 <= pos < tf_len and str(nucleotide) in base_names:
                        site_counts['watson'][str(nucleotide)][pos] += 1
            elif read.template_length < 0:  # Crick signal
                frag_end_1based = read.reference_end + 1
                frag_end_0based = frag_end_1based - 1
                if r1['start'] <= frag_end_1based <= r1['end']:
                    nucleotide = SeqFeature(FeatureLocation(frag_end_0based, frag_end_0based + 1)).extract(fasta_file[chrm])
                    pos = frag_end_1based - r1['start']
                    complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
                    if 0 <= pos < tf_len and str(nucleotide) in complement_map:
                        site_counts['crick'][complement_map[str(nucleotide)]][pos] += 1
        for strand in signal_strand_names:
            for base in base_names:
                tf_counts[strand][base].extend(site_counts[strand][base])

    return {
        'watson_signal': tf_counts['watson'],
        'crick_signal': tf_counts['crick'],
        'tf_len': tf_len,
        'num_sites': len(one_tf_df)
    }


def compute_individual_Fiber_seq_TFPhisMus(modified_bases_df, tfs_df, tf_name, motif_strand, fasta_file, fitdist, offset=0):
    base_names = ['A', 'C', 'G', 'T']
    signal_strand_names = ['watson', 'crick']
    tf_counts = {strand: {base: [] for base in base_names} for strand in signal_strand_names}

    # FIXED comparisons
    one_tf_df = tfs_df.loc[(tfs_df['tf_name'] == tf_name) & (tfs_df['strand'] == motif_strand)]
    if len(one_tf_df) == 0:
        return create_default_params_individual()

    tf_len = one_tf_df.iloc[0]['end'] - one_tf_df.iloc[0]['start']

    for _, r1 in one_tf_df.iterrows():
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
            modified_base = str(row[3]).upper()
            strand_info = row[5]
            count = int(row[11])  # Assumed to be modified count or signal count for NB
            pos = int(row[1]) - r1['start']
            if 0 <= pos < tf_len:
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
    base_names = ['A', 'C', 'G', 'T']
    signal_strand_names = ['watson', 'crick']
    tf_counts = {strand: {base: [] for base in base_names} for strand in signal_strand_names}

    combined_df = tfs_df.loc[(tfs_df['tf_name'].isin(tf_names_list)) & (tfs_df['strand'] == motif_strand)]
    if len(combined_df) == 0:
        return create_default_params_individual()

    for _, r1 in combined_df.iterrows():
        chrm = r1['chr']
        start = r1['start']
        end = r1['end']
        tf_len = end - start
        
        # initialize site counts for this TF site
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

        # dump *all bases* of this TF site into the global accumulator
        for strand in signal_strand_names:
            for base in base_names:
                tf_counts[strand][base].extend(site_counts[strand][base])

    # we no longer return a `tf_len`, since it's site-specific and irrelevant
    return {
        'watson_signal': tf_counts['watson'],
        'crick_signal': tf_counts['crick'],
        'tf_len': 1, # Because, downstream, we want to combine all of these count data regardless of how many bp per tf, and estimate mu and phi
        'num_sites': len(tf_counts['watson']['A'])
    }


def compute_individual_Fiber_seq_TF_binom(
    modified_bases_df,
    tfs_df,
    tf_name,
    motif_strand,
    fasta_file,
    offset=0,
    successes_col=11,
    trials_col=12
):
    """
    Compute binomial counts for a single TF on one motif strand for Fiber_seq data.
    Returns successes and trials for each base and signal strand.
    Successes are taken from `successes_col` and trials from `trials_col` in modified_bases_df.
    """
    base_names = ['A', 'C', 'G', 'T']
    signal_strand_names = ['watson', 'crick']
    tf_counts = {
        'watson': {'successes': {b: [] for b in base_names}, 'trials': {b: [] for b in base_names}},
        'crick':  {'successes': {b: [] for b in base_names}, 'trials': {b: [] for b in base_names}}
    }

    one_tf_df = tfs_df.loc[(tfs_df['tf_name'] == tf_name) & (tfs_df['strand'] == motif_strand)]
    if len(one_tf_df) == 0:
        return {
            'watson_signal': {'successes': {b: [] for b in base_names}, 'trials': {b: [] for b in base_names}},
            'crick_signal':  {'successes': {b: [] for b in base_names}, 'trials': {b: [] for b in base_names}},
            'tf_len': 13,
            'num_sites': 0
        }

    tf_len = one_tf_df.iloc[0]['end'] - one_tf_df.iloc[0]['start']

    if successes_col not in modified_bases_df.columns or trials_col not in modified_bases_df.columns:
        raise ValueError(f"successes_col ({successes_col}) or trials_col ({trials_col}) not found in modified_bases_df columns.")

    for _, r1 in one_tf_df.iterrows():
        chrm = r1['chr']
        site_successes = {strand: {b: [0] * tf_len for b in base_names} for strand in signal_strand_names}
        site_trials    = {strand: {b: [0] * tf_len for b in base_names} for strand in signal_strand_names}

        relevant_rows = modified_bases_df[
            (modified_bases_df[0] == chrm) &
            (modified_bases_df[1] < r1['end']) &
            (modified_bases_df[2] > r1['start'])
        ]
        for _, row in relevant_rows.iterrows():
            modified_base = str(row[3]).upper()
            strand_info = row[5]
            succ = int(row[successes_col])
            tot  = int(row[trials_col])
            pos = int(row[1]) - r1['start']
            if 0 <= pos < tf_len and modified_base in base_names:
                if strand_info == '+':
                    site_successes['watson'][modified_base][pos] += succ
                    site_trials['watson'][modified_base][pos]    += tot
                elif strand_info == '-':
                    site_successes['crick'][modified_base][pos] += succ
                    site_trials['crick'][modified_base][pos]    += tot

        for strand in signal_strand_names:
            for b in base_names:
                tf_counts[strand]['successes'][b].extend(site_successes[strand][b])
                tf_counts[strand]['trials'][b].extend(site_trials[strand][b])

    return {
        'watson_signal': tf_counts['watson'],
        'crick_signal':  tf_counts['crick'],
        'tf_len': tf_len,
        'num_sites': len(one_tf_df)
    }


def compute_combined_Fiber_seq_TF_binom(
    modified_bases_df,
    tfs_df,
    tf_names_list,
    motif_strand,
    fasta_file,
    offset=0,
    successes_col=11,
    trials_col=9
):
    base_names = ['A', 'C', 'G', 'T']
    signal_strand_names = ['watson', 'crick']
    tf_counts = {
        'watson': {'successes': {b: [] for b in base_names}, 'trials': {b: [] for b in base_names}},
        'crick':  {'successes': {b: [] for b in base_names}, 'trials': {b: [] for b in base_names}}
    }

    combined_df = tfs_df.loc[(tfs_df['tf_name'].isin(tf_names_list)) & (tfs_df['strand'] == motif_strand)]
    if len(combined_df) == 0:
        return {
            'watson': {'successes': {b: [] for b in base_names}, 'trials': {b: [] for b in base_names}},
            'crick':  {'successes': {b: [] for b in base_names}, 'trials': {b: [] for b in base_names}},
            'tf_len': 1,     # collapsed mode -> always 1
            'num_sites': 0
        }

    if successes_col not in modified_bases_df.columns or trials_col not in modified_bases_df.columns:
        raise ValueError(f"successes_col ({successes_col}) or trials_col ({trials_col}) not found in modified_bases_df columns.")

    for _, r1 in combined_df.iterrows():
        chrm = r1['chr']
        start = r1['start']
        end   = r1['end']
        tf_len = end - start

        site_successes = {strand: {b: [0] * tf_len for b in base_names} for strand in signal_strand_names}
        site_trials    = {strand: {b: [0] * tf_len for b in base_names} for strand in signal_strand_names}

        relevant_rows = modified_bases_df[
            (modified_bases_df[0] == chrm) &
            (modified_bases_df[1] < end) &
            (modified_bases_df[2] > start)
        ]

        for _, row in relevant_rows.iterrows():
            modified_base = str(row[3]).upper()
            strand_info   = row[5]
            succ = int(row[successes_col])
            tot  = int(row[trials_col])
            pos  = int(row[1]) - start
            if 0 <= pos < tf_len and modified_base in base_names:
                if strand_info == '+':
                    site_successes['watson'][modified_base][pos] += succ
                    site_trials['watson'][modified_base][pos]    += tot
                elif strand_info == '-':
                    site_successes['crick'][modified_base][pos] += succ
                    site_trials['crick'][modified_base][pos]    += tot

        # Flatten to global accumulators
        for strand in signal_strand_names:
            for b in base_names:
                tf_counts[strand]['successes'][b].extend(site_successes[strand][b])
                tf_counts[strand]['trials'][b].extend(site_trials[strand][b])

    return {
        'watson_signal': tf_counts['watson'],
        'crick_signal':  tf_counts['crick'],
        'tf_len': 1,                             # <-- force to 1 as before
        'num_sites': len(tf_counts['watson']['successes']['A'])
    }


def combine_motif_counts_binom(watson_counts, crick_counts):
    """
    Combine successes and trials across motif orientations to align positions,
    using consistent keys: 'watson_signal' and 'crick_signal'.

    Input:
        watson_counts, crick_counts: dicts with keys:
            - 'watson_signal': {'successes': {A,C,G,T}, 'trials': {A,C,G,T}}
            - 'crick_signal':  same as above
            - 'tf_len': motif length
            - 'num_sites': number of sites

    Output:
        combined dict with same structure:
            - 'watson_signal': successes/trials per base, aligned
            - 'crick_signal': successes/trials per base, aligned
            - 'tf_len': motif length
            - 'num_sites': combined number of sites
    """
    base_names = ['A', 'C', 'G', 'T']
    tf_len = watson_counts['tf_len']
    num_sites_w = watson_counts['num_sites']
    num_sites_c = crick_counts['num_sites']

    def reshape_counts_binom(counts, num_sites):
        reshaped = {
            'watson_signal': {'successes': {}, 'trials': {}},
            'crick_signal':  {'successes': {}, 'trials': {}}
        }
        for signal_key in ['watson_signal', 'crick_signal']:
            for base in base_names:
                succ = np.array(counts[signal_key]['successes'][base])
                tri  = np.array(counts[signal_key]['trials'][base])
                if num_sites == 0:
                    succ_reshaped = np.empty((0, tf_len), dtype=int)
                    tri_reshaped  = np.empty((0, tf_len), dtype=int)
                else:
                    succ_reshaped = succ.reshape((num_sites, tf_len))
                    tri_reshaped  = tri.reshape((num_sites, tf_len))
                reshaped[signal_key]['successes'][base] = succ_reshaped
                reshaped[signal_key]['trials'][base]    = tri_reshaped
        return reshaped

    # reshape watson and crick motif counts individually
    w_rs = reshape_counts_binom(watson_counts, num_sites_w)
    c_rs = reshape_counts_binom(crick_counts, num_sites_c)

    # combined output structure
    combined = {
        'watson_signal': {'successes': {}, 'trials': {}},
        'crick_signal':  {'successes': {}, 'trials': {}},
        'tf_len': tf_len,
        'num_sites': num_sites_w + num_sites_c
    }

    # edge case: no motifs
    if (num_sites_w + num_sites_c) == 0:
        for base in base_names:
            combined['watson_signal']['successes'][base] = np.empty((0, tf_len), dtype=int)
            combined['watson_signal']['trials'][base]    = np.empty((0, tf_len), dtype=int)
            combined['crick_signal']['successes'][base]  = np.empty((0, tf_len), dtype=int)
            combined['crick_signal']['trials'][base]     = np.empty((0, tf_len), dtype=int)
        return combined

    # Combine by aligning Watson motif’s Watson signal with reversed Crick motif’s Crick signal
    # and Watson motif’s Crick signal with reversed Crick motif’s Watson signal
    for base in base_names:
        combined['watson_signal']['successes'][base] = np.vstack((
            w_rs['watson_signal']['successes'][base],
            c_rs['crick_signal']['successes'][base][:, ::-1]
        ))
        combined['watson_signal']['trials'][base] = np.vstack((
            w_rs['watson_signal']['trials'][base],
            c_rs['crick_signal']['trials'][base][:, ::-1]
        ))

        combined['crick_signal']['successes'][base] = np.vstack((
            w_rs['crick_signal']['successes'][base],
            c_rs['watson_signal']['successes'][base][:, ::-1]
        ))
        combined['crick_signal']['trials'][base] = np.vstack((
            w_rs['crick_signal']['trials'][base],
            c_rs['watson_signal']['trials'][base][:, ::-1]
        ))

    return combined


def fit_binomial_parameters(tf_counts_combined, tf_len, num_sites):
    """
    Fit binomial parameter p per position, strand, base:
      p_hat = sum(successes) / sum(trials) across all sites for that position.
    Returns dict with arrays of p for each base, strand.
    """
    base_names = ['A', 'C', 'G', 'T']
    strand_names = ['watson_signal','crick_signal']
    params = {
        'p': {
            'watson_signal': {b: np.zeros(tf_len) for b in base_names},
            'crick_signal':  {b: np.zeros(tf_len) for b in base_names}
        }
    }

    for strand in strand_names:
        for b in base_names:
            succ_arr = tf_counts_combined[strand]['successes'][b]  # shape: (N, tf_len)
            tri_arr  = tf_counts_combined[strand]['trials'][b]     # shape: (N, tf_len)
            # Guard for empty
            if succ_arr.size == 0 or tri_arr.size == 0:
                params['p'][strand][b] = np.full(tf_len, 0.000)
                continue
            # Sum across sites per position
            succ_sum = succ_arr.sum(axis=0)
            tri_sum  = tri_arr.sum(axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                p_hat = np.where(tri_sum > 0, succ_sum / tri_sum, 0.000)
            params['p'][strand][b] = p_hat
    return params


def create_default_params_binomial(tf_len=1, default_p=0.22):
    base_names = ['A', 'C', 'G', 'T']
    strand_names = ['Watson Signal', 'Crick Signal']
    return {
        'p': {strand: {base: np.full(tf_len, default_p) for base in base_names} for strand in strand_names}
    }


def fit_nb_parameters(tf_counts, tf_len, num_sites, fitdist):
    base_names = ['A', 'C', 'G', 'T']
    strand_names = ['watson_signal', 'crick_signal']
    params = {
        'mu': {strand: {base: np.zeros(tf_len) for base in base_names} for strand in strand_names},
        'phi': {strand: {base: np.zeros(tf_len) for base in base_names} for strand in strand_names}
    }
    try:
        f = io.StringIO()
        with redirect_stdout(f):
            for strand in strand_names:
                for base in base_names:
                    counts_array = tf_counts[strand][base]
                    for pos in range(tf_len):
                        pos_counts = counts_array[:, pos]
                        nb_fit = fitdist.fitdist(vectors.IntVector(pos_counts), 'nbinom', method="mle")
                        estimates = nb_fit.rx2("estimate")
                        params['mu'][strand][base][pos] = estimates.rx2("mu")[0]
                        params['phi'][strand][base][pos] = estimates.rx2("size")[0]
    except Exception as e:
        print(f"Error fitting parameters: {e}")
        return create_default_params_individual()
    return params


def combine_motif_counts(watson_counts, crick_counts):
    base_names = ['A', 'C', 'G', 'T']
    tf_len = watson_counts['tf_len']
    num_sites_watson = watson_counts['num_sites']
    num_sites_crick = crick_counts['num_sites']
    combined = {'watson_signal': {b: None for b in base_names},
                'crick_signal': {b: None for b in base_names},
                'tf_len': tf_len,
                'num_sites': num_sites_watson + num_sites_crick}
    def reshape_counts(counts, num_sites):
        reshaped = {'Watson Signal': {}, 'Crick Signal': {}}
        for signal in ['watson_signal', 'crick_signal']:
            key = signal
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

    for base in base_names:
        combined['Watson Signal'][base] = np.vstack((
            watson_reshaped['watson_signal'][base],
            crick_reshaped['crick_signal'][base][:, ::-1]
        )) if (num_sites_watson + num_sites_crick) > 0 else np.empty((0, tf_len), dtype=int)

        combined['Crick Signal'][base] = np.vstack((
            watson_reshaped['crick_signal'][base],
            crick_reshaped['watson_signal'][base][:, ::-1]
        )) if (num_sites_watson + num_sites_crick) > 0 else np.empty((0, tf_len), dtype=int)
    return combined


def create_default_params_individual():
    import numpy as np
    base_names = ['A', 'C', 'G', 'T']
    strand_names = ['watson', 'crick']
    tf_len = 1
    return {
        'mu': {strand: {base: np.full(tf_len, 0.002) for base in base_names} for strand in strand_names},
        'phi': {strand: {base: np.full(tf_len, 100) for base in base_names} for strand in strand_names}
    }


def plot_p_heatmaps(p_dict, strand_label, tf_name):
    """
    Plot heatmap of binomial p for a given strand.
    p_dict: dict of {base: np.array of p values}, keys A,C,G,T.
    """
    bases = ["A", "C", "G", "T"]
    motif_len = len(next(iter(p_dict.values())))
    num_bases = len(bases)
    p_matrix = np.vstack([p_dict[base] for base in bases])
    plt.figure(figsize=(6, 4))
    im = plt.imshow(p_matrix, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    plt.title(f"{tf_name} - {strand_label} - Binomial p")
    plt.yticks(range(num_bases), bases)
    plt.xlabel("Motif Position")
    plt.ylabel("Base")
    plt.colorbar(im, label="p")
    for i in range(num_bases):
        for j in range(motif_len):
            val = p_matrix[i, j]
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=6)
    plt.tight_layout()
    plt.show()

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



def compute_Fiber_seq_background(
    modified_bases_df,
    segments,
    fasta_file,
    dist="nbinom",        # "nbinom" (default) or "binomial"
    offset=0,
    successes_col=11,     # for binomial Fiber-seq
    trials_col=9,
    fitdist=None
):
    """
    Compute background per-base Fiber-seq parameters across arbitrary genome segments.
    Vectorized + pyranges-based interval overlap (fast).
    """

    base_names = ['A', 'C', 'G', 'T']
    strand_map = {'+': 'watson_signal', '-': 'crick_signal'}

    if not segments:
        return (
            create_default_params_individual()
            if dist == "nbinom"
            else create_default_params_binomial()
        )

    # ----------------------------------------------------------------------
    # 1) Convert pileup -> PyRanges
    pile_df = modified_bases_df.rename(
        columns={0: "Chromosome", 1: "Start", 2: "End", 3: "Base", 5: "Strand"}
    )
    pile_pr = pr.PyRanges(pile_df)

    # 2) Convert segments -> PyRanges
    seg_df = pd.DataFrame(segments).rename(
        columns={"chrm": "Chromosome", "start": "Start", "stop": "End"}
    )
    seg_pr = pr.PyRanges(seg_df)

    # 3) Interval join: select only rows overlapping any segment
    relevant_rows = pile_pr.join(seg_pr).df
    # relevant_rows has all original pileup columns, indexed properly

    # ----------------------------------------------------------------------
    if dist == "binomial":
        bg_counts = {
            'watson_signal': {'successes': {b: [] for b in base_names}, 'trials': {b: [] for b in base_names}},
            'crick_signal':  {'successes': {b: [] for b in base_names}, 'trials': {b: [] for b in base_names}}
        }

        for base in base_names:
            base_rows = relevant_rows[relevant_rows["Base"].str.upper() == base]
            for strand_symbol, strand_label in strand_map.items():
                s_rows = base_rows[base_rows["Strand"] == strand_symbol]
                bg_counts[strand_label]["successes"][base] = s_rows[successes_col].to_numpy()
                bg_counts[strand_label]["trials"][base]    = s_rows[trials_col].to_numpy()

        combined_counts = {
            "watson_signal": {
                "successes": {b: bg_counts['watson_signal']['successes'][b].reshape(-1,1) for b in base_names},
                "trials":    {b: bg_counts['watson_signal']['trials'][b].reshape(-1,1)    for b in base_names},
            },
            "crick_signal": {
                "successes": {b: bg_counts['crick_signal']['successes'][b].reshape(-1,1) for b in base_names},
                "trials":    {b: bg_counts['crick_signal']['trials'][b].reshape(-1,1)    for b in base_names},
            },
            "tf_len": 1,
            "num_sites": len(segments),
        }
        return fit_binomial_parameters(combined_counts, 1, len(segments))

    # ----------------------------------------------------------------------
    elif dist == "nbinom":
        bg_counts = {
            "watson_signal": {b: [] for b in base_names},
            "crick_signal":  {b: [] for b in base_names},
            "tf_len": 1,
            "num_sites": len(segments),
        }

        for base in base_names:
            base_rows = relevant_rows[relevant_rows["Base"].str.upper() == base]
            for strand_symbol, strand_label in strand_map.items():
                s_rows = base_rows[base_rows["Strand"] == strand_symbol]
                bg_counts[f"{strand_label}_signal"][base] = s_rows[successes_col].to_numpy()

        combined_counts = {
            "watson_signal": {b: np.array(bg_counts["watson_signal"][b]).reshape(-1,1) for b in base_names},
            "crick_signal":  {b: np.array(bg_counts["crick_signal"][b]).reshape(-1,1)  for b in base_names},
            "tf_len": 1,
            "num_sites": len(segments),
        }
        if bg_counts["num_sites"] == 0:
            return create_default_params_individual()
        return fit_nb_parameters(combined_counts, 1, len(segments), fitdist)

    else:
        raise ValueError(f"Unsupported dist={dist}, must be 'nbinom' or 'binomial'")


def computeLinkers(nucFile):
    """
    get linker regions from file with nucleosome dyads
    """
    segments = []

    nucs = pd.read_csv(nucFile, sep = '\t', header = None)
    nucs['dyad'] = (nucs[1] + nucs[2])/2
    nucs['dyad'] = nucs['dyad'].astype(int)
    nucs = nucs.rename(columns = {0: 'chr'})

    for i, r in nucs.iterrows():
        if "micron" in r['chr']: continue
        chrm = r['chr']
        if int(r['dyad']) - 73 - 15 > 0:
            segments.append({"chrm": chrm, "start": int(r['dyad']) - 73 - 15, "stop": int(r['dyad']) - 73})
            segments.append({"chrm": chrm, "start": int(r['dyad']) + 73, "stop": int(r['dyad']) + 73 + 15})
    return segments


def compute_Fiber_seq_nucleosome(
    modified_bases_df,
    nucleosome_file,
    dist="nbinom",        # "nbinom" or "binomial"
    offset=0,
    successes_col=11,
    trials_col=9,
    fitdist=None
):
    """
    Compute per-base Fiber-seq-style parameters across 147bp nucleosomes.
    Vectorized: uses PyRanges join and computes relative positions directly.
    """

    base_names  = ["A", "C", "G", "T"]
    strand_map  = {"+": "watson_signal", "-": "crick_signal"}
    nuc_len     = 147

    # -------------------------------
    # 1) Load nucleosomes -> 147bp windows
    nucs = pd.read_csv(nucleosome_file, sep="\t", header=None)
    nucs["dyad"]    = ((nucs[1] + nucs[2]) // 2).astype(int)
    nucs = nucs.rename(columns={0: "Chromosome"})
    nucs["Start"]   = nucs["dyad"] - 73
    nucs["End"]     = nucs["dyad"] + 74
    nuc_pr = pr.PyRanges(nucs[["Chromosome", "Start", "End"]])

    num_sites = len(nuc_pr)
    if num_sites == 0:
        return create_default_params_individual() if dist=="nbinom" else create_default_params_binomial()

    # -------------------------------
    # 2) Pileup PyRanges
    pile_df = modified_bases_df.rename(
        columns={0: "Chromosome", 1: "Start", 2: "End", 3: "Base", 5: "Strand"}
    )
    pile_pr = pr.PyRanges(pile_df)

    # -------------------------------
    # 3) Vectorized interval join
    joined = pile_pr.join(nuc_pr).df
    if joined.empty:
        return create_default_params_individual() if dist=="nbinom" else create_default_params_binomial()

    # Rename PyRanges-suffixed columns for clarity:
    # pile:  Chromosome, Start, End, Base, Strand, ...
    # nuc:   Start_b, End_b
    joined = joined.rename(columns={"Start_b":"NucStart", "End_b":"NucEnd"})

    # Compute relative pos within nucleosome (0..146)
    joined["pos"] = joined["Start"] - joined["NucStart"]
    joined = joined[(joined["pos"] >= 0) & (joined["pos"] < nuc_len)]

    # -------------------------------
    # 4) Initialize count structures
    if dist == "binomial":
        combined_counts = {
            "watson_signal": {"successes": {b: np.zeros((num_sites, nuc_len), dtype=int) for b in base_names},
                              "trials":    {b: np.zeros((num_sites, nuc_len), dtype=int) for b in base_names}},
            "crick_signal": {"successes": {b: np.zeros((num_sites, nuc_len), dtype=int) for b in base_names},
                             "trials":    {b: np.zeros((num_sites, nuc_len), dtype=int) for b in base_names}},
            "tf_len": nuc_len,
            "num_sites": num_sites,
        }
    else:  # NB
        combined_counts = {
            "watson_signal": {b: np.zeros((num_sites, nuc_len), dtype=int) for b in base_names},
            "crick_signal":  {b: np.zeros((num_sites, nuc_len), dtype=int) for b in base_names},
            "tf_len": nuc_len,
            "num_sites": num_sites,
        }

    # -------------------------------
    # 5) Vectorized assignment
    # Each joined row belongs to (site_idx, pos). PyRanges "ID" column can help if needed.
    # If not present, we can create index by merge with nuc_pr.df.reset_index()
    nuc_df = nuc_pr.df.reset_index().rename(columns={"index":"site_id"})
    joined = joined.merge(nuc_df[["Chromosome","Start","End","site_id"]],
                          left_on=["Chromosome","NucStart","NucEnd"],
                          right_on=["Chromosome","Start","End"],
                          how="left")
    site_idx = joined["site_id"].to_numpy()
    pos      = joined["pos"].to_numpy()
    strands  = joined["Strand"].map(strand_map).to_numpy()
    bases    = joined["Base"].str.upper().to_numpy()

    if dist == "binomial":
        succ = joined[successes_col].to_numpy()
        trials = joined[trials_col].to_numpy()
        for i in range(len(joined)):
            lbl, b, s, t, si, p = strands[i], bases[i], succ[i], trials[i], site_idx[i], pos[i]
            if lbl is None or b not in base_names: continue
            combined_counts[lbl]["successes"][b][si, p] += s
            combined_counts[lbl]["trials"][b][si, p]    += t
        return fit_binomial_parameters(combined_counts, nuc_len, num_sites)

    else:  # NB
        cnts = joined[successes_col].to_numpy()
        for i in range(len(joined)):
            lbl, b, c, si, p = strands[i], bases[i], cnts[i], site_idx[i], pos[i]
            if lbl is None or b not in base_names: continue
            combined_counts[lbl][b][si, p] += c
        return fit_nb_parameters(combined_counts, nuc_len, num_sites, fitdist)



# segments = computeLinkers("/home/rapiduser/programs/RoboCOP/analysis/inputs/Chereji_2018_+1_-1_nucs.bed")
# segments = pd.DataFrame(segments)
# segments.columns = ['chr','start','end']
# # Ensure correct dtypes
# segments['start'] = segments['start'].astype(int)
# segments['end']   = segments['end'].astype(int)

# modified_bases_df = pd.read_csv("/home/rapiduser/projects/Fiber_seq/03202025_barcode01_sup_model_sorted_pileup_all_chr", sep='\t', header=None)
# split_columns = modified_bases_df[9].str.split(' ', expand=True)
# split_columns.columns = [i for i in range(9, 9 + split_columns.shape[1])]
# modified_bases_df = pd.concat([modified_bases_df.drop(columns=[9]), split_columns], axis=1)
# modified_bases_df[11] = modified_bases_df[11].astype(int)
# modified_bases_df[9] = modified_bases_df[9].astype(int)


######### nucleosome params #########

# nuc_params = compute_Fiber_seq_nucleosome(
#     modified_bases_df,
#     "/home/rapiduser/programs/RoboCOP/analysis/inputs/Chereji_2018_+1_-1_nucs.bed",
#     dist="binomial",
#     successes_col=11,
#     trials_col=9
# )

# # Save the dictionary to a file
# with open('inputs/nucleosome_params.pkl', 'wb') as f:
#     pickle.dump(nuc_params, f)


######### background params #########

# bg_params = compute_Fiber_seq_background(
#     modified_bases_df,
#     segments,
#     "/home/rapiduser/programs/RoboCOP/analysis/inputs/sacCer3_genome.fa",
#     dist="binomial",
#     successes_col=11,
#     trials_col=9
# )

# print(bg_params)

# # Save the dictionary to a file
# with open('inputs/bg_params.pkl', 'wb') as f:
#     pickle.dump(bg_params, f)


######### TF params #########

a = compute_Fiber_seq_TFPhisMus("Fiber_seq",\
                          "/home/rapiduser/projects/DMS-seq/DM1664/DM1664_trim_3prime_18bp_remaining_name_change_sorted.bam",\
                          "/home/rapiduser/projects/Fiber_seq/03202025_barcode01_sup_model_sorted_pileup_all_chr",\
                          "/home/rapiduser/programs/RoboCOP/analysis/inputs/rossi_peak_w_strand_conformed_to_PWM_abf1_reb1_atf2_azf1_1000_peakVal.bed",\
                            "/home/rapiduser/programs/RoboCOP/analysis/robocop_train/tmpDir",\
                            (0, 80),\
                                None,\
                                    0,\
                                        dist = "binomial")

# Okay maybe this is it
print(a)
abf1_reb1_params = a

# To load the dictionary later, you can use:

# Save the dictionary to a file
with open('inputs/abf1_reb1_params.pkl', 'wb') as f:
    pickle.dump(abf1_reb1_params, f)
