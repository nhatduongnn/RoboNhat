from robocop import robocop
from robocop.utils import concentration_probability_conversion, getNucleotides, readData
import numpy as np
import pandas
import os
from scipy import sparse

def save_sparse(f, k, v):
    v = np.array(v)
    v_sparse = sparse.csr_matrix(v)
    g = f.create_group(k)
    g.create_dataset('data', data=v_sparse.data)
    g.create_dataset('indices', data=v_sparse.indices)
    g.create_dataset('indptr', data=v_sparse.indptr)
    g.attrs['shape'] = v_sparse.shape

# read nucleotide sequence for given segments
def getNucSequence(nucFile, tmpDir, info_file, coords, idx = None):
    nucleotide_sequence = 1
    if idx != None:
        nucs = getNucleotides.getNucleotideSequence(nucFile, coords.iloc[idx]['chr'], coords.iloc[idx]['start'], coords.iloc[idx]['end'])
        k = "segment_" + str(idx)
        if k not in info_file.keys():
            g = info_file.create_group(k)
        else:
            g = info_file[k]

        # g_nucs = info_file.create_dataset(k + '/nucleotides', data = np.array(nucs))
        save_sparse(info_file, k+'/nucleotides', v=np.array(nucs))
        return nucleotide_sequence
        
    if nucFile:
        for i, r in coords.iterrows():
            nucs = getNucleotides.getNucleotideSequence(nucFile, r['chr'], r['start'], r['end'])
            k = "segment_" + str(i)
            if k not in info_file.keys():
                g = info_file.create_group(k)
            else:
                g = info_file[k]
            # g_nucs = info_file.create_dataset(k + '/nucleotides', data = np.array(nucs))
            save_sparse(info_file, k+'/nucleotides', v=np.array(nucs))
    else: nucleotide_sequence = None
    return nucleotide_sequence

# read MNase midpoint counts (both long and short fragments) for given segments
def getMNaseSmoothed(tmpDir, coords, fragRange, idx = None, tech = "MNase"):
    mnase_data_long = ""
    mnase_data_short = ""

    # offset bu 4 if ATAC-seq otherwise no offset
    offset = 4 if tech == "ATAC" else 0

    hdf = pandas.HDFStore(tmpDir + "midpoint_counts.h5", mode = "r")

    if not 1 == 1: 
        hdf_long = pandas.HDFStore(tmpDir + "midpoint_counts_long.h5", mode = "w")
        hdf_short = pandas.HDFStore(tmpDir + "midpoint_counts_short.h5", mode = "w")
        
        for k in hdf.keys():
            c = np.array(hdf.select(k))
            longs = np.sum(c[:, fragRange[0][0] : fragRange[0][1]], axis = 1)
            longs = np.convolve(longs, np.ones(11), mode = 'same')
            hdf_long.put(k, pandas.DataFrame(longs, columns = ['long']))
            shorts = np.sum(c[:, fragRange[1][0] : fragRange[1][1]], axis = 1)
            shorts = np.convolve(shorts, np.ones(11), mode = 'same')
            hdf_short.put(k, pandas.DataFrame(shorts, columns = ['short']))

    else:
        hdf_long = pandas.HDFStore(tmpDir + "midpoint_counts_long.h5", mode = "r")
        hdf_short = pandas.HDFStore(tmpDir + "midpoint_counts_short.h5", mode = "r")
    
        
    for i, r in coords.iterrows():
        cl = np.array(hdf_long.select(r['chr'])['long'])
        np.save(tmpDir + tech + "Long.idx" + str(i), cl[r['start'] - 1 : r['end']])

        cs = np.array(hdf_short.select(r['chr'])['short'])
        np.save(tmpDir + tech + "Short.idx" + str(i), cs[r['start'] - 1 : r['end']])

    hdf.close()
    hdf_long.close()
    hdf_short.close()

    return mnase_data_long, mnase_data_short

# read MNase midpoint counts (both long and short fragments) for given segments
def getMNase(mnaseFile, tmpDir, info_file, coords, fragRange, idx = None, tech = "MNase"):
    mnase_data_long = ""
    mnase_data_short = ""

    # offset bu 4 if ATAC-seq otherwise no offset
    offset = 4 if tech == "ATAC" else 0
    if mnaseFile != "":
        if idx != None:
            longs = readData.getValuesMNaseOneFileFragmentRange(mnaseFile, coords.iloc[idx]['chr'], coords.iloc[idx]['start'], coords.iloc[idx]['end'], fragRange[0], offset)
            shorts = readData.getValuesMNaseOneFileFragmentRange(mnaseFile, coords.iloc[idx]['chr'], coords.iloc[idx]['start'], coords.iloc[idx]['end'], fragRange[1], offset)

            k = "segment_" + str(idx)
            if k not in info_file.keys():
                g = info_file.create_group(k)
            else:
                g = info_file[k]

            # g_long = info_file.create_dataset(k + '/' + tech + '_long', data = np.array(longs))
            # g_short = info_file.create_dataset(k + '/' + tech + '_short', data = np.array(shorts))

            g_long = save_sparse(info_file, k+'/'+tech+'_long', v=np.array(longs))
            g_long = save_sparse(info_file, k+'/'+tech+'_short', v=np.array(shorts))
            return mnase_data_long, mnase_data_short

        for i, r in coords.iterrows():

            longs = readData.getValuesMNaseOneFileFragmentRange(mnaseFile, r['chr'], r['start'], r['end'], fragRange[0], offset)
            shorts = readData.getValuesMNaseOneFileFragmentRange(mnaseFile, r['chr'], r['start'], r['end'], fragRange[1], offset)
            
            k = "segment_" + str(i)
            if k not in info_file.keys():
                g = info_file.create_group(k)
            else:
                g = info_file[k]

            # g_long = info_file.create_dataset(k + '/' + tech + '_long', data = np.array(longs))
            # g_short = info_file.create_dataset(k + '/' + tech + '_short', data = np.array(shorts))
            g_long = save_sparse(info_file, k+'/'+tech+'_long', v=np.array(longs))
            g_long = save_sparse(info_file, k+'/'+tech+'_short', v=np.array(shorts))

    return mnase_data_long, mnase_data_short

def getKcounts(bamFile, nucleosomeFile, tfFile, chrSizesFile, tmpDir, coords, fragRange, windowSizeNuc, windowSizeTF, tech = "MNase"):
    readData.getKernelizedValues(bamFile, nucleosomeFile, tfFile, chrSizesFile, fragRange, windowSizeNuc, windowSizeTF, tmpDir)

def getFiber_seq(modkit_df, tmpDir, info_file, coords, nucleotide, idx = None, tech = "Fiber_seq"):
    """
    Processes modification data from a modification kit file and stores counts of a specific nucleotide's 
    modified occurrences across multiple genomic segments into an HDF5-like file group structure.

    Parameters:
    -----------
    modkit_df : df
        loaded df to the fiberseq modkit pileup bed file.
    tmpDir : str
        Temporary directory path (currently unused in the function).
    info_file : h5py.File or similar group-like object
        Open HDF5 file or group to store the methylation count datasets.
    coords : pandas.DataFrame
        DataFrame containing genomic coordinate segments with columns:
        - 'chr': chromosome name
        - 'start': start position of segment
        - 'end': end position of segment
    nucleotide : str
        The nucleotide base to count methylations on (e.g., 'A', 'C', 'G', or 'T').
    idx : int, optional, default=None
        Index of a specific segment in `coords` to process. If None, processes all segments.
    tech : str, optional, default="Fiber_seq"
        Technology or dataset name to label groups/datasets in the HDF5 file.

    Returns:
    --------
    tuple
        Two empty strings (placeholder variables `fiber_seq_data_count_meth_watson` and 
        `fiber_seq_data_count_meth_crick`) — actual data are saved into `info_file` groups/datasets.

    """
    fiber_seq_data_count_meth_watson = ""
    fiber_seq_data_count_meth_crick = ""


    if modkit_df != "":
        if idx != None:
            count_meth_watson,count_meth_crick = readData.getValuesFiber_seqOneFileNucleotide(modkit_df, coords.iloc[idx]['chr'], coords.iloc[idx]['start'], coords.iloc[idx]['end'], nucleotide, offset)

            k = "segment_" + str(idx)
            if k not in info_file.keys():
                g = info_file.create_group(k)
            else:
                g = info_file[k]

            g_count_meth_watson = info_file.create_dataset(k + '/' + tech + '_count_meth_watson', data = np.array(count_meth_watson))
            g_count_meth_crick = info_file.create_dataset(k + '/' + tech + '_count_meth_crick', data = np.array(count_meth_crick))

            return fiber_seq_data_count_meth_watson, fiber_seq_data_count_meth_crick

        for i, r in coords.iterrows():

            count_meth_watson,count_meth_crick = readData.getValuesFiber_seqOneFileNucleotide(modkit_df, r['chr'], r['start'], r['end'], nucleotide, offset)

            k = "segment_" + str(i)
            if k not in info_file.keys():
                g = info_file.create_group(k)
            else:
                g = info_file[k]

            g_count_meth_watson = info_file.create_dataset(k + '/' + tech + '_count_meth_watson', data = np.array(count_meth_watson))
            g_count_meth_crick = info_file.create_dataset(k + '/' + tech + '_count_meth_crick', data = np.array(count_meth_crick))

    return fiber_seq_data_count_meth_watson, fiber_seq_data_count_meth_crick



