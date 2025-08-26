def getFiber_seq(modkit_df, tmpDir, info_file, coords, nucleotide, idx = None, tech = "Fiber_seq"):
    fiber_seq_data_count_meth_watson = ""
    fiber_seq_data_count_meth_crick = ""


    if not modkit_df.empty:
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

            #count_meth_watson,count_meth_crick = readData.getValuesFiber_seqOneFileNucleotide(modkit_df, r['chr'], r['start'], r['end'], nucleotide, offset)
            count_meth_watson,count_meth_crick = getValuesFiber_seqOneFileNucleotide(modkit_df, r['chr'], r['start'], r['end'], nucleotide)

            k = "segment_" + str(i)
            if k not in info_file.keys():
                g = info_file.create_group(k)
            else:
                g = info_file[k]

            g_count_meth_watson = info_file.create_dataset(k + '/' + tech + '_count_meth_watson', data = np.array(count_meth_watson))
            g_count_meth_crick = info_file.create_dataset(k + '/' + tech + '_count_meth_crick', data = np.array(count_meth_crick))

    return fiber_seq_data_count_meth_watson, fiber_seq_data_count_meth_crick

def getValuesFiber_seqOneFileNucleotide(modkit_df, chrm, minStart, maxEnd, nucleotide):

    count_meth_watson = np.zeros(maxEnd - minStart + 1).astype(int)
    count_meth_crick = np.zeros(maxEnd - minStart + 1).astype(int)

    relevant_rows = modified_bases_df[
    (modified_bases_df[0] == chrm) &
    (modified_bases_df[1] < maxEnd) &
    (modified_bases_df[2] > minStart)
]

    for _, row in relevant_rows.iterrows():
        modified_base = row[3].upper()
        strand_info = row[5]
        count = int(row[11])
        pos = row[1] - minStart
        if strand_info == '+':
            if modified_base == nucleotide:
                count_meth_watson[pos] += count
        elif strand_info == '-':
            if modified_base == nucleotide:
                count_meth_crick[pos] += count

    return np.array(count_meth_watson),np.array(count_meth_crick)


import h5py
import pandas as pd
import numpy as np


info_file = h5py.File('./robocop_train/tmpDir/info.h5', mode = 'w') 
tmpDir = './robocop_train/tmpDir/'
coords = pd.read_csv('./coord_train.tsv', sep = "\t")
nucleotide = 'A'
tech= "Fiber_seq"
modkitFile = '/home/rapiduser/projects/Fiber_seq/03202025_barcode01_sup_model_sorted_pileup_all_chr'

# Load Modkit file only once outside loop
modified_bases_df = pd.read_csv(modkitFile, sep='\t', header=None)
# Split the 9th column into multiple columns (if following previous code pattern)
split_columns = modified_bases_df[9].str.split(' ', expand=True)
split_columns.columns = [i for i in range(9,9+split_columns.shape[1])]
modified_bases_df = pd.concat([modified_bases_df.drop(columns=[9]), split_columns], axis=1)


fiber_seq_data_count_meth_watson, fiber_seq_data_count_meth_crick = getFiber_seq(modified_bases_df, tmpDir, info_file, coords, nucleotide, tech = tech)
