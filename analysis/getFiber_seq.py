def getFiber_seq(modkit_df, tmpDir, info_file, coords, meth_nucleotide, pseudo_successes, pseudo_trials, idx = None, tech = "Fiber"):
    fiber_seq_data_count_meth_watson = ""
    fiber_seq_data_count_meth_crick = ""


    if not modkit_df.empty and tech == "Fiber":
        if idx != None:
            count_meth_watson,count_meth_crick,count_A_watson,count_A_crick = readData.getValuesFiber_seqOneFileNucleotide(modkit_df, coords.iloc[idx]['chr'], coords.iloc[idx]['start'], coords.iloc[idx]['end'], meth_nucleotide, pseudo_successes, pseudo_trials)

            k = "segment_" + str(idx)
            if k not in info_file.keys():
                g = info_file.create_group(k)
            else:
                g = info_file[k]

            # g_count_meth_watson = info_file.create_dataset(k + '/' + tech + '_count_meth_watson', data = np.array(count_meth_watson))
            # g_count_meth_crick = info_file.create_dataset(k + '/' + tech + '_count_meth_crick', data = np.array(count_meth_crick))
            # g_count_A_watson = info_file.create_dataset(k + '/' + tech + '_count_A_watson', data = np.array(count_A_watson))
            # g_count_A_crick = info_file.create_dataset(k + '/' + tech + '_count_A_crick', data = np.array(count_A_crick))

            g_count_meth_watson = save_sparse(info_file, k + '/' + tech + '_count_meth_watson', v = np.array(count_meth_watson))
            g_count_meth_crick = save_sparse(info_file, k + '/' + tech + '_count_meth_crick', v = np.array(count_meth_crick))
            g_count_A_watson = save_sparse(info_file, k + '/' + tech + '_count_A_watson', v = np.array(count_A_watson))
            g_count_A_crick = save_sparse(info_file, k + '/' + tech + '_count_A_crick', v = np.array(count_A_crick))

            return fiber_seq_data_count_meth_watson, fiber_seq_data_count_meth_crick

        for i, r in coords.iterrows():

            #count_meth_watson,count_meth_crick = readData.getValuesFiber_seqOneFileNucleotide(modkit_df, r['chr'], r['start'], r['end'], nucleotide, offset)
            count_meth_watson,count_meth_crick,count_A_watson,count_A_crick = getValuesFiber_seqOneFileNucleotide(modkit_df, r['chr'], r['start'], r['end'], meth_nucleotide, pseudo_successes, pseudo_trials)

            k = "segment_" + str(i)
            if k not in info_file.keys():
                g = info_file.create_group(k)
            else:
                g = info_file[k]

            # g_count_meth_watson = info_file.create_dataset(k + '/' + tech + '_count_meth_watson', data = np.array(count_meth_watson))
            # g_count_meth_crick = info_file.create_dataset(k + '/' + tech + '_count_meth_crick', data = np.array(count_meth_crick))
            # g_count_A_watson = info_file.create_dataset(k + '/' + tech + '_count_A_watson', data = np.array(count_A_watson))
            # g_count_A_crick = info_file.create_dataset(k + '/' + tech + '_count_A_crick', data = np.array(count_A_crick))

            g_count_meth_watson = save_sparse(info_file, k + '/' + tech + '_count_meth_watson', v = np.array(count_meth_watson))
            g_count_meth_crick = save_sparse(info_file, k + '/' + tech + '_count_meth_crick', v = np.array(count_meth_crick))
            g_count_A_watson = save_sparse(info_file, k + '/' + tech + '_count_A_watson', v = np.array(count_A_watson))
            g_count_A_crick = save_sparse(info_file, k + '/' + tech + '_count_A_crick', v = np.array(count_A_crick))

            # # Save the data as a numpy file
            # np.save('inputs/' + k + '_' + tech + '_count_meth_watson.npy', count_meth_watson)
            # np.save('inputs/' + k + '_' + tech + '_count_meth_crick.npy', count_meth_crick)
            # np.save('inputs/' + k + '_' + tech + '_count_A_watson.npy', count_A_watson)
            # np.save('inputs/' + k + '_' + tech + '_count_A_crick.npy', count_A_crick)

    return fiber_seq_data_count_meth_watson, fiber_seq_data_count_meth_crick

def getValuesFiber_seqOneFileNucleotide(modkit_df, chrm, minStart, maxEnd, meth_nucleotide, pseudo_successes, pseudo_trials):

    minStart = minStart-1 #Convert to 0-based

    count_meth_watson = np.zeros(maxEnd - minStart + 1).astype(int)
    count_A_watson = np.zeros(maxEnd - minStart + 1).astype(int)
    count_meth_crick = np.zeros(maxEnd - minStart + 1).astype(int)
    count_A_crick = np.zeros(maxEnd - minStart + 1).astype(int)

    relevant_rows = modkit_df[
    (modkit_df[0] == chrm) &
    (modkit_df[1] < maxEnd) &
    (modkit_df[2] > minStart)
]

    for _, row in relevant_rows.iterrows():
        modified_base = row[3].upper()
        strand_info = row[5]
        count = int(row[11])
        trials = int(row[9])
        pos = row[1] - minStart
        if strand_info == '+':
            if modified_base == meth_nucleotide:
                count_meth_watson[pos] += (count + pseudo_successes)
                count_A_watson[pos] += (trials + pseudo_trials)
        elif strand_info == '-':
            if modified_base == meth_nucleotide:
                count_meth_crick[pos] += (count + pseudo_successes)
                count_A_crick[pos] += (trials + pseudo_trials)

    return np.array(count_meth_watson),np.array(count_meth_crick), np.array(count_A_watson), np.array(count_A_crick)


import h5py
import pandas as pd
import numpy as np


info_file = h5py.File('./robocop_train/tmpDir/info.h5', mode = 'w') 
tmpDir = './robocop_train/tmpDir/'
coords = pd.read_csv('./coord_train.tsv', sep = "\t")
meth_nucleotide = 'A'
tech= "Fiber"
modkitFile = '/home/rapiduser/projects/Fiber_seq/03202025_barcode01_sup_model_sorted_pileup_all_chr'

# Load Modkit file only once outside loop
modified_bases_df = pd.read_csv(modkitFile, sep='\t', header=None)
# Split the 9th column into multiple columns (if following previous code pattern)
split_columns = modified_bases_df[9].str.split(' ', expand=True)
split_columns.columns = [i for i in range(9,9+split_columns.shape[1])]
modified_bases_df = pd.concat([modified_bases_df.drop(columns=[9]), split_columns], axis=1)
avg_successes = modified_bases_df[11].astype('int').mean()
avg_trials = modified_bases_df[9].astype('int').mean()


fiber_seq_data_count_meth_watson, fiber_seq_data_count_meth_crick = getFiber_seq(modified_bases_df, tmpDir, info_file, coords, meth_nucleotide, avg_successes, avg_trials, tech = tech)
print(fiber_seq_data_count_meth_watson)