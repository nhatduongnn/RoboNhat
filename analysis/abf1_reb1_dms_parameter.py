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

def computeMNaseTFPhisMus(bamFile, csvFile, tmpDir, fragRange, filename, offset = 0):
    """
    Negative binomial distribution for short fragments at TF
    binding sites.
    """


    samfile = pysam.AlignmentFile(bamFile, "rb")

    fasta_file = {}

    # Read in fasta file as a seq_record object, #!# need to not hardcode the file name
    for seq_record in SeqIO.parse("/home/rapiduser/programs/RoboCOP/analysis/inputs/SacCer3.fa", "fasta"):
        fasta_file[seq_record.id] = seq_record.seq


    tfs = pandas.read_csv(csvFile, sep = '\t', header = None)
    tfs = tfs.rename(columns = {0: 'chr', 1: 'start', 2: 'end', 3: 'tf_name', 4: 'score', 5: 'strand'})
    #tfs_abf1 = tfs.loc[tfs['tf_name'] == 'ABF1']
    tf_counts = tfs.groupby('tf_name')['chr'].count()
    ind_tfs = list(tf_counts.loc[tf_counts >= 50].index)
    combine_tfs = list(tf_counts.loc[tf_counts < 50].index)

    # #!# Enable to test motif
    # motif_list = []
    # motif_list1 = []

    def compute_individual_DMSTFPhisMus(samfile, tfs_df, tf_name, offset = 0):

        tfCounts_watson_A = []
        tfCounts_watson_C = []
        tfCounts_watson_G = []
        tfCounts_watson_T = []

        tfCounts_crick_A = []
        tfCounts_crick_C = []
        tfCounts_crick_G = []
        tfCounts_crick_T = []

        one_tf_df = tfs_df.loc[tfs_df['tf_name'] == tf_name]

        for i1, r1 in one_tf_df.iterrows():
            # mid = int(0.5*(r1['start'] + r1['end']))
            # minStart = mid - 5
            # maxEnd = mid + 5
            chrm = r1['chr']

            # # Only want to loop through ABF1 locations that are on the positive strands
            if r1['strand'] != '+' or r1['tf_name'] != tf_name:
                continue


            
            #!# Enable to test motif
            #motif_list = motif_list + [list(fasta.fetch(chrm, r1['start']-1, r1['end']).upper())]
            #motif_list1 = motif_list1 + [list(SeqFeature(FeatureLocation(r1['start']-1,r1['end'])).extract(fasta_file[chrm]).upper())]

            

            print(r1)

            # Calculate the length of this specific TF, save if for use later on in allocation of storage data structures
            tf_len = r1['end'] - r1['start'] + 1

            countMeth_watson_A = [1 for i in range(tf_len)]
            countMeth_watson_C = [1 for i in range(tf_len)]
            countMeth_watson_G = [1 for i in range(tf_len)]
            countMeth_watson_T = [1 for i in range(tf_len)]

            countMeth_crick_A = [1 for i in range(tf_len)]
            countMeth_crick_C = [1 for i in range(tf_len)]
            countMeth_crick_G = [1 for i in range(tf_len)]
            countMeth_crick_T = [1 for i in range(tf_len)]

            ## start +1 and end -1 to acount for methylations that are 5' of read starts and 3' of read ends
            region = samfile.fetch(chrm, r1['start']-1, r1['end']+1)
            for i in region:
                #print('start is {}-{} and length is {} and flag is {}'.format(i.reference_start,i.reference_end,i.template_length,i.flag))
                # Compute the fragment start and end based on template_length
                if i.template_length > 0:
                    ## Add one to go from 0 base to 1 base, the START coordinate in pysam is inclusive
                    ## But methylation of the start is 5' so gotta minus 1
                    frag_start = i.reference_start + 1 - 1
                    #frag_end = i.reference_start + i.template_length - 1
                    #if frag_start >= r1['start']-1 and frag_start < r1['end']:
                    if frag_start >= r1['start'] and frag_start <= r1['end']:
                        #nucleotide = fasta.fetch(chrm, frag_start-1, frag_start).upper()
                        nucleotide = SeqFeature(FeatureLocation(frag_start-1,frag_start)).extract(fasta_file[chrm])

                        ## Minus one because we want the position of the methylation
                        pos = frag_start - (r1['start'])
                        if nucleotide == 'A':
                            countMeth_watson_A[pos] += 1
                        elif nucleotide == 'C':
                            countMeth_watson_C[pos] += 1
                        elif nucleotide == 'G':
                            countMeth_watson_G[pos] += 1
                        elif nucleotide == 'T':
                            countMeth_watson_T[pos] += 1
                        else:
                            print(f"Unexpected nucleotide at frag_start: {nucleotide}")
                    else:
                        print('uh oh1')
                elif i.template_length < 0:
                    #frag_start = i.reference_start + i.template_length + 1
                    ## Add one to go from 0 base to 1 base, the END cooridnate in pysam is NOT incluside
                    ## so minus one
                    ## But methylation of the end of the read is 3' so then +1
                    frag_end = i.reference_end +1 - 1 + 1
                    #frag_end = i.reference_end
                    if frag_end >= r1['start'] and frag_end <= r1['end']:
                    #if frag_end >= r1['start'] and frag_end <= r1['start']+1:
                        #nucleotide = fasta.fetch(chrm, frag_end - 1, frag_end).upper()
                        nucleotide = SeqFeature(FeatureLocation(frag_end - 1,frag_end)).extract(fasta_file[chrm])
                        # Reverse-complement the fetched nucleotide for Crick strand
                        # pos is correct, remember the -14 position would be the 1st box
                        # on the crick strand. So from left to right are boxes -14 to -1
                        # But now I scrapped all that and changed it to frag_end minus start instead
                        #pos = frag_end - (r1['end']) -1
                        pos = frag_end - (r1['start'])
                        if nucleotide == 'A':
                            countMeth_crick_T[pos] += 1
                        elif nucleotide == 'C':
                            countMeth_crick_G[pos] += 1
                        elif nucleotide == 'G':
                            countMeth_crick_C[pos] += 1
                        elif nucleotide == 'T':
                            countMeth_crick_A[pos] += 1
                        else:
                            print(f"Unexpected nucleotide at frag_end: {nucleotide}")
                    else:
                        print('uh oh2')
                else:
                    continue  # Skip reads with undefined fragment size

                # Increment count for the 5' and 3' ends if they overlap the TF region

            # if r1['strand'] == '-':
            #     countMeth_watson_A

            tfCounts_watson_A = tfCounts_watson_A + countMeth_watson_A
            #print(len(tfCounts_watson_A))
            tfCounts_watson_C = tfCounts_watson_C + countMeth_watson_C
            tfCounts_watson_G = tfCounts_watson_G + countMeth_watson_G
            tfCounts_watson_T = tfCounts_watson_T + countMeth_watson_T

            tfCounts_crick_A = tfCounts_crick_A + countMeth_crick_A
            tfCounts_crick_C = tfCounts_crick_C + countMeth_crick_C
            tfCounts_crick_G = tfCounts_crick_G + countMeth_crick_G
            tfCounts_crick_T = tfCounts_crick_T + countMeth_crick_T
            
        # motif_list_array = np.array(motif_list1)

        #np.array(tfCounts).reshape(211,11)
        # tfCounts_watson_A_reshape = np.array(tfCounts_watson_A).reshape(77,14)
        # tfCounts_watson_C_reshape = np.array(tfCounts_watson_C).reshape(77,14)
        # tfCounts_watson_G_reshape = np.array(tfCounts_watson_G).reshape(77,14)
        # tfCounts_watson_T_reshape = np.array(tfCounts_watson_T).reshape(77,14)

        # tfCounts_crick_A_reshape = np.array(tfCounts_crick_A).reshape(77,14)
        # tfCounts_crick_C_reshape = np.array(tfCounts_crick_C).reshape(77,14)
        # tfCounts_crick_G_reshape = np.array(tfCounts_crick_G).reshape(77,14)
        # tfCounts_crick_T_reshape = np.array(tfCounts_crick_T).reshape(77,14)
        # motif_watson_matrix = np.zeros([4,9])
        # for j in range(0,9):
        #     motif_watson_matrix[0,j] = len([i for i in motif_list_array[:,j] if i == 'A'])
        #     motif_watson_matrix[1,j] = len([i for i in motif_list_array[:,j] if i == 'C'])
        #     motif_watson_matrix[2,j] = len([i for i in motif_list_array[:,j] if i == 'G'])
        #     motif_watson_matrix[3,j] = len([i for i in motif_list_array[:,j] if i == 'T'])
        
        # ax = sns.heatmap(motif_watson_matrix, annot=True, fmt=".2f", cmap="Reds")
        # ax.set_title('Count of Bases at Motif')
        # ax.set_yticklabels(["A", "C", "G", "T"], rotation=0)
        # plt.show()

        matrix_len = int(len(tfCounts_watson_A)/tf_len)
        tfCounts_watson_A_reshape = np.array(tfCounts_watson_A).reshape(matrix_len,tf_len)
        tfCounts_watson_C_reshape = np.array(tfCounts_watson_C).reshape(matrix_len,tf_len)
        tfCounts_watson_G_reshape = np.array(tfCounts_watson_G).reshape(matrix_len,tf_len)
        tfCounts_watson_T_reshape = np.array(tfCounts_watson_T).reshape(matrix_len,tf_len)

        tfCounts_crick_A_reshape = np.array(tfCounts_crick_A).reshape(matrix_len,tf_len)
        tfCounts_crick_C_reshape = np.array(tfCounts_crick_C).reshape(matrix_len,tf_len)
        tfCounts_crick_G_reshape = np.array(tfCounts_crick_G).reshape(matrix_len,tf_len)
        tfCounts_crick_T_reshape = np.array(tfCounts_crick_T).reshape(matrix_len,tf_len)
        


        full_cell_phi = np.zeros([8,tf_len])
        full_cell_mu = np.zeros([8,tf_len])
        try:
            fitdist = importr('fitdistrplus')
            f = io.StringIO()
            with redirect_stdout(f):
                for i in range(0,tf_len):
                    p_watson_A = fitdist.fitdist(vectors.IntVector(tfCounts_watson_A_reshape[:,i]), 'nbinom', method = "mle")
                    p_watson_C = fitdist.fitdist(vectors.IntVector(tfCounts_watson_C_reshape[:,i]), 'nbinom', method = "mle")
                    p_watson_G = fitdist.fitdist(vectors.IntVector(tfCounts_watson_G_reshape[:,i]), 'nbinom', method = "mle")
                    p_watson_T = fitdist.fitdist(vectors.IntVector(tfCounts_watson_T_reshape[:,i]), 'nbinom', method = "mle")

                    p_crick_A = fitdist.fitdist(vectors.IntVector(tfCounts_crick_A_reshape[:,i]), 'nbinom', method = "mle")
                    p_crick_C = fitdist.fitdist(vectors.IntVector(tfCounts_crick_C_reshape[:,i]), 'nbinom', method = "mle")
                    p_crick_G = fitdist.fitdist(vectors.IntVector(tfCounts_crick_G_reshape[:,i]), 'nbinom', method = "mle")
                    p_crick_T = fitdist.fitdist(vectors.IntVector(tfCounts_crick_T_reshape[:,i]), 'nbinom', method = "mle")
                
                    p_watson_A = p_watson_A.rx2("estimate")
                    full_cell_phi[0,i] = p_watson_A.rx2("size")[0]
                    full_cell_mu[0,i] = p_watson_A.rx2("mu")[0]


                    p_watson_C = p_watson_C.rx2("estimate")
                    full_cell_phi[1,i] = p_watson_C.rx2("size")[0]
                    full_cell_mu[1,i] = p_watson_C.rx2("mu")[0]

                    p_watson_G = p_watson_G.rx2("estimate")
                    full_cell_phi[2,i] = p_watson_G.rx2("size")[0]
                    full_cell_mu[2,i] = p_watson_G.rx2("mu")[0]

                    p_watson_T = p_watson_T.rx2("estimate")
                    full_cell_phi[3,i] = p_watson_T.rx2("size")[0]
                    full_cell_mu[3,i] = p_watson_T.rx2("mu")[0]

                    p_crick_A = p_crick_A.rx2("estimate")
                    full_cell_phi[4,i] = p_crick_A.rx2("size")[0]
                    full_cell_mu[4,i] = p_crick_A.rx2("mu")[0]

                    p_crick_C = p_crick_C.rx2("estimate")
                    full_cell_phi[5,i] = p_crick_C.rx2("size")[0]
                    full_cell_mu[5,i] = p_crick_C.rx2("mu")[0]

                    p_crick_G = p_crick_G.rx2("estimate")
                    full_cell_phi[6,i] = p_crick_G.rx2("size")[0]
                    full_cell_mu[6,i] = p_crick_G.rx2("mu")[0]

                    p_crick_T = p_crick_T.rx2("estimate")
                    full_cell_phi[7,i] = p_crick_T.rx2("size")[0]
                    full_cell_mu[7,i] = p_crick_T.rx2("mu")[0]
                
            # p = p.rx2("estimate")
            # size = p.rx2("size")[0]
            # mu = p.rx2("mu")[0]
            #params = {'mu': mu, 'phi': size}
            param = 0
            plt.imshow(full_cell_phi[0:4,:], cmap='Blues')
            plt.title('Phi (overdispersion) values for Watson strand')
            plt.colorbar()  # Add a colorbar to show the intensity scale
            plt.show()

            plt.imshow(full_cell_phi[4:8,:], cmap='Blues')
            plt.title('Phi (overdispersion) values for Crick strand')
            plt.colorbar()  # Add a colorbar to show the intensity scale
            plt.show()

            plt.imshow(full_cell_mu[0:4,:], cmap='Reds')
            plt.title('Mu (mean) values for Watson strand')
            plt.colorbar()  # Add a colorbar to show the intensity scale
            plt.show()

            plt.imshow(full_cell_mu[4:8,:], cmap='Reds')
            plt.title('Mu (mean) values for Crick strand')
            plt.colorbar()  # Add a colorbar to show the intensity scale
            plt.show()


            bob = np.concatenate(([tfCounts_watson_A_reshape.sum(axis=0)],[tfCounts_watson_C_reshape.sum(axis=0)],[tfCounts_watson_G_reshape.sum(axis=0)],[tfCounts_watson_T_reshape.sum(axis=0)]))
            plt.imshow(bob, cmap='Reds')
            plt.title('Meth counts watson')
            plt.colorbar()  # Add a colorbar to show the intensity scale
            plt.show()

            bob = np.concatenate(([tfCounts_crick_A_reshape.sum(axis=0)],[tfCounts_crick_C_reshape.sum(axis=0)],[tfCounts_crick_G_reshape.sum(axis=0)],[tfCounts_crick_T_reshape.sum(axis=0)]))
            plt.imshow(bob, cmap='Reds')
            plt.title('Meth counts crick')
            plt.colorbar()  # Add a colorbar to show the intensity scale
            plt.show()

        except Exception as e:
            # hard code values
            print('we here')
            if e.args[0][:14] == "Error in (func":
                mu = 0.002
                phi = 100
                params = {'mu': mu, 'phi': phi}
        return params
    
    for i in ind_tfs:
        print(i)
        if i == 'REB1':
            compute_individual_DMSTFPhisMus(samfile, tfs, i, offset = 0)
    
    return params_all


a = computeMNaseTFPhisMus("/home/rapiduser/projects/DMS-seq/DM1664/DM1664_trim_3prime_18bp_remaining_name_change_sorted.bam",\
                          "/home/rapiduser/programs/RoboCOP/analysis/inputs/MacIsaac_sacCer3_liftOver_Abf1_Reb1.bed",\
                            "/home/rapiduser/programs/RoboCOP/analysis/robocop_train/tmpDir",\
                            (0, 80),\
                                None,\
                                    0)


print(a)