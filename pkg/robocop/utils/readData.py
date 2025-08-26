##############################
# Read values form BAM file. #
##############################
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np
import pysam
import pandas
import os
import sys

# Given MNase file, chromosome, start, and end, extract fragment midpoint counts of
# fragments of range given by fragRange

def getValuesMNaseOneFileFragmentRange(MNaseFile, chrm, minStart, maxEnd, fragRange, offset = 0):
    countMid = np.zeros(maxEnd - minStart + 1).astype(int)
    samfile = pysam.AlignmentFile(MNaseFile, "rb")
    region = samfile.fetch(chrm, max(0, minStart - fragRange[1] - 1), maxEnd + fragRange[1] - 1)
    for i in region:
        if i.template_length == 0: continue
        if i.template_length - 2*offset>= 0:
            start = i.reference_start + 1 + offset
            end = i.reference_start + offset + i.template_length - 2*offset
        else:
            # ignore reads on antisense strand
            continue
        width = abs(i.template_length)
        if width >= fragRange[0] and width <= fragRange[1] and (start + end)/2 >= minStart and (start + end)/2 <= maxEnd:
            countMid[int((start + end)/2 - minStart)] += 1

    return np.array(countMid)

def getValuesMNaseFragmentRange(bamFiles, chromosome, start, stop, fragRange):
    mnase = np.array([getValuesMNaseOneFileFragmentRange(x, chromosome, start, stop, fragRange) for x in bamFiles])
    return mnase

def getChrSizes(chrSizesFile):
    chrSizes = {}
    with open(chrSizesFile) as infile:
        for line in infile:
            l = line.strip().split()
            if l[0] == 'chrM': continue
            chrSizes[l[0]] = int(l[1])

    return chrSizes

def getMidpointCounts(samfile, c, chrSize, fragRange):
        counts = np.zeros((fragRange[1] - fragRange[0] + 1, chrSize)).astype(int)

        try:
            regions = samfile.fetch(c, 0, chrSize)
        except ValueError as ve:
            return None 
        
        for r in regions:
            if r.template_length <= 0: continue
            if r.template_length < fragRange[0]: continue
            if r.template_length > fragRange[1]: continue
            rStart = r.reference_start + 1 # + offset
            rEnd = r.reference_start + r.template_length # - 2*offset
            m = (rStart + rEnd)/2
            width = abs(r.template_length)
            counts[r.template_length - fragRange[0], int(m)] += 1

        return counts


def get2DValues(bamFile, chrSizesFile, fragRange, tmpDir):
    chrSizes = getChrSizes(chrSizesFile)
    samfile = pysam.AlignmentFile(bamFile)

    pop_c = ['chrM']
    
    if not os.path.isfile(tmpDir + "midpoint_counts.h5"):
        hdf = pandas.HDFStore(tmpDir + "midpoint_counts.h5", mode = "w")
        for c in chrSizes:
            if c == 'chrM': continue
            counts = getMidpointCounts(samfile, c, chrSizes[c], fragRange)
            if counts is None:
                pop_c.append(c)
                continue
            counts_df = pandas.DataFrame(counts.T, columns = range(counts.shape[0]))
            hdf.put(c, counts_df)
        hdf.close()

    hdf = pandas.HDFStore(tmpDir + "midpoint_counts.h5", mode = "r")


def getValuesFiber_seqOneFileNucleotide(modkit_df, chrm, minStart, maxEnd, nucleotide, offset = 0):
    """
    Extracts and counts the occurrences of a specific modified nucleotide on Watson and Crick strands 
    within a given chromosome coordinate range from a modification kit CSV file.

    Parameters:
    -----------
    modkit_df : df
        loaded df to the fiberseq modkit pileup bed file.
    chrm : str
        Chromosome name to filter the data.
    minStart : int
        Minimum start position (inclusive) of the region of interest.
    maxEnd : int
        Maximum end position (inclusive) of the region of interest.
    nucleotide : str
        The nucleotide base to count (e.g., 'A', 'C', 'G', or 'T').
    offset : int, optional, default=0
        An optional offset to adjust position indexing (currently unused in the function).

    Returns:
    --------
    tuple of numpy.ndarray
        Two arrays of integers corresponding to counts of the specified modified nucleotide on:
        - Watson strand (positive strand) within the specified coordinate range.
        - Crick strand (negative strand) within the specified coordinate range.

    Notes:
    ------
    - The function assumes the input CSV has specific columns where:
        * Column 0: Chromosome
        * Column 1: Position
        * Column 2: End position
        * Column 3: Modified base
        * Column 5: Strand information ('+' or '-')
        * Column 9: A field containing space-separated data that is further split (not fully used here)
        * Column 11: Count of modified bases at that position
    - Positions are zero-indexed relative to `minStart`.
    - This function relies on pandas and numpy libraries.
    """
    ...
    modified_bases_df = pd.read_csv(modkit_df, sep='\t', header=None)
    # Split the 9th column into multiple columns
    split_columns = modified_bases_df[9].str.split(' ', expand=True)
    split_columns.columns = [i for i in range(9,9+split_columns.shape[1])]

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
        pos = row[1] - r1['start']
        if strand_info == '+':
            if modified_base == nucleotide:
                count_meth_watson[pos] += count
        elif strand_info == '-':
            if modified_base == nucleotide:
                count_meth_crick[pos] += count

    return np.array(count_meth_watson),np.array(count_meth_crick)
