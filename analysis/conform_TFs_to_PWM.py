import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import sys
sys.path.insert(0, '/home/rapiduser/programs/RoboCOP/pkg/')
import robocop.utils.parameterize as parameterize


import re

# Mapping for special cases
special_map = {
    "X": 23, "Y": 24, "M": 25, "MT": 25
}
special_map_rev = {v: k for k, v in special_map.items()}

# Roman numeral conversion helpers
roman_numerals = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")
]

def int_to_roman(n: int) -> str:
    result = ""
    for value, symbol in roman_numerals:
        while n >= value:
            result += symbol
            n -= value
    return result

def roman_to_int(s: str) -> int:
    roman_dict = {sym: val for val, sym in roman_numerals}
    i, total = 0, 0
    while i < len(s):
        if i + 1 < len(s) and s[i:i+2] in roman_dict:
            total += roman_dict[s[i:i+2]]
            i += 2
        else:
            total += roman_dict[s[i]]
            i += 1
    return total

def detect_format(chrom: str) -> str:
    if chrom.startswith("chr"):
        c = chrom[3:]
        if c.isdigit() or c in special_map:
            return "chrNumber"
        else:
            return "chrRoman"
    else:
        if chrom.isdigit() or chrom in special_map:
            return "number"
        else:
            return "roman"

def convert_chromosome(chrom: str, target_format: str) -> str:
    source_format = detect_format(chrom)
    
    # Normalize to integer
    if source_format in ("number", "chrNumber"):
        key = chrom[3:] if chrom.startswith("chr") else chrom
        num = int(key) if key.isdigit() else special_map[key]
    else:  # roman or chrRoman
        key = chrom[3:] if chrom.startswith("chr") else chrom
        if key in special_map:
            num = special_map[key]
        else:
            num = roman_to_int(key)
    
    # Convert to target
    if target_format == "number":
        return str(num) if num not in special_map_rev else special_map_rev[num]
    elif target_format == "chrNumber":
        return "chr" + (str(num) if num not in special_map_rev else special_map_rev[num])
    elif target_format == "roman":
        return int_to_roman(num) if num not in special_map_rev else special_map_rev[num]
    elif target_format == "chrRoman":
        return "chr" + (int_to_roman(num) if num not in special_map_rev else special_map_rev[num])
    else:
        raise ValueError(f"Unknown target format: {target_format}")


def score_sequence_with_pwm(seq, pwm):
    """Score a sequence (string) against a PWM matrix.
       PWM shape: (4, L), rows = A,C,G,T
    """
    base_to_idx = {"A":0, "C":1, "G":2, "T":3}
    score = 0.0
    for i, base in enumerate(seq.upper()):
        if base in base_to_idx:
            score += pwm[base_to_idx[base], i]
        else:
            score += pwm[4, i]  # if 'N' column exists
    return score

def best_match_for_tf(row, motifDict, genome_fasta):
    """Find best matching window for TF row in BED."""
    chrom, start, end, tf = row["chr"], row["start"], row["end"], row["TF"]

    chrom = convert_chromosome(chrom, "chrRoman")

    translate_motif_name = {name.split('_', 1)[0].lower(): name for name in motifDict.keys()}
    # midpoint
    midpoint = (start + end) // 2
    if tf in translate_motif_name.keys():

        pwm = motifDict[translate_motif_name[tf]][:4]   # take A,C,G,T rows only
        L = pwm.shape[1]

        # define search region
        search_start = max(0, midpoint - L)
        search_end = midpoint + L

        # extract sequence
        seq_record = genome_fasta[chrom]
        search_seq = str(seq_record.seq[search_start:search_end])

        best_score = -np.inf
        best_pos, best_strand, best_seq = None, None, None

        # sliding window
        for i in range(0, len(search_seq) - L + 1):
            window_seq = search_seq[i:i+L]

            # Watson strand
            score_w = score_sequence_with_pwm(window_seq, pwm)
            if score_w > best_score:
                best_score = score_w
                best_pos = search_start + i
                best_strand = "+"
                best_seq = window_seq

            # Crick strand (reverse complement)
            rc_seq = str(Seq(window_seq).reverse_complement())
            score_c = score_sequence_with_pwm(rc_seq, pwm)
            if score_c > best_score:
                best_score = score_c
                best_pos = search_start + i
                best_strand = "-"
                best_seq = rc_seq

        return {
            "chr": chrom,
            "start": best_pos,
            "end": best_pos + L,
            "strand": best_strand,
            "agree": 1 if best_strand == row['strand'] else 0,
            "TF": tf,
            "score": best_score,
            "best_seq": best_seq,
            "peakVal" : row['peakVal']
        }

# Example driver
def scan_bed_with_pwms(bed_file, pwm_file, genome_fasta_file):
    # load motifs
    motifDict = parameterize.getMotifsMEME(pwm_file)

    # load genome fasta into dict
    genome_fasta = SeqIO.to_dict(SeqIO.parse(genome_fasta_file, "fasta"))

    # load bed
    bed_df = pd.read_csv(bed_file, sep="\t")

    results = []
    for _, row in bed_df.iterrows():
        res = best_match_for_tf(row, motifDict, genome_fasta)
        if res is not None:   # only keep successful results
            results.append(res)

    return pd.DataFrame(results)



bob = scan_bed_with_pwms('/home/rapiduser/programs/RoboCOP/analysis/inputs/rossi_peak_w_strand_all_TFs.bed',\
                         '/home/rapiduser/programs/RoboCOP/analysis/inputs/motifs_meme.txt',\
                         '/home/rapiduser/programs/RoboCOP/analysis/inputs/SacCer3.fa'
                         )

bob.to_csv('/home/rapiduser/programs/RoboCOP/analysis/inputs/rossi_peak_w_strand_conformed_to_PWM_all_TFs_peakVal.bed', sep='\t', index=False)