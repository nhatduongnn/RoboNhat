
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pysam
import seaborn as sns
import pandas as pd
from matplotlib import collections  as mc
from Bio import SeqIO
from Bio.SeqFeature import SeqFeature, FeatureLocation
import logomaker


def to_roman(number, skip_error=False):
    """
    Convert number to roman numeral
    """
    try:
        return {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI', 7: 'VII',
         8: 'VIII', 9: 'IX', 10: 'X', 11: 'XI', 12: 'XII', 13: 'XIII', 
         14: 'XIV', 15: 'XV', 16: 'XVI'}[number]
    except KeyError:
        if skip_error: return -1
        else: raise ValueError(f"Unsupported value: {number}")
    

def from_roman(roman, skip_error=False):
    """
    Convert Roman numeral to number
    """
    try:
        return {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, 
            "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10, 
            "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, 
            "XV": 15, "XVI": 16}[roman]
    except KeyError:
        if skip_error: return -1
        else: raise ValueError(f"Unsupported value: {roman}")
            

def plot_density_scatter(x, y,  bw=(5, 15), cmap='magma_r', vmin=None, 
    vmax=None, ax=None, s=2, alpha=1., zorder=1, cbar=True):
    """
    Plot a scatter plot colored by the smoothed density of nearby points.
    """

    # Convert pandas series type data to list-like
    if type(x) == pd.core.series.Series: x = x.values
    if type(y) == pd.core.series.Series: y = y.values

    # Perform kernel density smoothing to compute a density value for each
    # point
    try:
        kde = sm.nonparametric.KDEMultivariate(data=[x, y], var_type='cc', bw=bw)
        z = kde.pdf([x, y])
    except ValueError:
        z = np.array([0] * len(x))

    # Use the default ax if none is provided
    if ax is None: ax = plt.gca()

    # Reindex the points by the sorting, higher values
    # should be drawn above lower values
    sorted_idx = np.argsort(z)
    z = z[sorted_idx]
    x = x[sorted_idx]
    y = y[sorted_idx]

    # Plot the outer border of the points in gray
    data_scatter = ax.scatter(x, y, color='none', edgecolor='#c0c0c0',
        s=2, zorder=zorder, cmap=cmap, rasterized=True)

    # plot the points
    s_plot = ax.scatter(x, y, c=z, s=3, zorder=zorder, edgecolors='none', 
        cmap=cmap, rasterized=True, vmax=vmax)
    if cbar == True:
        plt.colorbar(s_plot)

    return data_scatter


def plot_density_line(start, stop, length, ax=None):
    
    # Array of line segments as [(start x, start y), (end x, end y)]
    line_data = zip(zip(start,length), zip(stop,length))

    line_collection = mc.LineCollection(line_data, colors='blue', linewidths=1, alpha=0.05)

    # Use the default ax if none is provided
    if ax is None: ax = plt.gca()
        
    ax.add_collection(line_collection)

    return ax


def from_roman(roman, skip_error=False):
    """
    Convert Roman numeral to number
    """
    try:
        return {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, 
            "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10, 
            "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, 
            "XV": 15, "XVI": 16}[roman]
    except KeyError:
        if skip_error: return -1
        else: raise ValueError(f"Unsupported value: {roman}")
            

def plot_aggregate_density_plot(data_path, positions):
    
    samfile = pysam.AlignmentFile(data_path, "rb")
    
    # Create a matrix to store all of the aggregated data points
    h_mat = np.zeros([200,600])
    
    # Loop through each position (i.e each Abf1 site or each nucleosome dyad position), 
    # get all the reads within a 600 bp window and add it to the plot
    for i in positions.itertuples():
        for read in samfile.fetch(str(from_roman(i.chr.split('chr')[1])),i.pos-300,i.pos+300):
        #for read in samfile.fetch(str(i.chr),i.pos-300,i.pos+300):
            
            if not read.mate_is_reverse: continue

            midpoint = read.pos + 1 + int(read.template_length/2)
            dis_f_dyad = int(midpoint - i.pos)
            
            # Only if the read is less than 200 bp and is 200 bp away from the position (i.e Abf1 position or nucleosome dyad position)
            if abs(read.template_length) <= 200 and dis_f_dyad >= -200 and dis_f_dyad <= 200:
                h_mat[abs(read.template_length)-1,dis_f_dyad+300-1] += 1
                

    h_mat2 = h_mat[:,100:500]
    ax = sns.heatmap(h_mat2, cmap = 'coolwarm' )
    ax.invert_yaxis()
    
    plt.xticks(range(0,420,20),range(-200,220,20))

    plt.title( "2-D Heat Map of DMS-seq fraq around 2000 well positioned nucs" )

    #plt.show()
    
    return ax,h_mat


# def plot_motif_logo(fasta_file_dir, TF_start_pos_original, seq_width, shift, ax1, ax2, fasta_type='number', bases='AG'):
#     """
#     Plot sequence motif logos (probability and information content) centered around specified genomic positions.

#     This function extracts sequences from a FASTA file around given genomic coordinates,
#     calculates base probabilities and information content, and generates sequence logos 
#     showing the distributions of selected bases (A/G or A only or G only) on positive and negative strands.

#     Parameters
#     ----------
#     fasta_file_dir : str
#         Path to the FASTA file containing genome sequences.

#     TF_start_pos_original : pandas.DataFrame
#         A DataFrame with at least two columns: 
#         'chr' (chromosome identifier) and 'pos' (genomic position) around which to extract sequence windows.

#     seq_width : int
#         Width of the sequence window (in base pairs) centered at each position.

#     shift : int
#         Amount to shift each input position before extracting the sequence window.

#     ax1 : matplotlib.axes.Axes
#         Matplotlib axis object for plotting the probability logo.

#     ax2 : matplotlib.axes.Axes
#         Matplotlib axis object for plotting the information content (bit) logo.

#     fasta_type : str, optional, default='number'
#         Format of chromosome names in the FASTA file:
#         - 'roman' : e.g., 'chrI', 'chrII', etc.
#         - 'number' : e.g., 'chr1', 'chr2', etc.

#     bases : str, optional, default='AG'
#         Bases to display in the logos:
#         - 'AG' : Plot A/G bases upwards and C/T bases downwards.
#         - 'A'  : Plot only A bases upwards (flip T downwards).
#         - 'G'  : Plot only G bases upwards (flip C downwards).

#     Returns
#     -------
#     seq_df : pandas.DataFrame
#         DataFrame containing the extracted sequence windows.

#     seq_df_base_prob : pandas.DataFrame
#         DataFrame containing the per-position base probabilities (before strand flipping).

#     prob_logo : logomaker.Logo
#         Logomaker object for the probability logo.

#     bit_logo : logomaker.Logo
#         Logomaker object for the information content (bit) logo.

#     Notes
#     -----
#     - Positions that extend beyond chromosome boundaries are skipped with a warning printed.
#     - Base probabilities for C/T bases are flipped (negative values) to represent the complementary strand.
#     - Colors are assigned (blue for A/G, orange for C/T) with custom mapping.
#     - Logomaker is used for logo plotting, and fonts are set to 'Arial Rounded MT Bold'.
#     - Mirror-flip effects are applied for better visualization of complementary bases.

#     Dependencies
#     ------------
#     - Biopython (for SeqIO, SeqFeature, FeatureLocation)
#     - numpy
#     - pandas
#     - logomaker
#     - matplotlib

#     """

    
#     ## Create an empty dict to store all chromosome of the fasta file
#     fasta_file = {}
#     for seq_record in SeqIO.parse(fasta_file_dir, "fasta"):
#         if fasta_type == 'roman':
#             fasta_file[seq_record.id] = seq_record.seq
#         elif fasta_type == 'number':
#             if seq_record.id != 'chrM':
#                 fasta_file[from_roman(seq_record.id.split('chr')[1])] = seq_record.seq
#             elif seq_record.id == 'chrM':
#                 fasta_file[seq_record.id.split('chr')[1]] = seq_record.seq
        
#     ## Adjust for shift, so that we can center the plot around a certain base pair
#     TF_start_pos = TF_start_pos_original.copy()
#     TF_start_pos['pos'] = TF_start_pos['pos'] + shift
        
#     ## Create an empty matrix to store extracted sequences
#     seq_mat = np.zeros([TF_start_pos.shape[0],seq_width+1],dtype='str')
    
#     ## Iterate through all position values and extract sequences around that position
#     for i in TF_start_pos.reset_index(drop=True).itertuples():
#         start_wind =  i.pos-int(seq_width/2)-1
#         end_wind = i.pos+int(seq_width/2)
#         # Check if the window to be created by SeqFeature will be negative or larger than chromosome size
#         if start_wind <= 0 or end_wind > len(fasta_file[i.chr])-1:
#             print("chromosome is {}, position is {} and range is from {} to {}".format(i.chr,i.pos,start_wind,end_wind))
#             continue
#         else:
#             seq = SeqFeature(FeatureLocation(i.pos-int(seq_width/2)-1, i.pos+int(seq_width/2)), type="gene", strand=1).extract(fasta_file[i.chr])
#             #print(seq)
#             seq_mat[i.Index,:] = np.array(seq)
        
        
#     # Turn the extracted sequences into a DataFrame
#     seq_df = pd.DataFrame(seq_mat)
#     # Fill in missing values as a count of 0
#     seq_df_base_count = seq_df.apply(lambda x: x.value_counts().reindex(['A','C','G','T'],fill_value=0))
#     # Calculate the probability of each base at each bp position
#     seq_df_base_prob = seq_df_base_count.apply(lambda x: x/seq_df.shape[0])
    
#     # Modify the probability to plot the bases that we want
#     # Multiply by -1 to flip the C and T probabilities to the opposite strrand
#     seq_df_base_prob_AG_top_CT_bot = seq_df_base_prob.copy()
#     if bases == 'AG':
#         seq_df_base_prob_AG_top_CT_bot.loc["C"] = seq_df_base_prob_AG_top_CT_bot.loc["C"]*-1
#         seq_df_base_prob_AG_top_CT_bot.loc["T"] = seq_df_base_prob_AG_top_CT_bot.loc["T"]*-1
#     # Assign probability values to 0 if we don't want those bases plotted
#     elif bases == 'A':
#         seq_df_base_prob_AG_top_CT_bot.loc["G"] = 0
#         seq_df_base_prob_AG_top_CT_bot.loc["C"] = 0
#         seq_df_base_prob_AG_top_CT_bot.loc["T"] = seq_df_base_prob_AG_top_CT_bot.loc["T"]*-1
#     elif bases == 'G':
#         seq_df_base_prob_AG_top_CT_bot.loc["A"] = 0
#         seq_df_base_prob_AG_top_CT_bot.loc["T"] = 0
#         seq_df_base_prob_AG_top_CT_bot.loc["C"] = seq_df_base_prob_AG_top_CT_bot.loc["C"]*-1


#     ## Plot the probability plot
#     # create color scheme
#     color_scheme = {
#         'A' : 'blue',
#         'C' : 'orange',
#         'G' : 'blue',
#         'T' : 'orange'
#     }

    
#     # create Logo object
#     prob_logo = logomaker.Logo(seq_df_base_prob_AG_top_CT_bot.T,
#                                ax=ax1,
#                               shade_below=0,
#                               fade_below=0,
#                               color_scheme = color_scheme,
#                               font_name='Arial Rounded MT Bold')


#     # style using Logo methods
#     prob_logo.style_spines(visible=False)
#     prob_logo.style_spines(spines=['left', 'bottom'], visible=True)
#     prob_logo.style_xticks(rotation=90, fmt='%d', anchor=0)

#     # style using Axes methods
#     prob_logo.ax.set_ylabel("Probability", labelpad=10)
#     prob_logo.ax.yaxis.set_tick_params(pad=5)
#     prob_logo.ax.set_yticks(np.linspace(-1,1,5))
#     prob_logo.ax.set_yticklabels('%.1f'%x for x in abs(np.linspace(-1,1,5)))
#     prob_logo.ax.xaxis.set_ticks_position('none')
#     prob_logo.ax.xaxis.set_tick_params(pad=-1)
#     prob_logo.ax.set_xticks(range(0,seq_width+1,5))
#     prob_logo.ax.set_xticklabels('%d'%x for x in range(-1*int(seq_width/2),int(seq_width/2)+1,5))
#     prob_logo.style_spines(spines=['left', 'bottom'], visible=True)
    
#     for i in range(len(prob_logo.glyph_df['T'])):
#         prob_logo.glyph_df['T'][i].c = 'A'
#         prob_logo.glyph_df['C'][i].c = 'G'

    
#     prob_logo.style_glyphs_below(flip=True,mirror=True)

    
#     ## Further process the probability dataframe to get the information(bit) dataframe
#     max_information = 4*-0.25*np.log2(0.25)
#     max_info_per_base = max_information - (seq_df_base_prob.apply(lambda x: -x*np.log2(x))).sum(axis=0)
    
#     seq_df_base_prob_w_max_info = pd.concat([seq_df_base_prob,pd.DataFrame(max_info_per_base).T],axis=0)
    
#     seq_df_base_information = seq_df_base_prob_w_max_info.apply(lambda x: x*x.iloc[4]).iloc[:4,:]
    
#     seq_df_base_information_AG_top_CT_bot = seq_df_base_information.copy()
#     # Modify the probability to plot the bases that we want
#     # Multiply by -1 to flip the C and T probabilities to the opposite strrand
#     if bases == 'AG':
#         seq_df_base_information_AG_top_CT_bot.loc["C"] = seq_df_base_information_AG_top_CT_bot.loc["C"]*-1
#         seq_df_base_information_AG_top_CT_bot.loc["T"] = seq_df_base_information_AG_top_CT_bot.loc["T"]*-1
#     # Assign probability values to 0 if we don't want those bases plotted
#     elif bases == 'A':
#         seq_df_base_information_AG_top_CT_bot.loc["G"] = 0
#         seq_df_base_information_AG_top_CT_bot.loc["C"] = 0
#         seq_df_base_information_AG_top_CT_bot.loc["T"] = seq_df_base_information_AG_top_CT_bot.loc["T"]*-1
#     elif bases == 'G':
#         seq_df_base_information_AG_top_CT_bot.loc["A"] = 0
#         seq_df_base_information_AG_top_CT_bot.loc["T"] = 0
#         seq_df_base_information_AG_top_CT_bot.loc["C"] = seq_df_base_information_AG_top_CT_bot.loc["C"]*-1
    
    
#     ## Plot the information(bit) dataframe
#     # create Logo object
#     bit_logo = logomaker.Logo(seq_df_base_information_AG_top_CT_bot.T,
#                               ax=ax2,
#                               shade_below=0,
#                               fade_below=0,
#                               color_scheme = color_scheme,
#                               font_name='Arial Rounded MT Bold')


#     # style using Logo methods
#     bit_logo.style_spines(visible=False)
#     bit_logo.style_spines(spines=['left', 'bottom'], visible=True)
#     bit_logo.style_xticks(rotation=90, fmt='%d', anchor=0)

#     # style using Axes methods
#     bit_logo.ax.set_ylabel("Bit", labelpad=10)
#     bit_logo.ax.yaxis.set_tick_params(pad=5)
#     bit_logo.ax.set_yticks(np.linspace(-2,2,5))
#     bit_logo.ax.set_yticklabels('%d'%x for x in abs(np.linspace(-2,2,5)))
#     bit_logo.ax.xaxis.set_ticks_position('none')
#     bit_logo.ax.xaxis.set_tick_params(pad=-1)
#     bit_logo.ax.set_xticks(range(0,seq_width+1,5))
#     bit_logo.ax.set_xticklabels('%d'%x for x in range(-1*int(seq_width/2),int(seq_width/2)+1,5))
    
#     for i in range(len(bit_logo.glyph_df['T'])):
#         bit_logo.glyph_df['T'][i].c = 'A'
#         bit_logo.glyph_df['C'][i].c = 'G'

    
#     bit_logo.style_glyphs_below(flip=True,mirror=True)


#     #return fig,ax
#     return seq_df,seq_df_base_prob,prob_logo,bit_logo


# def plot_motif_logo(
#     fasta_file_dir,
#     TF_start_pos_original,
#     seq_width,
#     shift,
#     ax1=None,
#     ax2=None,
#     fasta_type='number',
#     bases='AG',
#     font_scale=1.0
# ):
#     """
#     Plot motif probability/information logo on provided axes.
#     font_scale: scales axis/tick label font size (default=1.0).
#     Now accounts for strand field: negative strand windows are reverse complemented.
#     TF_start_pos_original: DataFrame with 'Chromosome', 'pos' 1 indexed, and optionally 'Strand' columns.
#     """
#     # Set base font sizes
#     base_fontsize = 11 * font_scale
#     label_size = int(base_fontsize * 1.2)
#     tick_size = int(base_fontsize * 0.95)
#     fasta_file = {}
#     for seq_record in SeqIO.parse(fasta_file_dir, "fasta"):
#         if fasta_type == 'roman':
#             fasta_file[seq_record.id] = seq_record.seq
#         elif fasta_type == 'number':
#             if seq_record.id != 'chrM':
#                 # you'll need to define from_roman if using this:
#                 fasta_file[from_roman(seq_record.id.split('chr')[1])] = seq_record.seq
#             elif seq_record.id == 'chrM':
#                 fasta_file[seq_record.id.split('chr')[1]] = seq_record.seq
#     TF_start_pos = TF_start_pos_original.copy()
#     TF_start_pos['pos'] = TF_start_pos['pos'] + shift
#     seq_mat = np.zeros([TF_start_pos.shape[0], seq_width + 1], dtype='str')
#     for i in TF_start_pos.reset_index(drop=True).itertuples():
#         start_wind = i.pos - int(seq_width / 2) - 1
#         end_wind = i.pos + int(seq_width / 2)
#         if start_wind <= 0 or end_wind > len(fasta_file[i.Chromosome]) - 1:
#             print("chromosome is {}, position is {} and range is from {} to {}".format(i.Chromosome, i.pos, start_wind, end_wind))
#             continue
#         else:
#             seq = fasta_file[i.Chromosome][start_wind:end_wind]
#             # Key modification: handle strand!
#             strand_val = getattr(i, 'Strand', '+')  # default '+'
#             if str(strand_val) in ['-', '-1', -1]:
#                 seq = seq.reverse_complement()
#             seq_mat[i.Index, :] = np.array(seq)
#     seq_df = pd.DataFrame(seq_mat)
#     seq_df_base_count = seq_df.apply(lambda x: x.value_counts().reindex(['A','C','G','T'], fill_value=0))
#     seq_df_base_prob = seq_df_base_count.apply(lambda x: x/seq_df.shape[0])
#     seq_df_base_prob_AG_top_CT_bot = seq_df_base_prob.copy()
#     if bases == 'AG':
#         seq_df_base_prob_AG_top_CT_bot.loc["C"] *= -1
#         seq_df_base_prob_AG_top_CT_bot.loc["T"] *= -1
#         color_scheme = {'A': 'blue', 'C': 'orange', 'G': 'blue', 'T': 'orange'}
#     elif bases == 'A':
#         seq_df_base_prob_AG_top_CT_bot.loc["G"] = 0
#         seq_df_base_prob_AG_top_CT_bot.loc["C"] = 0
#         seq_df_base_prob_AG_top_CT_bot.loc["T"] *= -1
#         color_scheme = {'A': 'blue', 'C': 'orange', 'G': 'blue', 'T': 'orange'}
#     elif bases == 'G':
#         seq_df_base_prob_AG_top_CT_bot.loc["A"] = 0
#         seq_df_base_prob_AG_top_CT_bot.loc["T"] = 0
#         seq_df_base_prob_AG_top_CT_bot.loc["C"] *= -1
#         color_scheme = {'A': 'blue', 'C': 'orange', 'G': 'blue', 'T': 'orange'}
#     elif bases == 'ACGT':
#         color_scheme = {'A': 'blue', 'C': 'green', 'G': 'orange', 'T': 'red'}
    
    
#     prob_logo = None
#     if ax1 is not None:
#         prob_logo = logomaker.Logo(
#             seq_df_base_prob_AG_top_CT_bot.T,
#             ax=ax1,
#             shade_below=0,
#             fade_below=0,
#             color_scheme=color_scheme,
#             font_name='Arial Rounded MT Bold'
#         )
#         prob_logo.style_spines(visible=False)
#         prob_logo.style_spines(spines=['left', 'bottom'], visible=True)
#         prob_logo.style_xticks(rotation=0, fmt='%d', anchor=0)
#         prob_logo.ax.set_ylabel("Probability", fontsize=label_size)
#         prob_logo.ax.yaxis.set_tick_params(pad=5, labelsize=tick_size)
#         prob_logo.ax.xaxis.set_tick_params(pad=-1, labelsize=tick_size)
#         prob_logo.ax.set_yticks(np.linspace(-1, 1, 5))
#         prob_logo.ax.set_yticklabels(['%.1f' % x for x in abs(np.linspace(-1, 1, 5))], fontsize=tick_size)

#         ## Set x ticks
#         num_desired_ticks = 9
#         tick_indices = np.linspace(0, seq_width, num_desired_ticks, dtype=int)
#         tick_labels = np.linspace(-int(seq_width/2), int(seq_width/2), num_desired_ticks, dtype=int)
#         prob_logo.ax.set_xticks(tick_indices)
#         prob_logo.ax.set_xticklabels([f'{x}' for x in tick_labels], fontsize=tick_size)


#         for i in range(len(prob_logo.glyph_df['T'])):
#             prob_logo.glyph_df['T'][i].c = 'A'
#             prob_logo.glyph_df['C'][i].c = 'G'
#         prob_logo.style_glyphs_below(flip=True, mirror=True)

#     max_information = 4 - 0.25 * np.log2(0.25)
#     max_info_per_base = max_information - (seq_df_base_prob.apply(lambda x: -x*np.log2(x+1e-9))).sum(axis=0)
#     seq_df_base_prob_w_max_info = pd.concat([seq_df_base_prob, pd.DataFrame(max_info_per_base).T], axis=0)
#     seq_df_base_information = seq_df_base_prob_w_max_info.apply(lambda x: x*x.iloc[4]).iloc[:4,:]
#     seq_df_base_information_AG_top_CT_bot = seq_df_base_information.copy()
#     if bases == 'AG':
#         seq_df_base_information_AG_top_CT_bot.loc["C"] *= -1
#         seq_df_base_information_AG_top_CT_bot.loc["T"] *= -1
#     elif bases == 'A':
#         seq_df_base_information_AG_top_CT_bot.loc["G"] = 0
#         seq_df_base_information_AG_top_CT_bot.loc["C"] = 0
#         seq_df_base_information_AG_top_CT_bot.loc["T"] *= -1
#     elif bases == 'G':
#         seq_df_base_information_AG_top_CT_bot.loc["A"] = 0
#         seq_df_base_information_AG_top_CT_bot.loc["T"] = 0
#         seq_df_base_information_AG_top_CT_bot.loc["C"] *= -1
#     bit_logo = None
#     if ax2 is not None:
#         bit_logo = logomaker.Logo(
#             seq_df_base_information_AG_top_CT_bot.T,
#             ax=ax2,
#             shade_below=0,
#             fade_below=0,
#             color_scheme=color_scheme,
#             font_name='Arial Rounded MT Bold'
#         )
#         bit_logo.style_spines(visible=False)
#         bit_logo.style_spines(spines=['left', 'bottom'], visible=True)
#         bit_logo.style_xticks(rotation=0, fmt='%d', anchor=0)
#         bit_logo.ax.set_ylabel("Bit", fontsize=label_size)
#         bit_logo.ax.yaxis.set_tick_params(pad=5, labelsize=tick_size)
#         bit_logo.ax.xaxis.set_tick_params(pad=-1, labelsize=tick_size)
#         bit_logo.ax.set_yticks(np.linspace(-2, 2, 5))
#         bit_logo.ax.set_yticklabels(['%d' % x for x in abs(np.linspace(-2, 2, 5))], fontsize=tick_size)
#         bit_logo.ax.set_xticks(tick_indices)
#         bit_logo.ax.set_xticklabels([f'{x}' for x in tick_labels], fontsize=tick_size)
#         for i in range(len(bit_logo.glyph_df['T'])):
#             bit_logo.glyph_df['T'][i].c = 'A'
#             bit_logo.glyph_df['C'][i].c = 'G'
#         bit_logo.style_glyphs_below(flip=True, mirror=True)
#     return seq_df, seq_df_base_prob, prob_logo, bit_logo

# def plot_motif_logo(
#     fasta_file_dir,
#     TF_start_pos_original,
#     seq_width,
#     shift,
#     ax1=None,
#     ax2=None,
#     fasta_type='number',
#     bases='AG',
#     font_scale=1.0,
#     plot_negative_axis=True   ### NEW OPTION
# ):
#     """
#     Plot motif probability/information logo on provided axes.
#     font_scale: scales axis/tick label font size (default=1.0).
#     Now accounts for strand field: negative strand windows are reverse complemented.
#     TF_start_pos_original: DataFrame with 'Chromosome', 'pos' 1 indexed, and optionally 'Strand' columns.
#     plot_negative_axis: if False, only positive y-axis is plotted.
#     """

#     # Set base font sizes
#     base_fontsize = 11 * font_scale
#     label_size = int(base_fontsize * 1.2)
#     tick_size = int(base_fontsize * 0.95)

#     fasta_file = {}
#     for seq_record in SeqIO.parse(fasta_file_dir, "fasta"):
#         if fasta_type == 'roman':
#             fasta_file[seq_record.id] = seq_record.seq
#         elif fasta_type == 'number':
#             if seq_record.id != 'chrM':
#                 fasta_file[from_roman(seq_record.id.split('chr')[1])] = seq_record.seq
#             elif seq_record.id == 'chrM':
#                 fasta_file[seq_record.id.split('chr')[1]] = seq_record.seq

#     TF_start_pos = TF_start_pos_original.copy()
#     TF_start_pos['pos'] = TF_start_pos['pos'] + shift
#     seq_mat = np.zeros([TF_start_pos.shape[0], seq_width + 1], dtype='str')
#     for i in TF_start_pos.reset_index(drop=True).itertuples():
#         start_wind = i.pos - int(seq_width / 2) - 1
#         end_wind = i.pos + int(seq_width / 2)
#         if start_wind <= 0 or end_wind > len(fasta_file[i.Chromosome]) - 1:
#             print(f"chromosome is {i.Chromosome}, position is {i.pos} and range is from {start_wind} to {end_wind}")
#             continue
#         else:
#             seq = fasta_file[i.Chromosome][start_wind:end_wind]
#             strand_val = getattr(i, 'Strand', '+')  # default '+'
#             if str(strand_val) in ['-', '-1', -1]:
#                 seq = seq.reverse_complement()
#             seq_mat[i.Index, :] = np.array(seq)

#     seq_df = pd.DataFrame(seq_mat)
#     seq_df_base_count = seq_df.apply(lambda x: x.value_counts().reindex(['A','C','G','T'], fill_value=0))
#     seq_df_base_prob = seq_df_base_count.apply(lambda x: x/seq_df.shape[0])

#     seq_df_base_prob_mod = seq_df_base_prob.copy()

#     # === Apply base flipping logic only if negative axis is desired ===
#     if plot_negative_axis:
#         if bases == 'AG':
#             seq_df_base_prob_mod.loc["C"] *= -1
#             seq_df_base_prob_mod.loc["T"] *= -1
#             color_scheme = {'A': 'blue', 'C': 'orange', 'G': 'blue', 'T': 'orange'}
#         elif bases == 'A':
#             seq_df_base_prob_mod.loc["G"] = 0
#             seq_df_base_prob_mod.loc["C"] = 0
#             seq_df_base_prob_mod.loc["T"] *= -1
#             color_scheme = {'A': 'blue', 'C': 'orange', 'G': 'blue', 'T': 'orange'}
#         elif bases == 'G':
#             seq_df_base_prob_mod.loc["A"] = 0
#             seq_df_base_prob_mod.loc["T"] = 0
#             seq_df_base_prob_mod.loc["C"] *= -1
#             color_scheme = {'A': 'blue', 'C': 'orange', 'G': 'blue', 'T': 'orange'}
#         elif bases == 'ACGT':
#             color_scheme = {'A': 'blue', 'C': 'green', 'G': 'orange', 'T': 'red'}
#     else:
#         # Positive only: just normal colors
#         color_scheme = {'A': 'blue', 'C': 'green', 'G': 'orange', 'T': 'red'}

#     prob_logo = None
#     if ax1 is not None:
#         prob_logo = logomaker.Logo(
#             seq_df_base_prob_mod.T,
#             ax=ax1,
#             shade_below=0,
#             fade_below=0,
#             color_scheme=color_scheme,
#             font_name='Arial Rounded MT Bold'
#         )
#         prob_logo.style_spines(visible=False)
#         prob_logo.style_spines(spines=['left', 'bottom'], visible=True)
#         prob_logo.style_xticks(rotation=0, fmt='%d', anchor=0)
#         prob_logo.ax.set_ylabel("Probability", fontsize=label_size)
#         prob_logo.ax.yaxis.set_tick_params(pad=5, labelsize=tick_size)
#         prob_logo.ax.xaxis.set_tick_params(pad=-1, labelsize=tick_size)

#         if plot_negative_axis:
#             prob_logo.ax.set_yticks(np.linspace(-1, 1, 5))
#             prob_logo.ax.set_yticklabels([f'{abs(x):.1f}' for x in np.linspace(-1, 1, 5)], fontsize=tick_size)
#             for i in range(len(prob_logo.glyph_df['T'])):
#                 prob_logo.glyph_df['T'][i].c = 'A'
#                 prob_logo.glyph_df['C'][i].c = 'G'
#             prob_logo.style_glyphs_below(flip=True, mirror=True)
#         else:
#             prob_logo.ax.set_yticks(np.linspace(0, 1, 6))
#             prob_logo.ax.set_yticklabels([f'{x:.1f}' for x in np.linspace(0, 1, 6)], fontsize=tick_size)

#         # X ticks
#         num_desired_ticks = 9
#         tick_indices = np.linspace(0, seq_width, num_desired_ticks, dtype=int)
#         tick_labels = np.linspace(-int(seq_width/2), int(seq_width/2), num_desired_ticks, dtype=int)
#         prob_logo.ax.set_xticks(tick_indices)
#         prob_logo.ax.set_xticklabels([f'{x}' for x in tick_labels], fontsize=tick_size)

#     # === Info logo ===
#     max_information = 4 - 0.25 * np.log2(0.25)
#     max_info_per_base = max_information - (seq_df_base_prob.apply(lambda x: -x*np.log2(x+1e-9))).sum(axis=0)
#     seq_df_base_prob_w_max_info = pd.concat([seq_df_base_prob, pd.DataFrame(max_info_per_base).T], axis=0)
#     seq_df_base_information = seq_df_base_prob_w_max_info.apply(lambda x: x*x.iloc[4]).iloc[:4,:]

#     seq_df_base_information_mod = seq_df_base_information.copy()
#     if plot_negative_axis:
#         if bases == 'AG':
#             seq_df_base_information_mod.loc["C"] *= -1
#             seq_df_base_information_mod.loc["T"] *= -1
#         elif bases == 'A':
#             seq_df_base_information_mod.loc["G"] = 0
#             seq_df_base_information_mod.loc["C"] = 0
#             seq_df_base_information_mod.loc["T"] *= -1
#         elif bases == 'G':
#             seq_df_base_information_mod.loc["A"] = 0
#             seq_df_base_information_mod.loc["T"] = 0
#             seq_df_base_information_mod.loc["C"] *= -1

#     bit_logo = None
#     if ax2 is not None:
#         bit_logo = logomaker.Logo(
#             seq_df_base_information_mod.T,
#             ax=ax2,
#             shade_below=0,
#             fade_below=0,
#             color_scheme=color_scheme,
#             font_name='Arial Rounded MT Bold'
#         )
#         bit_logo.style_spines(visible=False)
#         bit_logo.style_spines(spines=['left', 'bottom'], visible=True)
#         bit_logo.style_xticks(rotation=0, fmt='%d', anchor=0)
#         bit_logo.ax.set_ylabel("Bit", fontsize=label_size)
#         bit_logo.ax.yaxis.set_tick_params(pad=5, labelsize=tick_size)
#         bit_logo.ax.xaxis.set_tick_params(pad=-1, labelsize=tick_size)

#         if plot_negative_axis:
#             bit_logo.ax.set_yticks(np.linspace(-2, 2, 5))
#             bit_logo.ax.set_yticklabels([f'{abs(int(x))}' for x in np.linspace(-2, 2, 5)], fontsize=tick_size)
#             for i in range(len(bit_logo.glyph_df['T'])):
#                 bit_logo.glyph_df['T'][i].c = 'A'
#                 bit_logo.glyph_df['C'][i].c = 'G'
#             bit_logo.style_glyphs_below(flip=True, mirror=True)
#         else:
#             # Positive only
#             ymax = float(seq_df_base_information.max().max()) * 1.1
#             bit_logo.ax.set_ylim(0, ymax)
#             bit_logo.ax.set_yticks(np.linspace(0, np.ceil(ymax), 5))
#             bit_logo.ax.set_yticklabels([f'{int(x)}' for x in np.linspace(0, np.ceil(ymax), 5)], fontsize=tick_size)

#         bit_logo.ax.set_xticks(tick_indices)
#         bit_logo.ax.set_xticklabels([f'{x}' for x in tick_labels], fontsize=tick_size)

#     return seq_df, seq_df_base_prob, prob_logo, bit_logo



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logomaker
from Bio import SeqIO

def plot_motif_logo(
    fasta_file_dir,
    TF_start_pos_original,
    seq_width,
    shift,
    ax1=None,
    ax2=None,
    fasta_type='number',
    bases='AG',
    font_scale=1.0,
    plot_negative_axis=True,
    box_color='lightgray',
    box_alpha=0.3
):
    """
    Plot motif probability/information logo on provided axes.
    Parameters
    ----------
    fasta_file_dir : str
        Path to fasta reference genome
    TF_start_pos_original : DataFrame
        Must contain 'Chromosome', 'pos', 'start', 'end', 
        and optionally 'Strand'.
        For + strand: pos is the motif center as usual.
        For - strand: pos is set to the 'end' coordinate.
    seq_width : int
        Window size around center to extract
    shift : int
        Extra shift for motif center
    ax1, ax2 : matplotlib Axes
        If not None, plots probability (ax1) and info logo (ax2)
    fasta_type : str
        How chromosomes are named ('number' or 'roman')
    bases : str
        Which bases to emphasize ('AG','A','G','ACGT')
    font_scale : float
        Scaling of tick/label fonts
    plot_negative_axis : bool
        If False, plot only positive axis (no flipping below 0)
    box_color : str
        Color of region highlight box
    box_alpha : float
        Transparency of region highlight box
    """

    # --- setup fonts ---
    base_fontsize = 11 * font_scale
    label_size = int(base_fontsize * 1.2)
    tick_size = int(base_fontsize * 0.95)

    # --- strand-aware box computation ---
    def _compute_relative_box(TF_df, seq_width):
        """Compute relative highlight coordinates accounting for strand orientation."""
        rel_coords = []
        for row in TF_df.itertuples():
            pos = row.pos
            strand_val = getattr(row, 'Strand', '+')
            if str(strand_val) in ['+', '1', '+1']:
                rel_start = row.start - pos
                rel_end   = row.end   - pos
            else:
                # Negative strand: invert relative orientation
                rel_start = pos - row.end
                rel_end   = pos - row.start
            # enforce ordering
            rel_start, rel_end = min(rel_start, rel_end), max(rel_start, rel_end)
            rel_coords.append((rel_start, rel_end))
        # Take median across occurrences
        rel_start = np.median([c[0] for c in rel_coords])
        rel_end   = np.median([c[1] for c in rel_coords])
        # Shift into logo-coords (center = seq_width/2)
        center = int(seq_width/2)
        return rel_start+center, rel_end+center

    # --- load FASTA ---
    fasta_file = {}
    for seq_record in SeqIO.parse(fasta_file_dir, "fasta"):
        if fasta_type == 'roman':
            fasta_file[seq_record.id] = seq_record.seq
        elif fasta_type == 'number':
            if seq_record.id != 'chrM':
                fasta_file[int(seq_record.id.split('chr')[1])] = seq_record.seq
            else:
                fasta_file[seq_record.id.split('chr')[1]] = seq_record.seq

    # --- adjust dataframe ---
    TF_start_pos = TF_start_pos_original.copy()
    TF_start_pos['pos'] = TF_start_pos['pos'] + shift

    seq_mat = np.zeros([TF_start_pos.shape[0], seq_width + 1], dtype='str')
    for i in TF_start_pos.reset_index(drop=True).itertuples():
        start_wind = i.pos - int(seq_width / 2) - 1
        end_wind   = i.pos + int(seq_width / 2)
        if start_wind <= 0 or end_wind > len(fasta_file[i.Chromosome]) - 1:
            print(f"chromosome {i.Chromosome}, position {i.pos} range {start_wind}:{end_wind}")
            continue
        else:
            seq = fasta_file[i.Chromosome][start_wind:end_wind]
            strand_val = getattr(i, 'Strand', '+')  # default '+'
            if str(strand_val) in ['-', '-1', -1]:
                seq = seq.reverse_complement()
            seq_mat[i.Index, :] = np.array(seq)

    # --- probability matrix ---
    seq_df = pd.DataFrame(seq_mat)
    seq_df_base_count = seq_df.apply(lambda x: x.value_counts().reindex(['A','C','G','T'], fill_value=0))
    seq_df_base_prob = seq_df_base_count.apply(lambda x: x/seq_df.shape[0])
    seq_df_base_prob_mod = seq_df_base_prob.copy()

    # --- Colors & flipping for negative ---
    if plot_negative_axis:
        if bases == 'AG':
            seq_df_base_prob_mod.loc["C"] *= -1
            seq_df_base_prob_mod.loc["T"] *= -1
            color_scheme = {'A': 'blue', 'C': 'orange', 'G': 'blue', 'T': 'orange'}
        elif bases == 'A':
            seq_df_base_prob_mod.loc["G"] = 0
            seq_df_base_prob_mod.loc["C"] = 0
            seq_df_base_prob_mod.loc["T"] *= -1
            color_scheme = {'A': 'blue', 'C': 'orange', 'G': 'blue', 'T': 'orange'}
        elif bases == 'G':
            seq_df_base_prob_mod.loc["A"] = 0
            seq_df_base_prob_mod.loc["T"] = 0
            seq_df_base_prob_mod.loc["C"] *= -1
            color_scheme = {'A': 'blue', 'C': 'orange', 'G': 'blue', 'T': 'orange'}
        elif bases == 'ACGT':
            color_scheme = {'A': 'blue', 'C': 'green', 'G': 'orange', 'T': 'red'}
    else:
        color_scheme = {'A': 'blue', 'C': 'green', 'G': 'orange', 'T': 'red'}

    # ticks
    num_desired_ticks = 9
    tick_indices = np.linspace(0, seq_width, num_desired_ticks, dtype=int)
    tick_labels  = np.linspace(-int(seq_width/2), int(seq_width/2), num_desired_ticks, dtype=int)

    # --- Probability logo ---
    prob_logo = None
    if ax1 is not None:
        prob_logo = logomaker.Logo(
            seq_df_base_prob_mod.T, ax=ax1,
            shade_below=0, fade_below=0,
            color_scheme=color_scheme,
            font_name='Arial Rounded MT Bold'
        )
        prob_logo.style_spines(visible=False)
        prob_logo.style_spines(spines=['left','bottom'], visible=True)
        prob_logo.style_xticks(rotation=0, fmt='%d', anchor=0)
        prob_logo.ax.set_ylabel("Probability", fontsize=label_size)
        prob_logo.ax.yaxis.set_tick_params(pad=5, labelsize=tick_size)
        prob_logo.ax.xaxis.set_tick_params(pad=-1, labelsize=tick_size)
        prob_logo.ax.set_xticks(tick_indices)
        prob_logo.ax.set_xticklabels([f'{x}' for x in tick_labels], fontsize=tick_size)
        if plot_negative_axis:
            prob_logo.ax.set_yticks(np.linspace(-1,1,5))
            prob_logo.ax.set_yticklabels([f'{abs(x):.1f}' for x in np.linspace(-1,1,5)], fontsize=tick_size)
            for i in range(len(prob_logo.glyph_df['T'])):
                prob_logo.glyph_df['T'][i].c = 'A'
                prob_logo.glyph_df['C'][i].c = 'G'
            prob_logo.style_glyphs_below(flip=True, mirror=True)
        else:
            prob_logo.ax.set_ylim(0,1)
            prob_logo.ax.set_yticks(np.linspace(0,1,6))
            prob_logo.ax.set_yticklabels([f'{x:.1f}' for x in np.linspace(0,1,6)], fontsize=tick_size)

        # --- add highlight box ---
        if {'start','end'}.issubset(TF_start_pos.columns):
            box_left, box_right = _compute_relative_box(TF_start_pos, seq_width)
            rect = patches.Rectangle(
                (box_left, prob_logo.ax.get_ylim()[0]),
                (box_right - box_left),
                prob_logo.ax.get_ylim()[1]-prob_logo.ax.get_ylim()[0],
                facecolor=box_color, alpha=box_alpha
            )
            prob_logo.ax.add_patch(rect)

    # --- Information content logo ---
    max_information = 4 - 0.25*np.log2(0.25)
    max_info_per_base = max_information - (seq_df_base_prob.apply(lambda x: -x*np.log2(x+1e-9))).sum(axis=0)
    seq_df_base_prob_w_max_info = pd.concat([seq_df_base_prob, pd.DataFrame(max_info_per_base).T], axis=0)
    seq_df_base_information = seq_df_base_prob_w_max_info.apply(lambda x: x*x.iloc[4]).iloc[:4,:]
    seq_df_base_information_mod = seq_df_base_information.copy()
    if plot_negative_axis:
        if bases == 'AG':
            seq_df_base_information_mod.loc["C"] *= -1
            seq_df_base_information_mod.loc["T"] *= -1
        elif bases == 'A':
            seq_df_base_information_mod.loc["G"] = 0
            seq_df_base_information_mod.loc["C"] = 0
            seq_df_base_information_mod.loc["T"] *= -1
        elif bases == 'G':
            seq_df_base_information_mod.loc["A"] = 0
            seq_df_base_information_mod.loc["T"] = 0
            seq_df_base_information_mod.loc["C"] *= -1

    bit_logo = None
    if ax2 is not None:
        bit_logo = logomaker.Logo(
            seq_df_base_information_mod.T, ax=ax2,
            shade_below=0, fade_below=0,
            color_scheme=color_scheme,
            font_name='Arial Rounded MT Bold'
        )
        bit_logo.style_spines(visible=False)
        bit_logo.style_spines(spines=['left','bottom'], visible=True)
        bit_logo.style_xticks(rotation=0, fmt='%d', anchor=0)
        bit_logo.ax.set_ylabel("Bit", fontsize=label_size)
        bit_logo.ax.yaxis.set_tick_params(pad=5, labelsize=tick_size)
        bit_logo.ax.xaxis.set_tick_params(pad=-1, labelsize=tick_size)
        bit_logo.ax.set_xticks(tick_indices)
        bit_logo.ax.set_xticklabels([f'{x}' for x in tick_labels], fontsize=tick_size)
        if plot_negative_axis:
            bit_logo.ax.set_yticks(np.linspace(-2,2,5))
            bit_logo.ax.set_yticklabels([f'{abs(int(x))}' for x in np.linspace(-2,2,5)], fontsize=tick_size)
            for i in range(len(bit_logo.glyph_df['T'])):
                bit_logo.glyph_df['T'][i].c = 'A'
                bit_logo.glyph_df['C'][i].c = 'G'
            bit_logo.style_glyphs_below(flip=True, mirror=True)
        else:
            ymax = float(seq_df_base_information.max().max()) * 1.1
            bit_logo.ax.set_ylim(0, ymax)
            bit_logo.ax.set_yticks(np.linspace(0,np.ceil(ymax),5))
            bit_logo.ax.set_yticklabels([f'{int(x)}' for x in np.linspace(0,np.ceil(ymax),5)], fontsize=tick_size)

        # --- add highlight box ---
        if {'start','end'}.issubset(TF_start_pos.columns):
            box_left, box_right = _compute_relative_box(TF_start_pos, seq_width)
            rect = patches.Rectangle(
                (box_left, bit_logo.ax.get_ylim()[0]),
                (box_right - box_left),
                bit_logo.ax.get_ylim()[1]-bit_logo.ax.get_ylim()[0],
                facecolor=box_color, alpha=box_alpha
            )
            bit_logo.ax.add_patch(rect)

    return seq_df, seq_df_base_prob, prob_logo, bit_logo