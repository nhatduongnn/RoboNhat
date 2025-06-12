
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


def plot_motif_logo(fasta_file_dir, TF_start_pos_original, seq_width, shift, ax1, ax2, fasta_type='number', bases='AG'):
    """
    Plot sequence motif logos (probability and information content) centered around specified genomic positions.

    This function extracts sequences from a FASTA file around given genomic coordinates,
    calculates base probabilities and information content, and generates sequence logos 
    showing the distributions of selected bases (A/G or A only or G only) on positive and negative strands.

    Parameters
    ----------
    fasta_file_dir : str
        Path to the FASTA file containing genome sequences.

    TF_start_pos_original : pandas.DataFrame
        A DataFrame with at least two columns: 
        'chr' (chromosome identifier) and 'pos' (genomic position) around which to extract sequence windows.

    seq_width : int
        Width of the sequence window (in base pairs) centered at each position.

    shift : int
        Amount to shift each input position before extracting the sequence window.

    ax1 : matplotlib.axes.Axes
        Matplotlib axis object for plotting the probability logo.

    ax2 : matplotlib.axes.Axes
        Matplotlib axis object for plotting the information content (bit) logo.

    fasta_type : str, optional, default='number'
        Format of chromosome names in the FASTA file:
        - 'roman' : e.g., 'chrI', 'chrII', etc.
        - 'number' : e.g., 'chr1', 'chr2', etc.

    bases : str, optional, default='AG'
        Bases to display in the logos:
        - 'AG' : Plot A/G bases upwards and C/T bases downwards.
        - 'A'  : Plot only A bases upwards (flip T downwards).
        - 'G'  : Plot only G bases upwards (flip C downwards).

    Returns
    -------
    seq_df : pandas.DataFrame
        DataFrame containing the extracted sequence windows.

    seq_df_base_prob : pandas.DataFrame
        DataFrame containing the per-position base probabilities (before strand flipping).

    prob_logo : logomaker.Logo
        Logomaker object for the probability logo.

    bit_logo : logomaker.Logo
        Logomaker object for the information content (bit) logo.

    Notes
    -----
    - Positions that extend beyond chromosome boundaries are skipped with a warning printed.
    - Base probabilities for C/T bases are flipped (negative values) to represent the complementary strand.
    - Colors are assigned (blue for A/G, orange for C/T) with custom mapping.
    - Logomaker is used for logo plotting, and fonts are set to 'Arial Rounded MT Bold'.
    - Mirror-flip effects are applied for better visualization of complementary bases.

    Dependencies
    ------------
    - Biopython (for SeqIO, SeqFeature, FeatureLocation)
    - numpy
    - pandas
    - logomaker
    - matplotlib

    """

    
    ## Create an empty dict to store all chromosome of the fasta file
    fasta_file = {}
    for seq_record in SeqIO.parse(fasta_file_dir, "fasta"):
        if fasta_type == 'roman':
            fasta_file[seq_record.id] = seq_record.seq
        elif fasta_type == 'number':
            if seq_record.id != 'chrM':
                fasta_file[from_roman(seq_record.id.split('chr')[1])] = seq_record.seq
            elif seq_record.id == 'chrM':
                fasta_file[seq_record.id.split('chr')[1]] = seq_record.seq
        
    ## Adjust for shift, so that we can center the plot around a certain base pair
    TF_start_pos = TF_start_pos_original.copy()
    TF_start_pos['pos'] = TF_start_pos['pos'] + shift
        
    ## Create an empty matrix to store extracted sequences
    seq_mat = np.zeros([TF_start_pos.shape[0],seq_width+1],dtype='str')
    
    ## Iterate through all position values and extract sequences around that position
    for i in TF_start_pos.reset_index(drop=True).itertuples():
        start_wind =  i.pos-int(seq_width/2)-1
        end_wind = i.pos+int(seq_width/2)
        # Check if the window to be created by SeqFeature will be negative or larger than chromosome size
        if start_wind <= 0 or end_wind > len(fasta_file[i.chr])-1:
            print("chromosome is {}, position is {} and range is from {} to {}".format(i.chr,i.pos,start_wind,end_wind))
            continue
        else:
            seq = SeqFeature(FeatureLocation(i.pos-int(seq_width/2)-1, i.pos+int(seq_width/2)), type="gene", strand=1).extract(fasta_file[i.chr])
            #print(seq)
            seq_mat[i.Index,:] = np.array(seq)
        
        
    # Turn the extracted sequences into a DataFrame
    seq_df = pd.DataFrame(seq_mat)
    # Fill in missing values as a count of 0
    seq_df_base_count = seq_df.apply(lambda x: x.value_counts().reindex(['A','C','G','T'],fill_value=0))
    # Calculate the probability of each base at each bp position
    seq_df_base_prob = seq_df_base_count.apply(lambda x: x/seq_df.shape[0])
    
    # Modify the probability to plot the bases that we want
    # Multiply by -1 to flip the C and T probabilities to the opposite strrand
    seq_df_base_prob_AG_top_CT_bot = seq_df_base_prob.copy()
    if bases == 'AG':
        seq_df_base_prob_AG_top_CT_bot.loc["C"] = seq_df_base_prob_AG_top_CT_bot.loc["C"]*-1
        seq_df_base_prob_AG_top_CT_bot.loc["T"] = seq_df_base_prob_AG_top_CT_bot.loc["T"]*-1
    # Assign probability values to 0 if we don't want those bases plotted
    elif bases == 'A':
        seq_df_base_prob_AG_top_CT_bot.loc["G"] = 0
        seq_df_base_prob_AG_top_CT_bot.loc["C"] = 0
        seq_df_base_prob_AG_top_CT_bot.loc["T"] = seq_df_base_prob_AG_top_CT_bot.loc["T"]*-1
    elif bases == 'G':
        seq_df_base_prob_AG_top_CT_bot.loc["A"] = 0
        seq_df_base_prob_AG_top_CT_bot.loc["T"] = 0
        seq_df_base_prob_AG_top_CT_bot.loc["C"] = seq_df_base_prob_AG_top_CT_bot.loc["C"]*-1


    ## Plot the probability plot
    # create color scheme
    color_scheme = {
        'A' : 'blue',
        'C' : 'orange',
        'G' : 'blue',
        'T' : 'orange'
    }

    
    # create Logo object
    prob_logo = logomaker.Logo(seq_df_base_prob_AG_top_CT_bot.T,
                               ax=ax1,
                              shade_below=0,
                              fade_below=0,
                              color_scheme = color_scheme,
                              font_name='Arial Rounded MT Bold')


    # style using Logo methods
    prob_logo.style_spines(visible=False)
    prob_logo.style_spines(spines=['left', 'bottom'], visible=True)
    prob_logo.style_xticks(rotation=90, fmt='%d', anchor=0)

    # style using Axes methods
    prob_logo.ax.set_ylabel("Probability", labelpad=10)
    prob_logo.ax.yaxis.set_tick_params(pad=5)
    prob_logo.ax.set_yticks(np.linspace(-1,1,5))
    prob_logo.ax.set_yticklabels('%.1f'%x for x in abs(np.linspace(-1,1,5)))
    prob_logo.ax.xaxis.set_ticks_position('none')
    prob_logo.ax.xaxis.set_tick_params(pad=-1)
    prob_logo.ax.set_xticks(range(0,seq_width+1,5))
    prob_logo.ax.set_xticklabels('%d'%x for x in range(-1*int(seq_width/2),int(seq_width/2)+1,5))
    prob_logo.style_spines(spines=['left', 'bottom'], visible=True)
    
    for i in range(len(prob_logo.glyph_df['T'])):
        prob_logo.glyph_df['T'][i].c = 'A'
        prob_logo.glyph_df['C'][i].c = 'G'

    
    prob_logo.style_glyphs_below(flip=True,mirror=True)

    
    ## Further process the probability dataframe to get the information(bit) dataframe
    max_information = 4*-0.25*np.log2(0.25)
    max_info_per_base = max_information - (seq_df_base_prob.apply(lambda x: -x*np.log2(x))).sum(axis=0)
    
    seq_df_base_prob_w_max_info = pd.concat([seq_df_base_prob,pd.DataFrame(max_info_per_base).T],axis=0)
    
    seq_df_base_information = seq_df_base_prob_w_max_info.apply(lambda x: x*x.iloc[4]).iloc[:4,:]
    
    seq_df_base_information_AG_top_CT_bot = seq_df_base_information.copy()
    # Modify the probability to plot the bases that we want
    # Multiply by -1 to flip the C and T probabilities to the opposite strrand
    if bases == 'AG':
        seq_df_base_information_AG_top_CT_bot.loc["C"] = seq_df_base_information_AG_top_CT_bot.loc["C"]*-1
        seq_df_base_information_AG_top_CT_bot.loc["T"] = seq_df_base_information_AG_top_CT_bot.loc["T"]*-1
    # Assign probability values to 0 if we don't want those bases plotted
    elif bases == 'A':
        seq_df_base_information_AG_top_CT_bot.loc["G"] = 0
        seq_df_base_information_AG_top_CT_bot.loc["C"] = 0
        seq_df_base_information_AG_top_CT_bot.loc["T"] = seq_df_base_information_AG_top_CT_bot.loc["T"]*-1
    elif bases == 'G':
        seq_df_base_information_AG_top_CT_bot.loc["A"] = 0
        seq_df_base_information_AG_top_CT_bot.loc["T"] = 0
        seq_df_base_information_AG_top_CT_bot.loc["C"] = seq_df_base_information_AG_top_CT_bot.loc["C"]*-1
    
    
    ## Plot the information(bit) dataframe
    # create Logo object
    bit_logo = logomaker.Logo(seq_df_base_information_AG_top_CT_bot.T,
                              ax=ax2,
                              shade_below=0,
                              fade_below=0,
                              color_scheme = color_scheme,
                              font_name='Arial Rounded MT Bold')


    # style using Logo methods
    bit_logo.style_spines(visible=False)
    bit_logo.style_spines(spines=['left', 'bottom'], visible=True)
    bit_logo.style_xticks(rotation=90, fmt='%d', anchor=0)

    # style using Axes methods
    bit_logo.ax.set_ylabel("Bit", labelpad=10)
    bit_logo.ax.yaxis.set_tick_params(pad=5)
    bit_logo.ax.set_yticks(np.linspace(-2,2,5))
    bit_logo.ax.set_yticklabels('%d'%x for x in abs(np.linspace(-2,2,5)))
    bit_logo.ax.xaxis.set_ticks_position('none')
    bit_logo.ax.xaxis.set_tick_params(pad=-1)
    bit_logo.ax.set_xticks(range(0,seq_width+1,5))
    bit_logo.ax.set_xticklabels('%d'%x for x in range(-1*int(seq_width/2),int(seq_width/2)+1,5))
    
    for i in range(len(bit_logo.glyph_df['T'])):
        bit_logo.glyph_df['T'][i].c = 'A'
        bit_logo.glyph_df['C'][i].c = 'G'

    
    bit_logo.style_glyphs_below(flip=True,mirror=True)


    #return fig,ax
    return seq_df,seq_df_base_prob,prob_logo,bit_logo

