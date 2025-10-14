############################################
# Subprograms for CTR analysis
# Author: Ichiro Akai
############################################

############################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import inspect
############################################
import MCMCbase20250816 as MCMCbase
############################################





########################################
########################################
# Color functions
########################################
########################################
########################################
# Rotating colors
#---------------------------------------
# Color code for resistors
# 0: black, 1: brown, 2: red, 3: orange, 4: yellow, 5: green, 6: blue, 7: violet, 8: gray, 9: white
########################################
IA_colors_resistors = ['black', 'brown', 'red', 'orange', 'yellow', 'green', 'blue', 'violet', 'gray',]
########################################
IA_colors_tab = IA_colors_resistors
########################################
def IA_colors_settab( new_colors_tab_ ):
    global IA_colors_tab
    IA_colors_tab = new_colors_tab_
    return IA_colors_tab
########################################
IA_colors_tab_idx = 0
########################################
def IA_colors_tab_setidx( idx_ ):
    global IA_colors_tab_idx
    IA_colors_tab_idx = idx_
    if IA_colors_tab_idx >= len(IA_colors_tab):
        IA_colors_tab_idx = 0
    return IA_colors_tab_idx
########################################
def IA_colors_tab_getcolor():
    global IA_colors_tab_idx
    #
    if IA_colors_tab_idx >= len(IA_colors_tab):
        IA_colors_tab_idx = 0
    IA_color = IA_colors_tab[IA_colors_tab_idx]
    #
    IA_colors_tab_idx += 1
    if IA_colors_tab_idx >= len(IA_colors_tab):
        IA_colors_tab_idx = 0
    return IA_color
########################################

########################################
########################################
# Graphing functions
########################################
########################################

########################################
Counter_GrfSpectraV1 = 0
########################################
GrfStyle_scatter_default ={
    's':        10,         # Marker size in points**2 (typographic points are 1/72 in.).
    'c':        'black',    # Color of the markers.
    'marker':   'o',        # Marker style. 
    'alpha':    1.0,        # The alpha blending value, between 0 (transparent) and 1 (opaque).
}
########################################
def Grf_scatter_( ax_, xdata_, ydata_, ylabel_, yplotstyle_ ):
    #
    if yplotstyle_ is None:
        yplotstyle_ = GrfStyle_scatter_default
    #
    if 's' not in yplotstyle_:
        yplotstyle_['s'] = GrfStyle_scatter_default['s']
    if 'c' not in yplotstyle_:
        yplotstyle_['c'] = GrfStyle_scatter_default['c']
    if 'marker' not in yplotstyle_:
        yplotstyle_['marker'] = GrfStyle_scatter_default['marker']
    if 'alpha' not in yplotstyle_:
        yplotstyle_['alpha'] = GrfStyle_scatter_default['alpha']
    #
    ax_.scatter( xdata_, ydata_, label=ylabel_, \
                 marker=yplotstyle_['marker'], s=yplotstyle_['s'], color=yplotstyle_['c'], alpha=yplotstyle_['alpha'] )
    #
    return
########################################
GrfStyle_plot_default ={
    'linestyle':   'solid',     # {'solid', 'dashed', 'dashdot', 'dotted', ...}
    'linewidth':    2,          # The line width, in points.  
    'c':            'black',    # Color of line.
    'alpha':        1,          # The alpha blending value, between 0 (transparent) and 1 (opaque).
}
########################################
def Grf_plot_( ax_, xdata_, ydata_, ylabel_, yplotstyle_ ):
    #
    if yplotstyle_ is None:
        yplotstyle_ = GrfStyle_plot_default
    #
    if 'linestyle' not in yplotstyle_:
        yplotstyle_['linestyle'] = GrfStyle_plot_default['linestyle']
    if 'linewidth' not in yplotstyle_:
        yplotstyle_['linewidth'] = GrfStyle_plot_default['linewidth']
    if 'c' not in yplotstyle_:
        yplotstyle_['c'] = GrfStyle_plot_default['c'] 
    if 'alpha' not in yplotstyle_:
        yplotstyle_['alpha'] = GrfStyle_plot_default['alpha']
    #
    ax_.plot( xdata_, ydata_, label=ylabel_, \
              linestyle=yplotstyle_['linestyle'], linewidth=yplotstyle_['linewidth'], color=yplotstyle_['c'], alpha=yplotstyle_['alpha'] )
    #
    return
########################################
GrfStyle_plotscatter_default ={
    'linestyle':        'solid',    # {'solid', 'dashed', 'dashdot', 'dotted', ...}
    'linewidth':        2,          # The line width, in points.  
    'c':                'black',    # Color of line.
    'marker':           'o',        # Marker style.
    'markerfacecolor':  'black',    # The face color of the marker.
    'markersize':       5,         # Marker size in points**2 (typographic points are 1/72 in.).
    'alpha':            1,          # The alpha blending value, between 0 (transparent) and 1 (opaque).
}
########################################
def Grf_plotscatter_( ax_, xdata_, ydata_, ylabel_, yplotstyle_ ):
    #
    if yplotstyle_ is None:
        yplotstyle_ = GrfStyle_plotscatter_default
    #
    if 'linestyle' not in yplotstyle_:
        yplotstyle_['linestyle'] = GrfStyle_plotscatter_default['linestyle']
    if 'linewidth' not in yplotstyle_:
        yplotstyle_['linewidth'] = GrfStyle_plotscatter_default['linewidth']
    if 'c' not in yplotstyle_:
        yplotstyle_['c'] = GrfStyle_plotscatter_default['c'] 
    if 'alpha' not in yplotstyle_:
        yplotstyle_['alpha'] = GrfStyle_plotscatter_default['alpha']
    if 'marker' not in yplotstyle_:
        yplotstyle_['marker'] = GrfStyle_plotscatter_default['marker']
    if 'markerfacecolor' not in yplotstyle_:
        yplotstyle_['markerfacecolor'] = yplotstyle_['c'] # GrfStyle_plotscatter_default['markerfacecolor']
    if 'markersize' not in yplotstyle_:
        yplotstyle_['markersize'] = GrfStyle_plotscatter_default['markersize']
    #
    ax_.plot( xdata_, ydata_, label=ylabel_, \
              linestyle=yplotstyle_['linestyle'], linewidth=yplotstyle_['linewidth'], color=yplotstyle_['c'], \
              marker=yplotstyle_['marker'], markerfacecolor=yplotstyle_['markerfacecolor'], markersize=yplotstyle_['markersize'],\
              alpha=yplotstyle_['alpha'] )
    #
    return
########################################
GrfStyle_fill_default ={
    'y2':               0,          # The y coordinates of the nodes defining the second curve.
    'linestyle':        'solid',    # {'solid', 'dashed', 'dashdot', 'dotted', ...}
    'linewidth':        2,          # The line width, in points.  
    'c':                'black',    # Color.
    'alpha':            0.5,        # The alpha blending value, between 0 (transparent) and 1 (opaque).
}
########################################
def Grf_fill_( ax_, xdata_, ydata_, ylabel_, yplotstyle_ ):
    #
    if yplotstyle_ is None:
        yplotstyle_ = GrfStyle_fill_default
    #
    if 'y2' not in yplotstyle_:
        yplotstyle_['y2'] = GrfStyle_fill_default['y2']
    if 'linestyle' not in yplotstyle_:
        yplotstyle_['linestyle'] = GrfStyle_fill_default['linestyle']
    if 'linewidth' not in yplotstyle_:
        yplotstyle_['linewidth'] = GrfStyle_fill_default['linewidth']
    if 'c' not in yplotstyle_:
        yplotstyle_['c'] = GrfStyle_fill_default['c'] 
    if 'alpha' not in yplotstyle_:
        yplotstyle_['alpha'] = GrfStyle_fill_default['alpha']
    #
    ax_.fill_between( xdata_, ydata_, label=ylabel_, \
                      linestyle=yplotstyle_['linestyle'], linewidth=yplotstyle_['linewidth'], color=yplotstyle_['c'], \
                      alpha=yplotstyle_['alpha'] )
    #
    return
########################################
GrfStyle_vline_default ={
    'linestyle':    'dashed',   # {'solid', 'dashed', 'dashdot', 'dotted', ...}
    'linewidth':    1,          # The line width, in points.  
    'c':            'red',      # Color of line.
    'alpha':        1,          # The alpha blending value, between 0 (transparent) and 1 (opaque).
    'ybottom':      0,          # The bottom y-coordinate ratio of the line.
    'ytop':         1,          # The top y-coordinate ratio of the line.
}
########################################
def Grf_vline_( ax_, xpos_, xposlabel_, xposstyle_ ):
    #
    if xposstyle_ is None:
        xposstyle_ = GrfStyle_vline_default
    #
    if 'linestyle' not in xposstyle_:
        xposstyle_['linestyle'] = GrfStyle_vline_default['linestyle']
    if 'linewidth' not in xposstyle_:
        xposstyle_['linewidth'] = GrfStyle_vline_default['linewidth']
    if 'c' not in xposstyle_:
        xposstyle_['c'] = GrfStyle_vline_default['c'] 
    if 'alpha' not in xposstyle_:
        xposstyle_['alpha'] = GrfStyle_vline_default['alpha']
    if 'ybottom' not in xposstyle_:
        xposstyle_['ybottom'] = GrfStyle_vline_default['ybottom']
    if 'ytop' not in xposstyle_:
        xposstyle_['ytop'] = GrfStyle_vline_default['ytop']
    #
    (y_bot_, y_top_) = ax_.get_ylim()
    y_amplitude_ = y_top_ - y_bot_
    yy_bot_ = y_bot_ + y_amplitude_ * xposstyle_['ybottom']
    yy_top_ = y_bot_ + y_amplitude_ * xposstyle_['ytop']
    #
    ax_.vlines( xpos_, yy_bot_, yy_top_, label=xposlabel_, \
              linestyle=xposstyle_['linestyle'], linewidth=xposstyle_['linewidth'], color=xposstyle_['c'], alpha=xposstyle_['alpha'] )
    #
    return
########################################
GrfStyle_hline_default ={
    'linestyle':    'dashed',   # {'solid', 'dashed', 'dashdot', 'dotted', ...}
    'linewidth':    1,          # The line width, in points.  
    'c':            'red',      # Color of line.
    'alpha':        1,          # The alpha blending value, between 0 (transparent) and 1 (opaque).
    'xleft':        0,          # The left x-coordinate ratio of the line.
    'xright':      1,           # The right x-coordinate ratio of the line.
}
########################################
def Grf_hline_( ax_, hpos_, hposlabel_, hposstyle_ ):
    #
    if hposstyle_ is None:
        hposstyle_ = GrfStyle_hline_default
    #
    if 'linestyle' not in hposstyle_:
        hposstyle_['linestyle'] = GrfStyle_hline_default['linestyle']
    if 'linewidth' not in hposstyle_:
        hposstyle_['linewidth'] = GrfStyle_hline_default['linewidth']
    if 'c' not in hposstyle_:
        hposstyle_['c'] = GrfStyle_hline_default['c'] 
    if 'alpha' not in hposstyle_:
        hposstyle_['alpha'] = GrfStyle_hline_default['alpha']
    if 'xleft' not in hposstyle_:
        hposstyle_['xleft'] = GrfStyle_hline_default['xleft']
    if 'xright' not in hposstyle_:
        hposstyle_['xright'] = GrfStyle_hline_default['xright']
    #
    (x_left_, x_right_) = ax_.get_xlim()
    x_amplitude_ = x_right_ - x_left_
    xx_left_  = x_left_  + x_amplitude_ * hposstyle_['xleft']
    xx_right_ = x_right_ + x_amplitude_ * hposstyle_['xright']
    #
    ax_.hlines( hpos_, xx_left_, xx_right_, label=hposlabel_, \
              linestyle=hposstyle_['linestyle'], linewidth=hposstyle_['linewidth'], color=hposstyle_['c'], alpha=hposstyle_['alpha'] )
    #
    return
########################################
GrfStyle_zeroline_default ={
    'linestyle':    'dashed',   # {'solid', 'dashed', 'dashdot', 'dotted', ...}
    'linewidth':    1,          # The line width, in points.  
    'c':            'black',    # Color of line.
    'alpha':        1,          # The alpha blending value, between 0 (transparent) and 1 (opaque).
}
########################################
def Grf_yzeroline_( ax_, zerolinestyle_ ):
    #
    if zerolinestyle_ is None:
        zerolinestyle_ = GrfStyle_zeroline_default
    #
    if 'linestyle' not in zerolinestyle_:
        zerolinestyle_['linestyle'] = GrfStyle_zeroline_default['linestyle']
    if 'linewidth' not in zerolinestyle_:
        zerolinestyle_['linewidth'] = GrfStyle_zeroline_default['linewidth']
    if 'c' not in zerolinestyle_:
        zerolinestyle_['c'] = GrfStyle_zeroline_default['c'] 
    if 'alpha' not in zerolinestyle_:
        zerolinestyle_['alpha'] = GrfStyle_zeroline_default['alpha']
    #
    ax_.axhline( y=0, color=zerolinestyle_['c'], linestyle=zerolinestyle_['linestyle'], \
                 linewidth=zerolinestyle_['linewidth'], alpha=zerolinestyle_['alpha']  )
    #
    return
########################################

########################################
# Constants: GrfPlotType_scatter, GrfPlotType_plot, GrfPlotType_plotscatter, GrfPlotType_fill, GrfPlotType_assert
########################################
GrfPlotType_scatter     = 1
GrfPlotType_plot        = 2
GrfPlotType_plotscatter = 3
GrfPlotType_fill        = 4 
GrfPlotType_vline       = 5
GrfPlotType_hline       = 6
GrfPlotType_assert      = 7
########################################





############################################
pi2 = 2 * np.pi
############################################
def q_vec(theta_deg_, xray_lambda_):
    """
    Calculate the q vector from the scattering angle and X-ray wavelength.
    
    Parameters:
    theta_deg_ : float
        Scattering angle in degrees.
    xray_lambda_ : float
        X-ray wavelength in Angstroms.
    
    Returns:
    float
        The q vector in inverse Angstroms.
    """
    theta_rad_ = np.radians(theta_deg_)
    q_ = pi2 * np.sin(theta_rad_) / xray_lambda_
    return q_
############################################

############################################
def Load_Data( Data_path_, skiprows=18, encoding='shift_jis' ):
    """
    Load data from the specified file path.
    """
    if not os.path.exists( Data_path_ ):
        print( 'Data file does not exist: %s' % Data_path_ )
        sys.exit( 1 )
    #
    try:
        df_ = pd.read_csv( Data_path_, skiprows=skiprows, encoding=encoding, skipinitialspace=True )
        #
        df_.rename(columns={'Axis (deg)':'2theta'}, inplace=True)
        df_['theta'] = df_['2theta'] / 2.0
        df_['I_norm'] = df_['I (cts)'] / df_['I0 (cts)']
        I_norm_max_ = np.amax( df_['I_norm'] )
        df_['I_norm_1'] = df_['I_norm'] / I_norm_max_
        #
        return df_
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
############################################


############################################
# CTRsets = [CTRset1, CTRset2, ...]
############################################
# CTRset = [ data_x, data_y, data_label, plottype, plotstyle ]
############################################
# plottype:
#   1, CTR.GrfPlotType_scatter:
#            CTR.Grf_scatter_( ax1_, data_x_, data_y_, data_label_, plotstyle_ )
#   2, CTR.GrfPlotType_plot:
#            CTR.Grf_plot_( ax1_, data_x_, data_y_, data_label_, plotstyle_ )
#   3, CTR.GrfPlotType_plotscatter:
#            CTR.Grf_plotscatter_( ax1_, data_x_, data_y_, data_label_, plotstyle_ )
#   4, CTR.GrfPlotType_fill:
#            CTR.Grf_fill_( ax1_, data_x_, data_y_, data_label_, plotstyle_ )
#   5, CTR.GrfPlotType_vline:
#            CTR.Grf_vline_( ax1_, xpos_, xposlabel_, xposstyle_ )
############################################
def GrfCTRv1( CTRsets_, title=None, y_digits=None,
              xlabel=r'$\theta$ (deg)', ylabel='Intensity (logarithmic scale)', 
              fontsize=16, figsize=(10,6), \
              adj_left=0.20, adj_right=0.98, adj_top=0.93,  adj_bottom=0.12, 
              legend_fontsize=12, legend_loc='upper left' ):
    """
    Plot the CTR data.
    """
    #
    #
    plt.rcParams['font.size'] = fontsize
    #
    fig_ = plt.figure( figsize = figsize )
    fig_.subplots_adjust( left = adj_left, right = adj_right, top = adj_top,  bottom = adj_bottom )
    #
    ax1_ = fig_.add_subplot(1,1,1)
    ax1_.set_xlabel( xlabel )
    ax1_.set_ylabel( ylabel )
    ax1_.set_yscale( 'log' )
    #
    for CTRset_ in CTRsets_:
        #
        if( len(CTRset_) == 5 ):
            data_x_, data_y_, data_label_, plottype_, plotstyle_ = CTRset_
        elif ( len(CTRset_) == 4 ):
            data_x_, data_y_, data_label_, plottype_             = CTRset_
            plotstyle_ = None
        else:
            sys.exit( 'Error: CTRset length is not 4 or 5.' )
        #
        assert plottype_ < GrfPlotType_assert, 'Error: plottype_ >= %d in GrfCTRv1()' % GrfPlotType_assert
        #
        if( plottype_ == GrfPlotType_scatter ):
            Grf_scatter_( ax1_, data_x_, data_y_, data_label_, plotstyle_ )
        elif( plottype_ == GrfPlotType_plot ):
            Grf_plot_( ax1_, data_x_, data_y_, data_label_, plotstyle_ )
        elif( plottype_ == GrfPlotType_plotscatter ):
            Grf_plotscatter_( ax1_, data_x_, data_y_, data_label_, plotstyle_ )
        elif( plottype_ == GrfPlotType_fill ):
            Grf_fill_( ax1_, data_x_, data_y_, data_label_, plotstyle_ )
        elif( plottype_ == GrfPlotType_vline ):
            (y_bot_, y_top_) = ax1_.get_ylim()
            xpos_ = data_x_
            xposlabel_ = data_label_
            xposstyle_ = plotstyle_
            Grf_vline_( ax1_, xpos_, xposlabel_, xposstyle_ )
            ax1_.set_ylim(y_bot_, y_top_)
        elif( plottype_ == GrfPlotType_hline ):
            (x_left_, x_right_) = ax1_.get_xlim()
            ypos_ = data_y_
            yposlabel_ = data_label_
            yposstyle_ = plotstyle_
            Grf_hline_( ax1_, ypos_, yposlabel_, yposstyle_ )
            ax1_.set_xlim(x_left_, x_right_)
    #
    if y_digits is not None:
        (y_bot_, y_top_) = ax1_.get_ylim()
        y_bot_ = y_top_ / (10 ** y_digits)
        ax1_.set_ylim(y_bot_, y_top_)
    #
    if title is not None:
        plt.title( title )
    #
    plt.legend( fontsize=legend_fontsize, loc=legend_loc )
    #
    plt.show()
    #
    return fig_


############################################
############################################
# Dictionary for CTR data set 
############################################
############################################

############################################
# Loading the data file and Make the Dictionary for CTR data set
# ----------- 
# 1. Loading data to DataFrame.
# 2. Normalization y_data by the maximum of y_data.
# 3. Extracting data to be analysed based on WeakerDigits.
# 4. Draw the data.
############################################
def CTR_dict_Load_Data( Data_path_, Data_ID_, xray_lambda_, WeakerDigits=1.0E-2, x_key='theta', y_key='I_norm', y_digits=8 ):
    """
    Set dataset for CTR_dict
    ---
    1. Loading data to DataFrame.
    2. Normalization y_data by the maximum of y_data.
    3. Extracting data to be analysed based on WeakerDigits.
    4. Draw the data.
    ---
    Parameters:
    Data_path_ : str
        Path of the loading data file.
    Data_ID_ : str
        Str of the Data identifier
    xray_lambda_ : float
        X-ray wavelength in Angstroms.
    WeakerDigits_ : float
        ピーク強度から、何桁以下のデータを分析対象とするかを設定する
        Set the number of digits to be analysed below the peak intensity.
    x_key:
        Key in the loading data for x-axis data.
    y_key:
        Key in the loading data for y-axis data.
    y_digits:
        y-axis range (order) in the drawing graph.
    
    Returns: Dict, fig
    """
    ############################################
    print( 'CTR_dict_Load_Data: Data_path    =', Data_path_   )
    print( 'CTR_dict_Load_Data: Data_ID      =', Data_ID_     )
    print( 'CTR_dict_Load_Data: WeakerDigits =', WeakerDigits )
    print( 'CTR_dict_Load_Data: x_key        =', x_key        )
    print( 'CTR_dict_Load_Data: y_key        =', y_key        )
    print( 'CTR_dict_Load_Data: y_digits     =', y_digits     )
    #    
    ############################################
    df_CTR_    = Load_Data( Data_path_ )
    #
    print( 'df_CTR.shape   =', df_CTR_.shape   )
    print( 'df_CTR.columns =', df_CTR_.columns )
    print( df_CTR_.head( 5 ) )
    print( df_CTR_.tail( 5 ) )
    #
    # q_vec
    ############################################
    x_data_org_ = np.array( df_CTR_[ x_key ] )
    q_data_org_ = q_vec( x_data_org_, xray_lambda_ )
    y_data_org_ = np.array( df_CTR_[ y_key ] )
    #
    ############################################
    y_data_org_norm_ = y_data_org_ / np.amax( y_data_org_ )
    #
    ############################################
    y_data_ = y_data_org_norm_[ y_data_org_norm_ <= WeakerDigits ]
    q_data_ = q_data_org_[ y_data_org_norm_ <= WeakerDigits ] 
    x_data_ = x_data_org_[ y_data_org_norm_ <= WeakerDigits ] 
    #
    ############################################
    CTR_dict_ = { 'CTR_dataset' :
                    {   'CTR_dataset'       : df_CTR_,
                        'Data_ID'           : Data_ID_, 
                        'x_key'             : x_key,
                        'y_key'             : y_key,
                        'WeakerDigits'      : WeakerDigits,
                        'Data_x'            : x_data_,
                        'Data_q'            : q_data_,
                        'Data_y'            : y_data_,
                        'Data_x_org'        : x_data_org_,
                        'Data_y_org'        : y_data_org_,
                        'Data_y_org_norm'   : y_data_org_norm_
                    }
                }
    #
    ############################################
    CTRsets_ = [ [ x_data_org_, y_data_org_norm_, Data_ID_+':Orig.(Norm)',  GrfPlotType_plot,  {'c':'black'}], 
                 [ x_data_,     y_data_,          Data_ID_+':Target',       GrfPlotType_plot,  {'c':'red'  }],
                 [ 0,           WeakerDigits,     'WeakerDigits',           GrfPlotType_hline, {'c':'green','linestyle':'dashed'}] ]
    #
    fig_ = GrfCTRv1( CTRsets_, title=Data_ID_, y_digits=y_digits )
    #
    return CTR_dict_, fig_
############################################

############################################
# Geting the Data_q and Data_Y from the Dictionary for CTR data set
############################################
def CTR_dict_Get_Data_ID( CTR_dict_ ):
    """
    Geting the Data_ID from the Dictionary for CTR data set
    """
    return CTR_dict_['CTR_dataset']['Data_ID']
############################################
def CTR_dict_Get_x_data( CTR_dict_ ):
    """
    Geting the Data_x from the Dictionary for CTR data set
    """
    return CTR_dict_['CTR_dataset']['Data_x']
############################################
def CTR_dict_Get_y_data( CTR_dict_ ):
    """
    Geting the Data_y from the Dictionary for CTR data set
    """
    return CTR_dict_['CTR_dataset']['Data_y']
############################################
def CTR_dict_Get_q_data( CTR_dict_ ):
    """
    Geting the Data_q from the Dictionary for CTR data set
    """
    return CTR_dict_['CTR_dataset']['Data_q']
############################################

############################################
def PyMC_Make_CTR_Regression( Data_dict_, Reg_function_, chain=None, output_path=None ):
    """
    Make the  regression data of MAP, MLE, MEAN and MEDIAN estimates.

	    Parameters:
	        Data_dict_ : dictionary
		        The dictionary for the dataset.
	    Reg_function_ : 
            the function for regression
      chain : None, int
          When None, chain-mereged data is used. 
          When integer (started from 1), specified chain data is used.
        
	    Returns:
		    Dictionary, Dataframe of regression data.
    """
    assert isinstance( Data_dict_, dict ), 'PyMC_Make_Regression: Data_dict_ must be a dictionary.'
    #
    assert 'CTR_dataset' in Data_dict_, 'PyMC_Make_Regression: \'CTR_dataset\' is not included in Data_dict_.'
    #
    assert 'PyMC_Setup' in Data_dict_, 'PyMC_Make_Regression: \'PyMC_Setup\' is not included in Data_dict_.'
    #
    assert 'Parameters' in Data_dict_, 'PyMC_Make_Regression: \'Parameters\' is not included in Data_dict_.'
    #
    chains_ = Data_dict_['PyMC_Setup']['Chains']
    method_ = Data_dict_['PyMC_Setup']['Method']
    #
    ################################
    if chain is None:
        chain_key_ = 'Stat_Merged'
    else:
        chain_key_ = 'Stat_Chain=%d' % chain
    #
    print( 'PyMC_Make_Regression: %s' % chain_key_ )
    #
    ################################
    # Target data
    Data_ID_ = Data_dict_['CTR_dataset']['Data_ID']
    Data_x_  = Data_dict_['CTR_dataset']['Data_x']
    Data_q_  = Data_dict_['CTR_dataset']['Data_q']
    Data_y_  = Data_dict_['CTR_dataset']['Data_y']
    Data_len_ = len( Data_x_ )
    Data_x_min_ = np.amin( Data_x_ )
    Data_x_max_ = np.amax( Data_x_ )
    Data_q_min_ = np.amin( Data_q_ )
    Data_q_max_ = np.amax( Data_q_ )
    Data_y_min_ = np.amin( Data_y_ )
    Data_y_max_ = np.amax( Data_y_ )
    #
    if not ('PyMC_Regression' in Data_dict_):
        Data_dict_['PyMC_Regression'] = {}
    #
    Data_dict_['PyMC_Regression']['Data_ID'] = Data_ID_
    Data_dict_['PyMC_Regression']['Data_x']  = Data_x_
    Data_dict_['PyMC_Regression']['Data_q']  = Data_q_
    Data_dict_['PyMC_Regression']['Data_y']  = Data_y_
    #
    print( 'PyMC_Make_Regression: Target Data; %s (%d), x = %g - %g, y = %g - %g' % (Data_ID_, Data_len_, Data_x_min_, Data_x_max_, Data_y_min_, Data_y_max_) )
    #
    ################################
    # parse the regression function
    Data_dict_['PyMC_Regression']['Reg_function']  = Reg_function_
    Reg_function_name_ = Reg_function_.__code__.co_name
    Data_dict_['PyMC_Regression']['Reg_function_name']  = Reg_function_name_
    #
    # import inspect
    # https://chatgpt.com/c/687ee45b-3bf8-8012-95ce-9fe1e284a88e
    sig_ = inspect.signature( Reg_function_ )
    params_ = sig_.parameters
    #
    # The number of arguments for Reg_function_
    Reg_function_argc_ = len( params_ )
    #
    Reg_function_idx_ = 1
    Reg_function_str_ = '%s(' % Reg_function_name_
    Reg_argm_names_ = []
    #
    for name_, param_ in params_.items():
        #
        if Reg_function_idx_ < Reg_function_argc_:
            Reg_function_str_ += '%s, ' % name_
            # print( '%s' % name_, end=', ' )
        else:
            Reg_function_str_ += '%s)' % name_
            # print( '%s' % name_, end=')' )
        #
        Reg_argm_names_.append(name_)
        #
        Reg_function_idx_ += 1
    #
    print( 'PyMC_Make_Regression: Function; %s' % Reg_function_str_ )
    #
    #
    # print( Reg_argm_names_ )
    ################################
    # MAP estimates
    param_names_, param_values_ = MCMCbase.PyMC_Get_Estimates(Data_dict_, 'map', chain=chain )
    #
    kwargs = {'q_data': Data_q_}
    #
    param_idx_ = 0
    for param_name_ in param_names_:
        kwargs[ param_name_ ] = param_values_[ param_idx_ ]
        param_idx_ += 1
    #
    print( kwargs )
    #
    # Calculate the regression
    Reg_MAP_ = Reg_function_( **kwargs )
    #
    Data_dict_['PyMC_Regression']['Reg_MAP'] = Reg_MAP_
    #
    ################################
    # MLE estimates
    param_names_, param_values_ = MCMCbase.PyMC_Get_Estimates(Data_dict_, 'mle', chain=chain )
    #
    kwargs = {'q_data': Data_q_}
    #
    param_idx_ = 0
    for param_name_ in param_names_:
        kwargs[ param_name_ ] = param_values_[ param_idx_ ]
        param_idx_ += 1
    #
    # print( kwargs )
    #
    # Calculate the regression
    Reg_MLE_ = Reg_function_( **kwargs )
    #
    Data_dict_['PyMC_Regression']['Reg_MLE'] = Reg_MLE_
    #
    ################################
    # MEAN estimates
    param_names_, param_values_ = MCMCbase.PyMC_Get_Estimates(Data_dict_, 'mean', chain=chain )
    #
    kwargs = {'q_data': Data_q_}
    #
    param_idx_ = 0
    for param_name_ in param_names_:
        kwargs[ param_name_ ] = param_values_[ param_idx_ ]
        param_idx_ += 1
    #
    # print( kwargs )
    #
    # Calculate the regression
    Reg_MEAN_ = Reg_function_( **kwargs )
    #
    Data_dict_['PyMC_Regression']['Reg_MEAN'] = Reg_MEAN_
    #
    ################################
    # MEDIAN estimates
    param_names_, param_values_ = MCMCbase.PyMC_Get_Estimates(Data_dict_, 'median', chain=chain )
    #
    kwargs = {'q_data': Data_q_}
    #
    param_idx_ = 0
    for param_name_ in param_names_:
        kwargs[ param_name_ ] = param_values_[ param_idx_ ]
        param_idx_ += 1
    #
    # print( kwargs )
    #
    # Calculate the regression
    Reg_MEDIAN_ = Reg_function_( **kwargs )
    #
    Data_dict_['PyMC_Regression']['Reg_MEDIAN'] = Reg_MEDIAN_
    #
    ################################
    # Dataframe
    df_dict_ = dict( Data_x=Data_x_, Data_y=Data_y_, Reg_MAP=Reg_MAP_, Reg_MLE=Reg_MLE_, Reg_MEAN=Reg_MEAN_, Reg_MEDIAN=Reg_MEDIAN_)
    df_ = pd.DataFrame( data=df_dict_ )
    #
    ######################
    if output_path is not None:
        outpath_xlsx_ = output_path + Data_ID_ + '-Regressions.xlsx'
        df_.to_excel( outpath_xlsx_, sheet_name='Regression-'+Data_ID_ )
    #
    return Data_dict_, df_, Reg_function_name_ + r'(x;$\boldsymbol{\theta}$)'
############################################
