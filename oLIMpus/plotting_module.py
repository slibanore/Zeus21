import oLIMpus

from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.cm import get_cmap
from matplotlib import colors as cc 
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from oLIMpus.maps_LIM import build_lightcone, CoevalBox_LIM_zeuslike, CoevalBoxes_percell


plt.rcParams.update({"text.usetex": True, "font.family": "Times new roman"}) # Use latex fonts
plt.rcParams['lines.linewidth'] = 2
colors = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors) # Set the color palette as default

min_value = -100
max_value = 50
mid_point = abs(min_value)/(abs(min_value)+abs(max_value))
colors_list = [(0, 'cyan'),
                (mid_point/1.5, 'blue'),
                (mid_point, "black"),
                ((1.+mid_point)/2.2, 'red'),
                (1, 'yellow')]
eor_colour = cc.LinearSegmentedColormap.from_list("eor_colour",colors_list)


summer_cmap = cm.summer
colors_list = [(0, "black"),
    (0.1, summer_cmap(0)),    
    (0.2, summer_cmap(50)),
    (0.5, summer_cmap(150)),
    (1, summer_cmap(255))]     

LIM_colour_1 = cc.LinearSegmentedColormap.from_list("LIM_colour_1",colors_list)


winter_cmap = cm.winter
colors_list = [(0, "black"),
    (0.1, winter_cmap(0)),    
    (0.2, winter_cmap(50)),
    (0.5, winter_cmap(150)),
    (1, winter_cmap(255))]     
LIM_colour_2 = cc.LinearSegmentedColormap.from_list("LIM_colour_2",colors_list)


def plot_T21(z, 
             Lbox, 
             Ncell,
            seed, 
            RSDMODE, 
            AstroParams, 
            ClassyCosmo,
            HMFintclass,
            CosmoParams, 
            ZMIN, 
            _islice= 0,
            input_text_label = None,
            ax = None,
            fig = None,
            **kwargs
    ):
    
    correlations = oLIMpus.Correlations(CosmoParams, ClassyCosmo)   
    coefficients = oLIMpus.get_T21_coefficients(CosmoParams, ClassyCosmo, AstroParams, HMFintclass, zmin=ZMIN)
   
    PS21 = oLIMpus.Power_Spectra(CosmoParams, AstroParams, ClassyCosmo, correlations, coefficients, RSD_MODE = RSDMODE)

    BMF = oLIMpus.BMF(coefficients, HMFintclass,CosmoParams,AstroParams)

    Maps_T21 = oLIMpus.maps.T21_bubbles(coefficients,PS21,z,Lbox,Ncell,seed,correlations,CosmoParams,AstroParams,ClassyCosmo,BMF)

    coeval_slice_T21_lin = Maps_T21.T21map[_islice]

    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))

    plt.sca(ax)

    im = ax.imshow(coeval_slice_T21_lin,extent=(0,Lbox,0,Lbox),cmap=eor_colour,vmax =max_value, vmin = min_value,**kwargs)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.0e")

    ax.set_xticks([])
    ax.set_yticks([])

    xticks = np.linspace(min_value,max_value,4)
    array_ticks = np.vectorize(lambda x: f"{x:.0f}")(xticks)

    cbar.set_ticks(xticks, labels = array_ticks)

    text_label_helper = r'$\rm T_{21}$'
    if input_text_label is None:
        text_label = text_label_helper
    else:
        text_label = text_label_helper + ',\,' + input_text_label

    ax.text(
        0.05, 0.05, text_label, 
        color='black',
        fontsize=10,
        ha='left', va='bottom',
        transform=ax.transAxes,  # Coordinates relative to the axis (0 to 1)
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3')  # White rounded background
    )

    return fig, ax


def plot_Inu(z, 
             Lbox, 
             Ncell, 
             seed, 
             RSDMODE, 
             analytical, 
             LineParams, 
             AstroParams, 
             ClassyCosmo, 
             HMFintclass, 
             CosmoParams, 
             ZMIN, 
             _islice = 0,
            input_text_label = None,
            ax = None,
            fig = None,
            **kwargs
        ):

    correlations = oLIMpus.Correlations_LIM(LineParams, CosmoParams, ClassyCosmo)
    coefficients = oLIMpus.get_LIM_coefficients(CosmoParams,  AstroParams, HMFintclass, LineParams, zmin=ZMIN)

    PSLIM = oLIMpus.Power_Spectra_LIM(CosmoParams, AstroParams, LineParams, correlations, coefficients, RSD_MODE = RSDMODE)

    correlations_21 = oLIMpus.Correlations(CosmoParams, ClassyCosmo)   
    coefficients_21 = oLIMpus.get_T21_coefficients(CosmoParams, ClassyCosmo, AstroParams, HMFintclass, zmin=ZMIN)
   
    PS21 = oLIMpus.Power_Spectra(CosmoParams, AstroParams, ClassyCosmo, correlations_21, coefficients_21, RSD_MODE = RSDMODE)

    if analytical:
        Maps_line = CoevalBox_LIM_zeuslike(coefficients,correlations,PSLIM,z,Lbox,Ncell,1,seed)
        coeval_slice_line = Maps_line.LIMmap[_islice]

    else:
        Maps_line = CoevalBoxes_percell('full',coefficients,correlations,PSLIM,correlations_21,z,LineParams,AstroParams,HMFintclass,CosmoParams,Lbox,Ncell,seed)
        coeval_slice_line = Maps_line.Inu_box[_islice]

    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))

    im = ax.imshow(coeval_slice_line,extent=(0,Lbox,0,Lbox),cmap=LIM_colour_1,**kwargs)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.0e")
    ax.set_xticks([])
    ax.set_yticks([])

    xticks = np.linspace(np.min(coeval_slice_line),np.max(coeval_slice_line),4)
    array_ticks = np.vectorize(lambda x: f"{x:.0f}")(xticks)

    cbar.set_ticks(xticks, labels = array_ticks)

    text_label_helper = r'$\rm I_{%s}$'%LineParams.LINE
    if input_text_label is None:
        text_label = text_label_helper
    else:
        text_label = text_label_helper + ',\,' + input_text_label


    ax.text(
        0.05, 0.05, text_label, 
        color='black',
        fontsize=10,
        ha='left', va='bottom',
        transform=ax.transAxes,  # Coordinates relative to the axis (0 to 1)
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3')  # White rounded background
    )

    return 


def plot_lightcone(which_lightcone,
             input_zvals,
             Lbox, 
             Ncell, 
             seed, 
             RSDMODE, 
             analytical, 
             LineParams, 
             AstroParams, 
             ClassyCosmo, 
             HMFintclass, 
             CosmoParams,      
             ZMIN, 
             _islice = 0,             
            input_text_label = None,
            ax = None,
            fig = None,
            cmap = None,
            **kwargs
            ):

    zvals = input_zvals[::-1]

    lightcone = build_lightcone(which_lightcone,
             input_zvals,
             Lbox, 
             Ncell, 
             seed, 
             RSDMODE, 
             analytical, 
             LineParams, 
             AstroParams, 
             ClassyCosmo, 
             HMFintclass, 
             CosmoParams,      
             ZMIN,)
    
    if which_lightcone == 'density':
        text_label_helper = r'$\delta$'
        use_cmap = 'magma'
        vmin = -0.6
        vmax = 0.6
    elif which_lightcone == 'SFRD':
        text_label_helper = r'$\rm SFRD\,[M_\odot/{\rm (yr Mpc^{-3})}}]$'
        use_cmap = 'bwr'
        vmin = 1e-5
        vmax = 1e-1
    elif which_lightcone == 'xHI':
        text_label_helper = r'$x_{\rm HI}$'
        use_cmap = 'gray'
        vmin = 0.
        vmax = 1.
    elif which_lightcone == 'T21':
        text_label_helper = r'$T_{21}\,[{\mu\rm K}]$'
        use_cmap = eor_colour
        vmin = min_value
        vmax = max_value
    elif which_lightcone == 'LIM':
        text_label_helper = r'$I_{%s}\,[{\rm Jy/sr}]$'%LineParams.LINE
        use_cmap = LIM_colour_1
        vmin = np.min(lightcone)
        vmax = np.max(lightcone)
    else:
        print('Check lightcone')
        return 

    if cmap:
        use_cmap = cmap

    if ax is None or fig is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 2))

    if input_text_label is None:
        text_label = text_label_helper
    else:
        text_label = text_label_helper + ',\,' + input_text_label

    extent = [zvals[0], zvals[-1], 0, Lbox]

    if which_lightcone == 'SFRD':
        im = ax.imshow(lightcone[:,_islice,:], aspect='auto', extent=extent, cmap=use_cmap, origin='lower', norm = LogNorm(vmin=vmin, vmax=vmax))    
    else:
        im = ax.imshow(lightcone[:,0,:], aspect='auto', extent=extent, cmap=use_cmap, origin='lower', vmin = vmin,vmax=vmax)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.0e")
    ax.set_ylabel(text_label,fontsize=15)
    ax.set_xlabel(r'$z$',fontsize=15)

    plt.tight_layout()

    return 