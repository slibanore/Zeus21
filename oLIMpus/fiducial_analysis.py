import numpy as np 
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.cm import get_cmap
from matplotlib import colors as cc 
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

plt.rcParams.update({"text.usetex": True, "font.family": "Times new roman"}) # Use latex fonts
plt.rcParams['lines.linewidth'] = 2
colors = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226','#7f1d1d']
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

################################################
################################################
################################################
################################################

seed = 1065
RSDMODE = 0
ZMIN = 5.
islice = 0

Lbox_fid = 300
Nbox_fid = 150

zvals = np.linspace(20.,6.,71)

values_epsstar = np.linspace(0.01,1,10)
values_fesc = np.linspace(0.01,1,10)
values_OmC = np.linspace(0.1,0.2,10)

CosmoParams_input_fid = dict(
        omegab= 0.0223828, 
        omegac = 0.1201075, 
        h_fid = 0.67810, 
        As = 2.100549e-09, 
        ns = 0.9660499, 
        tau_fid = 0.05430842, 
        HMF_CHOICE= "ST",
        Rsmmin = 0.5)

AstroParams_input_fid = dict(

        alphastar = 0.5,
        betastar = -0.5,
        epsstar = 0.1,
        Mc = 3e11,
        dlog10epsstardz = 0.0,

        fesc10 = 0.1,
        alphaesc = 0.0,
        L40_xray = 3.0,

        second_order_SFRD = False,
        STOCHASTICITY = False,)

LineParams_input_1_fid = dict(
        LINE = 'OIII',
        nu_rest = 5.985e14, 
        OBSERVABLE_LIM = 'Inu',
        _R = 3, # for now we consider radius larger than NL level 
        LINE_MODEL = 'Yang24',
        sigma_LSFR = 0.,)

LineParams_input_2_fid = dict(
        LINE = 'Ha',
        nu_rest = 4.57e14, 
        OBSERVABLE_LIM = 'Inu',
        _R = 3, # for now we consider radius larger than NL level 
        LINE_MODEL = 'Yang24',
        sigma_LSFR = 0.,)
