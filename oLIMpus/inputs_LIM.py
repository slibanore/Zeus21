"""
Takes inputs for LIM and stores them in useful classes

Author: Sarah Libanore
BGU - October 2024
"""

class LineParams_Input:
    def __init__ (self, 
                 LINE = 'CII',
                 nu_rest = 1.897e12, 
                 OBSERVABLE_LIM = 'Tnu',
                 _R = 3, # for now we consider radius larger than NL level 
                 LINE_MODEL = 'Lagache18',
                 sigma_LSFR = 0.,
                 ):
        
        # parameters for all lines
        # which line and its rest frame frequency 
        self.LINE = LINE
        self.nu_rest = nu_rest
        # which observable
        self.OBSERVABLE_LIM = OBSERVABLE_LIM
        # radius for the fcoll computation
        self._R = _R
        # which model to use
        self.LINE_MODEL = LINE_MODEL
        # stochasticity
        self.sigma_LSFR = sigma_LSFR


class Line_Parameters:
    "Class to pass the parameters of LIM as input"

    def __init__(self, LineParams_Input):
        
        self.LINE = LineParams_Input.LINE # which line to use 
        self.OBSERVABLE_LIM = LineParams_Input.OBSERVABLE_LIM
        self.nu_rest = LineParams_Input.nu_rest # rest frame frequency in Hz (= 158 um)
                
        # resolution in Mpc, default same value as Cosmo_Params.Rsmmin. Should be set by the point where the exponential approximation breaks (?)                   
        if LineParams_Input._R < 2.:
            print('Your resolution introduces too large non linear corrections on small scales! ')
            print('We use instead MIN_R_NONLINEAR = 2 Mpc')
            self._R = 2.
        else:
            self._R = LineParams_Input._R 

        self.LINE_MODEL = LineParams_Input.LINE_MODEL 

        self.sigma_LSFR = LineParams_Input.sigma_LSFR # scatter in the luminosity - SFR relation




# parameter sets of our default models 
Lagache_CII_params = {
        'LINE': 'CII',
        'nu_rest': 1.897e12, 
        'OBSERVABLE_LIM': 'Tnu',
        '_R ': 3, 
        'LINE_MODEL': 'Lagache18',
        'alpha_SFR_0': 1.4-0.07*10,
        'beta_SFR_0': 7.1-0.07*10,
        'sigma_LSFR': 0.,
        'alpha_SFR': 0.,
        'beta_SFR': 0.,
    }

Yang_OIII_params = [{
        'LINE': 'OIII',
        'nu_rest': 6.042e14, # 4960 AA, 5007 AA # multi line available
        'OBSERVABLE_LIM': 'Tnu',
        '_R ': 3, 
        'LINE_MODEL': 'Yang24', # 2409.03997
        'alpha_SFR_0': 9.82e-2,
        'beta_SFR_0': 6.90e-1,
        'sigma_LSFR': 0.,
        'N': 2.75e7,
        'SFR1': 1.24e2,
    },
    {
        'LINE': 'OIII',
        'nu_rest': 5.985e14, # 4960 AA, 5007 AA # multi line available
        'OBSERVABLE_LIM': 'Tnu',
        '_R ': 3, 
        'LINE_MODEL': 'Yang24', # 2409.03997
        'alpha_SFR_0': 9.83e-2,
        'beta_SFR_0': 6.92e-1,
        'sigma_LSFR': 0.,
        'N': 8.07e7,
        'SFR1': 1.28e2,
    }]

Yang_OII_params = {
        'LINE': 'OII',
        'nu_rest': 8.05e14, # 3727 AA 
        'OBSERVABLE_LIM': 'Tnu',
        '_R ': 3, 
        'LINE_MODEL': 'Yang24', # 2409.03997
        'alpha_SFR_0': -2.43e-1,
        'beta_SFR_0': 2.5,
        'sigma_LSFR': 0.,
        'N': 2.14e6,
        'SFR1': 5.91e1,
    }

Yang_Halpha_params = {
        'LINE': 'Ha',
        'nu_rest': 4.57e14, # 656.28 AA 
        'OBSERVABLE_LIM': 'Tnu',
        '_R ': 3, 
        'LINE_MODEL': 'Yang24', # 2409.03997
        'alpha_SFR_0': 9.94e-3,
        'beta_SFR_0': 5.25e-1,
        'sigma_LSFR': 0.,
        'N': 4.54e7,
        'SFR1': 3.18e1,
    }

Yang_Hbeta_params = {
        'LINE': 'Hb',
        'nu_rest': 6.17e14, # 486.13 AA 
        'OBSERVABLE_LIM': 'Tnu',
        '_R ': 3, 
        'LINE_MODEL': 'Yang24', # 2409.03997
        'alpha_SFR_0': 7.98e-3,
        'beta_SFR_0': 5.61e-1,
        'sigma_LSFR': 0.,
        'N': 1.61e7,
        'SFR1': 1.74e1,
    }


THESAN_OIII_params = {
        'LINE': 'OIII',
        'nu_rest': 6.17e14, # 486.13 AA 
        'OBSERVABLE_LIM': 'Tnu',
        '_R ': 3, 
        'LINE_MODEL': 'THESAN21', # 2111.02411
        'a': 7.84,
        'ma': 1.24,
        'mb': 1.19,
        'log10_SFR_b': 0., 
        'mc': 0.53,
        'log10_SFR_c':0.66
    }


THESAN_OII_params = {
        'LINE': 'OII',
        'nu_rest': 8.05e14, # 486.13 AA 
        'OBSERVABLE_LIM': 'Tnu',
        '_R ': 3, 
        'LINE_MODEL': 'THESAN21', # 2111.02411
        'a': 7.08,
        'ma': 1.11,
        'mb': 1.31,
        'log10_SFR_b': 0.,
        'mc': 0.64,
        'log10_SFR_c':0.54
    }

THESAN_Halpha_params = {
        'LINE': 'Ha',
        'nu_rest': 4.57e14, # 486.13 AA 
        'OBSERVABLE_LIM': 'Tnu',
        '_R ': 3, 
        'LINE_MODEL': 'THESAN21', # 2111.02411
        'a': 8.08,
        'ma': 0.96,
        'mb': 0.88,
        'log10_SFR_b':0.,
        'mc': 0.45,
        'log10_SFR_c':0.96
    }

THESAN_Hbeta_params = {
        'LINE': 'Hb',
        'nu_rest': 6.17e14, # 486.13 AA 
        'OBSERVABLE_LIM': 'Tnu',
        '_R ': 3, 
        'LINE_MODEL': 'THESAN21', # 2111.02411
        'a': 7.62,
        'ma': 0.96,
        'mb': 0.86,
        'log10_SFR_b':0.,
        'mc': 0.41,
        'log10_SFR_c':0.96
    }
