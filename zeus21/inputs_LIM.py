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
                 alpha_SFR_0 = 1.4,
                 beta_SFR_0 = 7.1,
                 sigma_LSFR = 0.,

                 CII_alpha_SFR_z = -0.07,
                 CII_beta_SFR_z = -0.07,

                 line_N = 0.,
                 line_SFR1 = 0.
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
        # power law coefficients (constant)
        self.alpha_SFR_0 = alpha_SFR_0
        self.beta_SFR_0 = beta_SFR_0
        # stochasticity
        self.sigma_LSFR = sigma_LSFR

        # parameters that are good for CII, from Lagache2018
        # parameter evolution in z 
        self.CII_alpha_SFR_z = CII_alpha_SFR_z
        self.CII_beta_SFR_z = CII_beta_SFR_z

        # parameters that are good for OIII and Halpha, Hbeta, from Yang2024
        # amplitude and SFR normalization
        self.line_N = line_N
        self.line_SFR1 = line_SFR1


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

        self.alpha_SFR_0 = LineParams_Input.alpha_SFR_0 # param of line luminosity - SFR model 
        self.beta_SFR_0 = LineParams_Input.beta_SFR_0  # param of line luminosity - SFR model  
        self.sigma_LSFR = LineParams_Input.sigma_LSFR # scatter in the luminosity - SFR relation

        # for Lagache CII model
        self.CII_alpha_SFR_z = LineParams_Input.CII_alpha_SFR_z # param of line luminosity - SFR model 
        self.CII_beta_SFR_z = LineParams_Input.CII_beta_SFR_z  # param of line luminosity - SFR model  

        # for Yang OIII and Halpha beta model
        self.line_N = LineParams_Input.line_N # param of line luminosity - SFR model 
        self.line_SFR1 = LineParams_Input.line_SFR1  # param of line luminosity - SFR model  



# parameter sets of our default models 
CII_params = {
        'LINE': 'CII',
        'nu_rest': 1.897e12, 
        'OBSERVABLE_LIM': 'Tnu',
        '_R ': 3, 
        'LINE_MODEL': 'Lagache18',
        'alpha_SFR_0': 1.4-0.07*10,
        'beta_SFR_0': 7.1-0.07*10,
        'sigma_LSFR': 0.,

        'CII_alpha_SFR_0': 0.,
        'CII_beta_SFR_0': 0.,
    }

OIII_params = {
        'LINE': 'OIII',
        'nu_rest': [6.042e14, 5.985e14], # 4960 AA, 5007 AA # multi line available
        'OBSERVABLE_LIM': 'Tnu',
        '_R ': 3, 
        'LINE_MODEL': 'Yang24', # 2409.03997
        'alpha_SFR_0': [9.82e-2, 9.83e-2],
        'beta_SFR_0': [6.90e-1, 6.92e-1],
        'sigma_LSFR': 0.,

        'line_N': [2.75e7, 8.07e7],
        'line_SFR1': [1.24e2, 1.28e2],
    }

OII_params = {
        'LINE': 'OII',
        'nu_rest': 8.041e14, # 3727.29 AA 
        'OBSERVABLE_LIM': 'Tnu',
        '_R ': 3, 
        'LINE_MODEL': 'Yang24', # 2409.03997
        'alpha_SFR_0': -2.43e-1,
        'beta_SFR_0': 2.5,
        'sigma_LSFR': 0.,

        'line_N': 2.14e6,
        'line_SFR1': 5.91e1,
    }