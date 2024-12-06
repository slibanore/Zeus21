"""
Takes inputs for LIM and stores them in useful classes

Author: Sarah Libanore
BGU - October 2024
"""

class LineParams_Input:
    def __init__ (self, 
                 LINE = 'CII',
                 OBSERVABLE_LIM = 'Tnu',
                 _R = 3, # for now we consider radius larger than NL level 
                 CII_MODEL = 'Lagache18',
                 CII_nu_rest = 1.897e12, 
                 CII_alpha_SFR_0 = 1.4,
                 CII_alpha_SFR_z = -0.07,
                 CII_beta_SFR_0 = 7.1,
                 CII_beta_SFR_z = -0.07,
                 CII_sigma_LSFR = 0.5):
        
        self.LINE = LINE
        self.OBSERVABLE_LIM = OBSERVABLE_LIM
        self._R = _R
        self.CII_MODEL = CII_MODEL
        self.CII_nu_rest = CII_nu_rest
        self.CII_alpha_SFR_0 = CII_alpha_SFR_0
        self.CII_alpha_SFR_z = CII_alpha_SFR_z
        self.CII_beta_SFR_0 = CII_beta_SFR_0
        self.CII_beta_SFR_z = CII_beta_SFR_z
        self.CII_sigma_LSFR = CII_sigma_LSFR



class Line_Parameters:
    "Class to pass the parameters of LIM as input"

    def __init__(self, LineParams_Input):
        
        self.LINE = LineParams_Input.LINE # which line to use 
        self.OBSERVABLE_LIM = LineParams_Input.OBSERVABLE_LIM
                
        # resolution in Mpc, default same value as Cosmo_Params.Rsmmin. Should be set by the point where the exponential approximation breaks (?)                   
        if LineParams_Input._R < 2.:
            print('Your resolution introduces too large non linear corrections on small scales! ')
            print('We use instead MIN_R_NONLINEAR = 2 Mpc')
            self._R = 2.
        else:
            self._R = LineParams_Input._R 

        self.CII_MODEL = LineParams_Input.CII_MODEL 
        self.CII_nu_rest = LineParams_Input.CII_nu_rest # rest frame frequency in Hz (= 158 um)
        self.CII_alpha_SFR_0 = LineParams_Input.CII_alpha_SFR_0 # param of line luminosity - SFR model 
        self.CII_alpha_SFR_z = LineParams_Input.CII_alpha_SFR_z # param of line luminosity - SFR model 
        self.CII_beta_SFR_0 = LineParams_Input.CII_beta_SFR_0  # param of line luminosity - SFR model  
        self.CII_beta_SFR_z = LineParams_Input.CII_beta_SFR_z  # param of line luminosity - SFR model  

        self.CII_sigma_LSFR = LineParams_Input.CII_sigma_LSFR # scatter in the luminosity - SFR relation