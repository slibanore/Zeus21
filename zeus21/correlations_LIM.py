"""

Code to compute LIM correlation functions from power spectra and functions of them. Based on correlations.py 

Author: Sarah Libanore
BGU - November 2024

"""
from copy import copy
from .correlations import * 

class Correlations_LIM:
    "Class that calculates and keeps the correlation functions."

    def __init__(self, Line_Parameters, Cosmo_Parameters, ClassCosmo):

        #we choose the k to match exactly the log FFT of input Rtabsmoo.
        self._klistCF, _dummy_ = mcfit.xi2P(Cosmo_Parameters._Rtabsmoo, l=0, lowring=True)(0*Cosmo_Parameters._Rtabsmoo, extrap=False) # this is an array of R since this is the radius for the correlation ! 
        self.NkCF = len(self._klistCF)

        self._PklinCF = np.zeros(self.NkCF) # P(k) in 1/Mpc^3
        for ik, kk in enumerate(self._klistCF):
            self._PklinCF[ik] = ClassCosmo.pk(kk, 0.0) # function .pk(k,z)

        self._xif = mcfit.P2xi(self._klistCF, l=0, lowring=True)

        self.WINDOWTYPE = 'TOPHAT'
        #options are 'TOPHAT', 'TOPHAT1D' and 'GAUSS' (for now). TOPHAT is calibrated for EPS, but GAUSS has less ringing

        # auto LIM
        self.xi_LIMLIM_CF = self.get_xi_LIMLIM_z0(Cosmo_Parameters, Line_Parameters)

        # cross 21cm LIM
        self.xi_RLIM_CF = self.get_xi_R1LIM_z0(Cosmo_Parameters, Line_Parameters)

    # ---- from correlations.py 
    def _WinTH(self,k,R):
        x = k * R
        return 3.0/x**2 * (np.sin(x)/x - np.cos(x))

    def _WinTH1D(self,k,R):
        x = k * R
        return  np.sin(x)/x

    def _WinG(self,k,R):
        x = k * R * constants.RGauss_factor
        return np.exp(-x**2/2.0)

    def Window(self, k, R):
        if self.WINDOWTYPE == 'TOPHAT':
            return self._WinTH(k, R)
        elif self.WINDOWTYPE == 'GAUSS':
            return self._WinG(k, R)
        elif self.WINDOWTYPE == 'TOPHAT1D':
            return self._WinTH1D(k, R)
        else:
            print('ERROR in Window. Wrong type')
    # ----

# ---------------------
# correlation between R1 = varying for 21cm and R2 = fixed resolution for LIM
    def get_xi_R1LIM_z0 (self, Cosmo_Parameters, Line_Parameters):
        "same as get_xi_z0_lin but smoothed over two different radii with Window(k,R) same separations rs as get_xi_z0_lin so it does not output them."
        
        lengthRarray = Cosmo_Parameters.NRs
        windowR1 = self.Window(self._klistCF.reshape(lengthRarray, 1, 1), Cosmo_Parameters._Rtabsmoo.reshape(1, 1, lengthRarray)) 

        windowRLIM = self.Window(self._klistCF.reshape(1, lengthRarray, 1), Line_Parameters._R) # only one value for the resolution but array on the ks
        
        _PkRLIM = np.array([[self._PklinCF]]) * windowR1 * windowRLIM
        
        self.rlist_CF, xi_RLIM_CF = self._xif(_PkRLIM, extrap = False)  #rlist and xi are arrays since they FT the k array and the power spectrum

        return xi_RLIM_CF

# correlation between R1 and R2 = fixed resolution for LIM
    def get_xi_LIMLIM_z0(self,  Cosmo_Parameters, Line_Parameters):
        "same as get_xi_z0_lin but smoothed over two different radii with Window(k,R) \
        same separations rs as get_xi_z0_lin so it does not output them."
        
        #lengthRarray = Cosmo_Parameters.NRs
        windowRLIM_1 = self.Window(self._klistCF, Line_Parameters._R) # only one value for the resolution but array on the ks
        windowRLIM_2 = self.Window(self._klistCF, Line_Parameters._R) # only one value for the resolution but array on the ks
        
        _PkLIMLIM = np.array([[self._PklinCF]]) * windowRLIM_1 * windowRLIM_2
        
        self.rlist_CF, xi_LIMLIM_CF = self._xif(_PkLIMLIM, extrap = False) #rlist and xi are arrays since they FT the k array and the power spectrum

        return xi_LIMLIM_CF
# ---------------------

class Power_Spectra_LIM:
    "Get power spetrum from correlation functions and coefficients"

    def __init__(self, Cosmo_Parameters, Astro_Parameters, Line_Parameters, Correlations, T21_coefficients, LIM_coefficients, RSD_MODE=1):

#        print("STEP 0: Variable Setup")
        #set up some variables following correlations.py
        self._rs_input_mcfit = Correlations.rlist_CF 
        self.klist_PS = Correlations._klistCF
        self.RSD_MODE = RSD_MODE #redshift-space distortion mode. 0 = None (mu=0), 1 = Spherical avg (like 21-cmFAST), 2 = LoS only (mu=1). 2 is more observationally relevant, whereas 1 the standard assumption in sims. 0 is just for comparison with real-space #TODO: mode to save at different mu

        # for the moment we only work with LIM power spectrum, we will then add the cross

        # LIM case
        self.window_LIM = self.get_LIM_window(Cosmo_Parameters, Correlations, LIM_coefficients)

        self._k3over2pi2 = (self.klist_PS**3)/(2.0 * np.pi**2)

        #and now define power spectra:
        #for LIM, first linear
        self._Pk_LIM_lin = self.window_LIM**2 * Correlations._PklinCF

        self.Deltasq_LIM_lin = self._Pk_LIM_lin * self._k3over2pi2 

        # get all non linear correlation function 
        self.get_all_corrs_LIM(Line_Parameters, Cosmo_Parameters, Correlations, LIM_coefficients)

        #nonlinear corrections too:
        if Line_Parameters._R < constants.MAX_R_NONLINEAR:   
            self._d_Pk_LIM_nl = self.get_list_PS(self._deltaxi_LIM,  LIM_coefficients.zintegral) 
        else:
            self._d_Pk_LIM_nl = 0.

        self.Deltasq_LIM = self.Deltasq_LIM_lin + self._d_Pk_LIM_nl * self._k3over2pi2 

       #calculate some growth 
        self._lingrowthd = cosmology.growth(Cosmo_Parameters, LIM_coefficients.zintegral)

        # cross correlation with the density
        self._Pk_dLIM_lin = (self.window_LIM.T * self._lingrowthd).T * Correlations._PklinCF
        
        self._Pk_dLIM =  self._Pk_dLIM_lin

        if(constants.FLAG_DO_DENS_NL) and Line_Parameters._R < constants.MAX_R_NONLINEAR:  #note that the nonlinear terms (cross and auto) below here have the growth already accounted for

            self._d_Pk_dLIM_nl = self.get_list_PS(self._deltaxi_dLIM, LIM_coefficients.zintegral)
            self._Pk_dLIM += self._d_Pk_dLIM_nl

        self.Deltasq_dLIM_lin = self._Pk_dLIM_lin * self._k3over2pi2 
        self.Deltasq_dLIM = self._Pk_dLIM * self._k3over2pi2 

        # in the correlations.py file, there are xa, Tk , deltaNL, xion and their cross correlations

        # finally, step 2 in the correlation file gets the 21cm power spectrum 


    def get_LIM_window(self, Cosmo_Parameters, Correlations, LIM_coefficients): 
        "Returns the LIM window function for all z in zintegral"
        
        # in the LIM case, we do NOT need to fourier transform this combination of coefficients, since they are not dependent on R, but they are simply constants 
        zGreaterMatrix100 = np.copy(LIM_coefficients.zGreaterMatrix)
        zGreaterMatrix100[np.isnan(zGreaterMatrix100)] = 100

        coeffzpLIM = LIM_coefficients.coeff1_LIM

        growthRmatrix = cosmology.growth(Cosmo_Parameters, zGreaterMatrix100) # only 1 R due to resolution 

        coeffRmatrix = LIM_coefficients.coeff2_LIM # only 1 R due to resolution 
        gammaRmatrix = LIM_coefficients.gammaLIM_index * growthRmatrix # only 1 R due to resolution 
        
        _wincoeffsMatrix_z = coeffRmatrix * gammaRmatrix # only 1 R due to resolution - it is a Nz x 1 matrix 
        
        _wincoeffsMatrix = _wincoeffsMatrix_z          
        _win_LIM = _wincoeffsMatrix

        _win_LIM *= np.array([coeffzpLIM]).T

        return _win_LIM

    # same from correlations.py 
    def get_Pk_from_xi(self, rsinput, xiinput):
        "Generic Fourier Transform, returns Pk from an input Corr Func xi. kPf should be the same as _klistCF"

        kPf, Pf = mcfit.xi2P(rsinput, l=0, lowring=True)(xiinput, extrap=False)

        return kPf, Pf


    def get_all_corrs_LIM(self, Line_Parameters, Cosmo_Parameters, Correlations, LIM_coefficients):
        "Returns the LIM components of the correlation functions of all observables at each z in zintegral"

        zGreaterMatrix100 = np.copy(LIM_coefficients.zGreaterMatrix)
        zGreaterMatrix100[np.isnan(zGreaterMatrix100)] = 100

        # corrdNL is the correlation xi^R0R0*(k,z) = FT(P(k)W^R0(k))^2) 
        # np._ix indexes its first and second dimension i.e. the Rs one!
        # hence for us it just becomes a flag
        if Line_Parameters._R < constants.MAX_R_NONLINEAR:
            
            _iRnonlinear = [-1] 
            corrdNL = Correlations.xi_LIMLIM_CF[np.ix_(_iRnonlinear,_iRnonlinear)]
            #for R<RNL fix at RNL, avoids corelations blowing up at low R
            if Line_Parameters._R < constants.MIN_R_NONLINEAR:
                print('Your resolution introduces too large non linear corrections on small scales! ')
                print('It should have been changed in the input file, check why this did not happened')

            corrdNL = corrdNL.reshape((1, *corrdNL.shape))
        else:
            _iRnonlinear = [-1] 
            print('The resolution allows for fully linear calculation')

        growthRmatrix = cosmology.growth(Cosmo_Parameters,zGreaterMatrix100[:, _iRnonlinear])
        gammaR1 = LIM_coefficients.gammaLIM_index[:, _iRnonlinear] * growthRmatrix

        coeffzp1_LIM = LIM_coefficients.coeff1_LIM

        coeffR1_LIM = LIM_coefficients.coeff2_LIM[:,_iRnonlinear]

        coeffmatrix_LIM = coeffR1_LIM.reshape(len(LIM_coefficients.zintegral), 1, len(_iRnonlinear),1) * coeffR1_LIM.reshape(len(LIM_coefficients.zintegral), len(_iRnonlinear), 1,1)

        if Line_Parameters._R < constants.MAX_R_NONLINEAR:
            gammamatrixR1R1 = gammaR1.reshape(len(LIM_coefficients.zintegral), 1, len(_iRnonlinear),1) * gammaR1.reshape(len(LIM_coefficients.zintegral), len(_iRnonlinear), 1,1)

            gammaTimesCorrdNL = ne.evaluate('gammamatrixR1R1 * corrdNL')#np.einsum('ijkl,ijkl->ijkl', gammamatrixR1R1, corrdNL, optimize = True) #same thing as gammamatrixR1R1 * corrdNL but faster
            expGammaCorrMinusLinear = ne.evaluate('exp(gammaTimesCorrdNL) - 1 - gammaTimesCorrdNL')

            self._deltaxi_LIM = np.einsum('ijkl->il', coeffmatrix_LIM * expGammaCorrMinusLinear, optimize = True)
            self._deltaxi_LIM *= np.array([coeffzp1_LIM]).T**2 #brings it to Inu units

        if (constants.FLAG_DO_DENS_NL) and Line_Parameters._R < constants.MAX_R_NONLINEAR:

            D_coeffR1LIM = coeffR1_LIM.reshape(*coeffR1_LIM.shape, 1)
            D_gammaR1 = gammaR1.reshape(*gammaR1.shape , 1)
            D_growthRmatrix = growthRmatrix[:,:1].reshape(*growthRmatrix[:,:1].shape, 1)
            D_corrdNL = corrdNL[:1,0,:,:]

            self._deltaxi_dLIM = np.sum(D_coeffR1LIM * ((np.exp(D_gammaR1 * D_growthRmatrix * D_corrdNL )-1.0 ) - D_gammaR1 * D_growthRmatrix * D_corrdNL), axis = 1)

            self._deltaxi_dLIM *= np.array([coeffzp1_LIM]).T
            
            
        return 1


    def get_list_PS(self, xi_list, zlisttoconvert):
        "Returns the power spectrum given a list of CFs (xi_list) evaluated at z=zlisttoconvert as input"

        _Pk_list = []

        for izp,zp in enumerate(zlisttoconvert):

            _kzp, _Pkzp = self.get_Pk_from_xi(self._rs_input_mcfit,xi_list[izp])
            _Pk_list.append(_Pkzp)
            #can ignore _kzp, it's the same as klist_PS above by construction


        return np.array(_Pk_list)
