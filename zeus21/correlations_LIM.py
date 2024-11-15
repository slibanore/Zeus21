"""

Code to compute LIM correlation functions from power spectra and functions of them. Based on correlations.py 

Author: Sarah Libanore
BGU - November 2024

"""

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
        
        lengthRarray = Cosmo_Parameters.NRs
        windowRLIM_1 = self.Window(self._klistCF.reshape(lengthRarray, 1, 1), Line_Parameters._R) # only one value for the resolution but array on the ks
        windowRLIM_2 = self.Window(self._klistCF.reshape(1, lengthRarray, 1), Line_Parameters._R) # only one value for the resolution but array on the ks
        
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
        _dummy_kwindow_LIM, self.window_LIM = self.get_LIM_window(Cosmo_Parameters, Correlations, LIM_coefficients)

        #calculate some growth etc, and the bubble biases for the xHI linear window function:
        self._lingrowthd = cosmology.growth(Cosmo_Parameters, LIM_coefficients.zintegral)

        # get all non linear correlation function 
        self.get_all_corrs_LIM(Cosmo_Parameters, Correlations, LIM_coefficients)

        self._k3over2pi2 = (self.klist_PS**3)/(2.0 * np.pi**2)

        #and now define power spectra:
        #for LIM, first linear
        self._Pk_LIM_lin = self.window_LIM**2 * Correlations._PklinCF

        self.Deltasq_LIM_lin = self._Pk_LIM_lin * self._k3over2pi2 

        #nonlinear corrections too:
        self._d_Pk_LIM_nl = self.get_list_PS(self._deltaxi_LIM,  LIM_coefficients.zintegral) 

        self.Deltasq_LIM = self.Deltasq_LIM_lin + self._d_Pk_LIM_nl * self._k3over2pi2 

        # in the correlations.py file, there are xa, Tk , deltaNL, xion and their cross correlations

        # finally, step 2 in the correlation file gets the 21cm power spectrum 


    def get_LIM_window(self, Cosmo_Parameters, Correlations, LIM_coefficients): 
        "Returns the LIM window function for all z in zintegral"
        
        zGreaterMatrix100 = np.copy(LIM_coefficients.zGreaterMatrix)
        zGreaterMatrix100[np.isnan(zGreaterMatrix100)] = 100

        coeffzpLIM = LIM_coefficients.coeff1_LIM

        growthRmatrix = cosmology.growth(Cosmo_Parameters, zGreaterMatrix100) # only 1 R due to resolution 

        coeffRmatrix = LIM_coefficients.coeff2_LIM # only 1 R due to resolution 
        gammaRmatrix = LIM_coefficients.gammaLIM_index * growthRmatrix # only 1 R due to resolution 

        _wincoeffs = coeffRmatrix * gammaRmatrix # only 1 R due to resolution - it is a Nz x 1 matrix but since we need to reshape to Nz x NRs

        _wincoeffsMatrix = np.transpose([_wincoeffs[:,0]]) + Cosmo_Parameters._Rtabsmoo
        
        if(Cosmo_Parameters.Flag_emulate_21cmfast==False): #do the standard 1D TopHat

            # ??? this scaling related to the R in the correlation function (array), in the integral of the FFT 
            _wincoeffsMatrix /= (4*np.pi * Cosmo_Parameters._Rtabsmoo**2) * (Cosmo_Parameters._Rtabsmoo * Cosmo_Parameters._dlogRR) 

            # here we use array of R since this is where the correlation function is computed
            _kwinLIM, _win_LIM = self.get_Pk_from_xi(Cosmo_Parameters._Rtabsmoo, _wincoeffsMatrix)

        else:
            _kwinLIM = self.klist_PS

            coeffRgammaRmatrix = coeffRmatrix * gammaRmatrix
            coeffRgammaRmatrix = coeffRgammaRmatrix.reshape(*coeffRgammaRmatrix.shape, 1)

            dummyMesh, RLIMMesh, kWinLIMMesh = np.meshgrid(LIM_coefficients.zintegral, [LIM_coefficients.Rtabsmoo_LIM], _kwinLIM, indexing = 'ij', sparse = True)

            _win_LIM = coeffzpLIM * coeffRgammaRmatrix * Correlations._WinTH(RLIMMesh, kWinLIMMesh)
            # _win_LIM = np.sum(_win_LIM, axis = 1) ??? shouldn't be needed since we have only 1 R 

        return _kwinLIM, _win_LIM

    # same from correlations.py 
    def get_Pk_from_xi(self, rsinput, xiinput):
        "Generic Fourier Transform, returns Pk from an input Corr Func xi. kPf should be the same as _klistCF"

        kPf, Pf = mcfit.xi2P(rsinput, l=0, lowring=True)(xiinput, extrap=False)

        return kPf, Pf


    def get_all_corrs_LIM(self, Cosmo_Parameters, Correlations, LIM_coefficients):
        "Returns the LIM components of the correlation functions of all observables at each z in zintegral"
    
        zGreaterMatrix100 = np.copy(LIM_coefficients.zGreaterMatrix)
        zGreaterMatrix100[np.isnan(zGreaterMatrix100)] = 100

        # does this refer to the correlation or to the smoothing?
        _iRnonlinear = np.arange(Cosmo_Parameters.indexmaxNL)

        #for R<RNL fix at RNL, avoids corelations blowing up at low R
        corrdNL[0:Cosmo_Parameters.indexminNL,0:Cosmo_Parameters.indexminNL] = corrdNL[Cosmo_Parameters.indexminNL,Cosmo_Parameters.indexminNL]
        corrdNL = corrdNL.reshape((1, *corrdNL.shape))

        growthRmatrix = cosmology.growth(Cosmo_Parameters,zGreaterMatrix100)

        gammaR1 = LIM_coefficients.gammaLIM_index * growthRmatrix

        coeffzp1LIM = LIM_coefficients.coeff1_LIM
        coeffR1LIM = LIM_coefficients.coeff2_LIM

        gammamatrixR1R1 = gammaR1.reshape(len(LIM_coefficients.zintegral), 1, 1,1) * gammaR1.reshape(len(LIM_coefficients.zintegral), 1, 1,1)

        coeffmatrixLIM = coeffR1LIM.reshape(len(LIM_coefficients.zintegral), 1, 1,1) * coeffR1LIM.reshape(len(LIM_coefficients.zintegral), 1, 1,1)

        gammaTimesCorrdNL = ne.evaluate('gammamatrixR1R1 * corrdNL')
        expGammaCorrMinusLinear = ne.evaluate('exp(gammaTimesCorrdNL) - 1 - gammaTimesCorrdNL')

        self._deltaxi_LIM = np.einsum('ijkl->il', coeffmatrixLIM * expGammaCorrMinusLinear, optimize = True)
        self._deltaxi_LIM *= np.array([coeffzp1LIM]).T**2 # units ?? 

        if (constants.FLAG_DO_DENS_NL):
            D_coeffRLIM = coeffR1LIM.reshape(*coeffR1LIM.shape, 1)
            D_gammaR1 = gammaR1.reshape(*gammaR1.shape , 1)
            D_growthRmatrix = growthRmatrix[:,:1].reshape(*growthRmatrix[:,:1].shape, 1)
            D_corrdNL = corrdNL[:1,0,:,:]

            self._deltaxi_dLIM = np.sum(D_coeffRLIM * ((np.exp(D_gammaR1 * D_growthRmatrix * D_corrdNL )-1.0 ) - D_gammaR1 * D_growthRmatrix * D_corrdNL), axis = 1)

            self._deltaxi_dLIM *= np.array([coeffzp1LIM]).T

            self._deltaxi_d = (np.exp(growthRmatrix[:,:1]**2 * corrdNL[0,0,0,:]) - 1.0) - growthRmatrix[:,:1]**2 * corrdNL[0,0,0,:]
            
        return 1


    def get_list_PS(self, xi_list, zlisttoconvert):
        "Returns the power spectrum given a list of CFs (xi_list) evaluated at z=zlisttoconvert as input"

        _Pk_list = []

        for izp,zp in enumerate(zlisttoconvert):

            _kzp, _Pkzp = self.get_Pk_from_xi(self._rs_input_mcfit,xi_list[izp])
            _Pk_list.append(_Pkzp)
            #can ignore _kzp, it's the same as klist_PS above by construction


        return np.array(_Pk_list)
