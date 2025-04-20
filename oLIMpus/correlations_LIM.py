"""

Code to compute LIM correlation functions from power spectra and functions of them. Based on correlations.py 

Author: Sarah Libanore
BGU - November 2024

"""
from copy import copy
from zeus21.correlations import * 

class Correlations_LIM:
    "Class that calculates and keeps the correlation functions."

    def __init__(self, Line_Parameters, Cosmo_Parameters, ClassCosmo):

        #we choose the k to match exactly the log FFT of input Rtabsmoo.
        self._klistCF, _dummy_ = mcfit.xi2P(Cosmo_Parameters._Rtabsmoo, l=0, lowring=True)(0*Cosmo_Parameters._Rtabsmoo, extrap=False) # this is an array of r since this is the radius for the correlation ! 
        self.NkCF = len(self._klistCF)

        # linear matter power spectrum at z = 0 
        self._PklinCF = np.zeros(self.NkCF) # P(k) in 1/Mpc^3
        for ik, kk in enumerate(self._klistCF):
            self._PklinCF[ik] = ClassCosmo.pk(kk, 0.0) # function .pk(k,z)

        # function to transform the power specrum in correlation function
        self._xif = mcfit.P2xi(self._klistCF, l=0, lowring=True)

        # smooth window
        self.WINDOWTYPE = 'TOPHAT'
        #options are 'TOPHAT', 'TOPHAT1D' and 'GAUSS' (for now). TOPHAT is calibrated for EPS, but GAUSS has less ringing

        # linear matter correlation functon smoothed over the same R0 (the LIM input) at z = 0 
        self.xi_linearmatter_R0R0 = self.get_xi_R0_z0(Line_Parameters) 

        # linear matter correlation functon smoothed over the R0 (the LIM input) and array of R at z = 0  
        # ---> not needed for the LIM auto case, will maybe be needed to cross with 21cm analytically
        self.xi_linearmatter_RarrR0 = self.get_xi_RarrR0_z0(Cosmo_Parameters, Line_Parameters)

    # ---- same as in correlations.py 
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
    # linear matter correlation function 
    def get_xi_R0_z0(self,  Line_Parameters):
        
        #lengthRarray = Cosmo_Parameters.NRs
        windowR = self.Window(self._klistCF, Line_Parameters._R) # only one value for the resolution but array on the ks
        
        _PkLIMLIM = np.array([self._PklinCF]) * windowR**2 

        self.rlist_CF, xi_R0R0_CF = self._xif(_PkLIMLIM, extrap = False) #rlist and xi are arrays since they FT the k array and the power spectrum

        return xi_R0R0_CF

# correlation between R1 = varying for 21cm and R2 = fixed resolution for LIM
    def get_xi_RarrR0_z0 (self, Cosmo_Parameters, Line_Parameters):
        "same as get_xi_z0_lin but smoothed over two different radii with Window(k,R) same separations rs as get_xi_z0_lin so it does not output them."
        
        lengthRarray = Cosmo_Parameters.NRs
        windowRarr = self.Window(self._klistCF.reshape(lengthRarray, 1, 1), Cosmo_Parameters._Rtabsmoo.reshape(1, 1, lengthRarray)) 

        windowR0 = self.Window(self._klistCF.reshape(1, lengthRarray, 1), Line_Parameters._R) # only one value for the resolution but array on the ks
        
        _PkRLIM = np.array([[self._PklinCF]]) * windowRarr * windowR0
        
        self.rlist_CF, xi_RarrR0_CF = self._xif(_PkRLIM, extrap = False)  #rlist and xi are arrays since they FT the k array and the power spectrum

        return xi_RarrR0_CF
# ---------------------


class Power_Spectra_LIM:
    "Get LIM auto power spetrum from correlation functions and coefficients as function of z"

    def __init__(self, Cosmo_Parameters, Astro_Parameters, Line_Parameters, Correlations, LIM_coefficients, RSD_MODE=1):

#        print("STEP 0: Variable Setup")
        self._r_CF = Correlations.rlist_CF # array of r for the correlation function

        self.klist_PS = Correlations._klistCF # array of k for the power spectrum 
        self._k3over2pi2 = (self.klist_PS**3)/(2.0 * np.pi**2)

        self.RSD_MODE = RSD_MODE #redshift-space distortion mode. 0 = None (mu=0), 1 = Spherical avg (like 21-cmFAST), 2 = LoS only (mu=1). 2 is more observationally relevant, whereas 1 the standard assumption in sims. 0 is just for comparison with real-space #TODO: mode to save at different mu

        # LIM case
        '''
        #and now define power spectra:
        # LINEAR LIM power spectrum 
        # c1^2 c2^2 gamma_R0^2 D^2 Pk_lin 
        self._Pk_LIM_lin = self.window_LIM**2 * Correlations_LIM._PklinCF

        self.Deltasq_LIM_lin = self._Pk_LIM_lin * self._k3over2pi2 

        # get all non linear correlation function 
        self.get_all_corrs_LIM(Line_Parameters, Astro_Parameters, Cosmo_Parameters, Correlations, LIM_coefficients)

        # NON LINEAR LIM power spectrum 
        # computed by transforming to Pk the correlation function estimated from the rho_L approximation
        # corrections  ????
        if Line_Parameters._R < constants.MAX_R_NONLINEAR:   
            self._corr_Pk_LIM_nl = self.get_list_PS(self._deltaxi_LIM,  LIM_coefficients.zintegral) 
        else:
            self._corr_Pk_LIM_nl = 0.
        self.Deltasq_LIM = self.Deltasq_LIM_lin + self._corr_Pk_LIM_nl * self._k3over2pi2 

        # WE ALSO NEED THE CROSS LIM-MATTER TO PRODUCE THE MAPS

        # calculate growth factor for the density part
        self._lingrowthd = cosmology.growth(Cosmo_Parameters, LIM_coefficients.zintegral)
        '''

        # LINEAR LIM power spectrum 
        # c1^2 c2^2 gamma_R0^2 D^2 Pk_lin 
        self.window_LIM = self.get_LIM_window(Cosmo_Parameters, Astro_Parameters, LIM_coefficients)
        self._Pk_LIM_lin = self.window_LIM**2 * Correlations._PklinCF
        self.Deltasq_LIM_lin = self._Pk_LIM_lin * self._k3over2pi2 
        
        if Line_Parameters._R > constants.MAX_R_NONLINEAR:   
            # on linear scales
            self._Pk_LIM = self._Pk_LIM_lin
            self.Deltasq_LIM = self.Deltasq_LIM_lin 
            
        else:
            # NON LINEAR LIM power spectrum 
            # get the full non linear correlation function from <exp()exp()>
            self.get_all_corrs_LIM(Line_Parameters, Astro_Parameters, Cosmo_Parameters,Correlations, LIM_coefficients)
            # transform to power spectrum 
            self._Pk_LIM = self.get_list_PS(self._xiR0_LIM,  LIM_coefficients.zintegral)

            #self._Pk_LIM.T[:Cosmo_Parameters.indexminNL] = self._Pk_LIM_lin.T[:Cosmo_Parameters.indexminNL]            

            self.Deltasq_LIM = self._Pk_LIM * self._k3over2pi2 

        # WE ALSO NEED THE CROSS LIM-MATTER TO PRODUCE THE MAPS

        # calculate growth factor for the density part
        self._lingrowthd = cosmology.growth(Cosmo_Parameters, LIM_coefficients.zintegral)

        # LINEAR cross correlation delta-lim
        # c1 c2 gamma_R0 D^2 Pk_lin
        self._Pk_deltaLIM_lin = (self.window_LIM.T * self._lingrowthd).T * Correlations._PklinCF
        self.Deltasq_deltaLIM_lin = self._Pk_deltaLIM_lin * self._k3over2pi2 

        # NON LINEAR cross correlation delta-lim 
        self._Pk_deltaLIM =  self._Pk_deltaLIM_lin

        #!!!!!!!!!!!!!!!!!!!! TO CORRECT
        if(constants.FLAG_DO_DENS_NL) and Line_Parameters._R < constants.MAX_R_NONLINEAR:  #note that the nonlinear terms (cross and auto) below here have the growth already accounted for

            self._corr_Pk_deltaLIM_nl = self.get_list_PS(self._deltaxi_deltaLIM, LIM_coefficients.zintegral)
            self._Pk_deltaLIM += self._corr_Pk_deltaLIM_nl

        self.Deltasq_deltaLIM = self._Pk_deltaLIM * self._k3over2pi2 


    def get_LIM_window(self, Cosmo_Parameters, Astro_Parameters, LIM_coefficients): 
        "Returns the LIM window function for all z in zintegral"
        
        # in the LIM case, we do NOT need to fourier transform this combination of coefficients, since they are not dependent on R, but they are simply constants 
        zGreaterMatrix100 = np.copy(LIM_coefficients.zGreaterMatrix)
        zGreaterMatrix100[np.isnan(zGreaterMatrix100)] = 100

        c1_LIM = LIM_coefficients.coeff1_LIM

        growth_R0 = cosmology.growth(Cosmo_Parameters, zGreaterMatrix100) # only 1 R due to resolution 

        c2_LIM = LIM_coefficients.coeff2_LIM # only 1 R due to resolution 
        gamma_R0 = LIM_coefficients.gammaLIM_index * growth_R0 # only 1 R due to resolution 
        
        if Astro_Parameters.second_order_SFRD:
            _win_LIM = c2_LIM * gamma_R0*LIM_coefficients.sigmaofRtab_LIM * np.array([c1_LIM]).T /(-1+2.*LIM_coefficients.gamma2_LIM_index*LIM_coefficients.sigmaofRtab_LIM**2) # only 1 R due to resolution - it is a Nz x 1 matrix 
        else:
            _win_LIM = c2_LIM * (gamma_R0 *LIM_coefficients.sigmaofRtab_LIM)* np.array([c1_LIM]).T # only 1 R due to resolution - it is a Nz x 1 matrix 
        
        return _win_LIM
    

    def get_all_corrs_LIM(self, Line_Parameters, Astro_Parameters, Cosmo_Parameters, Correlations, LIM_coefficients):
        "Returns the LIM components of the correlation functions of all observables at each z in zintegral"

        zGreaterMatrix100 = np.copy(LIM_coefficients.zGreaterMatrix)
        zGreaterMatrix100[np.isnan(zGreaterMatrix100)] = 100

        # if we are running this function, it means that we are in non linear scales! 

        # these are all simply array in z and k / r
        growthRmatrix = (cosmology.growth(Cosmo_Parameters,zGreaterMatrix100))**2

        gammaR0 = LIM_coefficients.gammaLIM_index
        sigmaR0 = LIM_coefficients.sigmaofRtab_LIM
        g1 = gammaR0 * sigmaR0

        xi_matter_R0_z0 = Correlations.xi_linearmatter_R0R0

        xi_matter_R0_z = ne.evaluate('xi_matter_R0_z0 * growthRmatrix/ (sigmaR0 * sigmaR0)')
        xi_LIM_R0_z = ne.evaluate('g1 * g1 * xi_matter_R0_z')

        if Astro_Parameters.second_order_SFRD:

            gammaR0_NL = LIM_coefficients.gamma2_LIM_index
            g1NL = gammaR0_NL * sigmaR0**2

            numerator_NL = ne.evaluate('xi_LIM_R0_z + 2 * g1 * g1 * (0.5 - g1NL * (1 - xi_matter_R0_z * xi_matter_R0_z))')
            
            denominator_NL = ne.evaluate('1. - 4 * g1NL + 4 * g1NL * g1NL * (1 - xi_matter_R0_z * xi_matter_R0_z)')
            
            norm1 = ne.evaluate('exp(g1 * g1 / (2 - 4 * g1NL)) / sqrt(1 - 2 * g1NL)') 
            
            log_norm = ne.evaluate('log(sqrt(denominator_NL) * norm1 * norm1)')

            nonlinearcorrelation = ne.evaluate('exp(numerator_NL/denominator_NL - log_norm)-1')

            #nonlinearcorrelation = self.exponentialcorrelation_quad(gammaR0,gammaR0_NL, gammaR0, gammaR0_NL, sigmaR0, sigmaR0, xi_matter_R0_z0*growthRmatrix)
        else:
            nonlinearcorrelation = ne.evaluate('exp(xi_LIM_R0_z)-1')

            # nonlinearcorrelation = self.exponentialcorrelation_linear(gammaR0, gammaR0, sigmaR0, sigmaR0, xi_matter_R0_z0*growthRmatrix)

        nonlinearcorrelation[0:Cosmo_Parameters.indexminNL] = nonlinearcorrelation[Cosmo_Parameters.indexminNL]
        # compared with the 21cm case, here there is no sum since we are using only 1 R0 
        c1_LIM = LIM_coefficients.coeff1_LIM
        c2_LIM = LIM_coefficients.coeff2_LIM

        self._xiR0_LIM = c2_LIM**2 * nonlinearcorrelation * np.array([c1_LIM]).T**2 
        
        

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TO CORRECT
        if (constants.FLAG_DO_DENS_NL) and Line_Parameters._R < constants.MAX_R_NONLINEAR:

            D_coeffR1LIM = coeffR1_LIM.reshape(*coeffR1_LIM.shape, 1)
            D_gammaR1 = gammaR1.reshape(*gammaR1.shape , 1)
            D_growthRmatrix = growthRmatrix[:,:1].reshape(*growthRmatrix[:,:1].shape, 1)
            D_corrdNL = corrdNL[:1,0,:,:]

            # SarahLibanore: TO CHECK
            D_g1NL = gammaR1NL.reshape(*gammaR1NL.shape,1)

            if Cosmo_Parameters.second_order_SFRD:
                D_numerator = D_gammaR1 * D_growthRmatrix * D_corrdNL + D_gammaR1**2 / 2.
                D_denominator = 1. - 2 * D_g1NL + 4 * g1NL * (1 - D_corrdNL * D_corrdNL)
                D_log_norm = np.log(np.sqrt(D_denominator) * norm1)

                self._deltaxi_dLIM = np.sum(D_coeffR1LIM * ((np.exp(D_numerator/D_denominator - D_log_norm)-1.0 ) - D_gammaR1 * D_growthRmatrix * D_corrdNL), axis = 1)
            else:
                self._deltaxi_dLIM = np.sum(D_coeffR1LIM * ((np.exp(D_gammaR1 * D_growthRmatrix * D_corrdNL )-1.0 ) - D_gammaR1 * D_growthRmatrix * D_corrdNL), axis = 1)

            self._deltaxi_dLIM *= np.array([coeffzp1_LIM]).T
            

        '''
        # I don't need the linear part
        # xi_lognormal is the correlation xi^R0R0(r,z) = FT(P(k)W^R0(k))^2) 
        # np._ix indexes its first and second dimension i.e. the R0 one!
        # hence for us it just becomes a flag
        if Line_Parameters._R < constants.MAX_R_NONLINEAR:
            
            _iRnonlinear = [-1] 
            corrdNL = Correlations.xi_LIMLIM_CF[np.ix_(_iRnonlinear,_iRnonlinear)]
            #for R<RNL fix at RNL, avoids corelations blowing up at low R
            if Line_Parameters._R < constants.MIN_R_NONLINEAR:
                print('Your resolution introduces too large non linear corrections on small scales! ')
                print('It should have been changed in the input file, check why this did not happened\n')

            corrdNL = corrdNL.reshape((1, *corrdNL.shape))
        else:
            _iRnonlinear = [-1] 
            print('The resolution allows for fully linear calculation')
        _iRnonlinear = [-1]
        growthRmatrix = cosmology.growth(Cosmo_Parameters,zGreaterMatrix100[:, _iRnonlinear])
        growthRmatrix1 = growthRmatrix.reshape(len(LIM_coefficients.zintegral), 1, len(_iRnonlinear),1)
        growthRmatrix2 = growthRmatrix.reshape(len(LIM_coefficients.zintegral), len(_iRnonlinear), 1,1)
        growth_corr = growthRmatrix1 * growthRmatrix2

        gammaR1 = LIM_coefficients.gammaLIM_index[:, _iRnonlinear] * growthRmatrix
        sigmaR1 = LIM_coefficients.sigmaofRtab_LIM[:, _iRnonlinear] 
        sR1 = (sigmaR1).reshape(len(LIM_coefficients.zintegral), 1, len(_iRnonlinear),1)
        sR2 = (sigmaR1).reshape(len(LIM_coefficients.zintegral), len(_iRnonlinear), 1,1)

        coeffzp1_LIM = LIM_coefficients.coeff1_LIM
        coeffR1_LIM = LIM_coefficients.coeff2_LIM[:,_iRnonlinear]

        coeffmatrix_LIM = coeffR1_LIM.reshape(len(LIM_coefficients.zintegral), 1, len(_iRnonlinear),1) * coeffR1_LIM.reshape(len(LIM_coefficients.zintegral), len(_iRnonlinear), 1,1)

        if Line_Parameters._R < constants.MAX_R_NONLINEAR:

            g1 = (gammaR1 * sigmaR1).reshape(len(LIM_coefficients.zintegral), 1, len(_iRnonlinear),1)
            g2 = (gammaR1 * sigmaR1).reshape(len(LIM_coefficients.zintegral), len(_iRnonlinear), 1,1)
            gammamatrixR1R1 = g1 * g2

            corrdNL_gs = ne.evaluate('corrdNL * growth_corr/ (sR1 * sR2)')
            gammaTimesCorrdNL = ne.evaluate('gammamatrixR1R1 * corrdNL_gs')

            if Astro_Parameters.second_order_SFRD:

                gammaR1NL = LIM_coefficients.gamma2_LIM_index[:, _iRnonlinear] 
                g1NL = (gammaR1NL * sigmaR1**2).reshape(len(LIM_coefficients.zintegral), 1, len(_iRnonlinear),1)
                g2NL = (gammaR1NL * sigmaR1**2).reshape(len(LIM_coefficients.zintegral), len(_iRnonlinear), 1,1)

                numerator_NL = ne.evaluate('gammaTimesCorrdNL+ g1 * g1 * (0.5 - g2NL * (1 - corrdNL_gs * corrdNL_gs)) + g2 * g2 * (0.5 - g1NL * (1 - corrdNL_gs * corrdNL_gs))')
                
                denominator_NL = ne.evaluate('1. - 2 * g1NL - 2 * g2NL + 4 * g1NL * g2NL * (1 - corrdNL_gs * corrdNL_gs)')
                
                norm1 = ne.evaluate('exp(g1 * g1 / (2 - 4 * g1NL)) / sqrt(1 - 2 * g1NL)') 
                norm2 = ne.evaluate('exp(g2 * g2 / (2 - 4 * g2NL)) / sqrt(1 - 2 * g2NL)') 
                
                log_norm = ne.evaluate('log(sqrt(denominator_NL) * norm1 * norm2)')
                nonlinearcorrelation = ne.evaluate('exp(numerator_NL/denominator_NL - log_norm)')

                # use second order in SFRD lognormal approx
                expGammaCorrMinusLinear = ne.evaluate('nonlinearcorrelation - 1-gammaTimesCorrdNL')
            else:

                expGammaCorrMinusLinear = ne.evaluate('exp(gammaTimesCorrdNL) - 1 - gammaTimesCorrdNL')

            self._deltaxi_LIM = np.einsum('ijkl->il', coeffmatrix_LIM * expGammaCorrMinusLinear, optimize = True)
            self._deltaxi_LIM *= np.array([coeffzp1_LIM]).T**2 #brings it to Inu units

            self._deltaxi_LIM = np.einsum('ijkl->il', coeffmatrix_LIM * expGammaCorrMinusLinear, optimize = True)
            self._deltaxi_LIM *= np.array([coeffzp1_LIM]).T**2 #brings it to Inu units
                            
        '''

        return 1


    # same from correlations.py 
    def get_Pk_from_xi(self, rsinput, xiinput):
        "Generic Fourier Transform, returns Pk from an input Corr Func xi. kPf should be the same as _klistCF"

        kPf, Pf = mcfit.xi2P(rsinput, l=0, lowring=True)(xiinput, extrap=False)

        return kPf, Pf

    # same from correlations.py 
    def get_list_PS(self, xi_list, zlisttoconvert):
        "Returns the power spectrum given a list of CFs (xi_list) evaluated at z=zlisttoconvert as input"

        _Pk_list = []
        for izp,zp in enumerate(zlisttoconvert):
            _kzp, _Pkzp = self.get_Pk_from_xi(self._r_CF,xi_list[izp])
            _Pk_list.append(_Pkzp)

        return np.array(_Pk_list)


    def exponentialcorrelation_linear(self, gamma1, gamma2, sigmaR1, sigmaR2, xi12):

        #note that it's in units of sigmaR (gamma 1 and 2 are for deltaR1 and deltaR2, gammaNL is the second-order correction)

        g1, g2 = gamma1*sigmaR1, gamma2*sigmaR2

        xi = xi12/(sigmaR1 * sigmaR2)# dimless, -1 to 1. 

        numerator = g1*g2 * xi 

        return np.exp(numerator)-1.0


    def exponentialcorrelation_quad(self, gamma1,gamma1NL, gamma2, gamma2NL, sigmaR1, sigmaR2, xi12):

        #note that it's in units of sigmaR (gamma 1 and 2 are for deltaR1 and deltaR2, gammaNL is the second-order correction)

        g1, g1NL = gamma1*sigmaR1, gamma1NL*sigmaR1**2

        g2, g2NL = gamma2*sigmaR2, gamma2NL*sigmaR2**2

        xi = xi12/(sigmaR1 * sigmaR2)# dimless, -1 to 1. 


        denominator = 1 - 2*g1NL - 2 * g2NL + 4*g1NL*g2NL*(1-xi**2) 

        numerator = g1*g2 * xi + g1**2*(0.5-g2NL*(1-xi**2)) + g2**2*(0.5-g1NL*(1-xi**2))


        norm1 = np.exp(g1**2/(2.0-4.0*g1NL))/np.sqrt(1.0-2.0*g1NL)

        norm2 = np.exp(g2**2/(2.0-4.0*g2NL))/np.sqrt(1.0-2.0*g2NL)

        return np.exp(numerator/denominator)/np.sqrt(denominator)/norm1/norm2-1.0


# class Power_Spectra_cross() ???