'''
Introduce star forming lines in Zeus to compute cross correlations with 21cm.

Author: Sarah Libanore
BGU - October 2024
'''

# we want to be consistent with the functions defined in the sfrd.py module of the original Zeus code
from . import sfrd as SZ 
import numpy as np 
import astropy.units as au
import astropy.constants as ac

class get_LIM_coefficients:
    "Loops through rhoL integrals and obtains avg Inu and the coefficients for its power spectrum. Takes input zmin, which minimum z we integrate down to. "

    def __init__(self, Cosmo_Parameters, Astro_Parameters, HMF_interpolator, Line_Parameters, zmin = 10.0): # zmin should be decreased!!! 

        ######################################################################################
        ### STEP 0: Defining Constants and storage variables

        #define comoving distance quantities
        # !!! SL: for us, Rtabsmoo --> Rtabsmoo_LIM is a scalar, given as input 
        # !!! SL: it should work also in case you want to use an array (e.g. redshift dependent)
        self.Rtabsmoo_LIM = Line_Parameters._R
        
        # ----- same as sfrd.py
        #define the integration redshifts, goes as log(z) (1+ doesn't change sampling much)
        self.zmax_integral = SZ.constants.ZMAX_INTEGRAL
        self.zmin = zmin 
        self._dlogzint_target = 0.02/SZ.constants.precisionboost
        self.Nzintegral = np.ceil(1.0 + np.log(self.zmax_integral/self.zmin)/self._dlogzint_target).astype(int)
        self.dlogzint = np.log(self.zmax_integral/self.zmin)/(self.Nzintegral-1.0) #exact value rather than input target above
        self.zintegral = np.logspace(np.log10(self.zmin), np.log10(self.zmax_integral), self.Nzintegral) #note these are also the z at which we "observe", to share computational load
        
        # !!! SL: here we only use 1 value for R, so the matrix is Nz x 1 !!! 
        # !!! SL: we keep the matrix form to still allow a R array ---> if resolution based on instrument, angular res is fixed but gets converted

        #define table of redshifts and distances
        # SL: since LIM is local we do not need to transform z[\chi(z) + R]
        self.rGreaterMatrix = np.transpose([Cosmo_Parameters.chiofzint(self.zintegral)]) 
        self.zGreaterMatrix = Cosmo_Parameters.zfofRint(self.rGreaterMatrix)
        
        self.ztabRsmoo = np.nan_to_num(np.copy(self.zGreaterMatrix), nan = 100)
        if(Cosmo_Parameters.Flag_emulate_21cmfast == True): 
        
            self.zGreaterMatrix = np.append(self.zintegral.reshape(len(self.zGreaterMatrix), 1), self.zGreaterMatrix, axis = 1)

            self.zGreaterMatrix = (self.zGreaterMatrix[:, 1:] + self.zGreaterMatrix[:, :-1])/2
            
            self.rGreaterMatrix[self.rGreaterMatrix > Cosmo_Parameters.chiofzint(50.0)] = Cosmo_Parameters.chiofzint(50.0)
            
            self.ztabRsmoo = np.append(self.zintegral.reshape(len(self.ztabRsmoo), 1), self.ztabRsmoo, axis = 1)
            self.ztabRsmoo = (self.ztabRsmoo[:, 1:] + self.ztabRsmoo[:, :-1])/2
        else:
            self.zGreaterMatrix[self.rGreaterMatrix > Cosmo_Parameters.chiofzint(50.0)] = np.nan
            self.rGreaterMatrix[self.rGreaterMatrix > Cosmo_Parameters.chiofzint(50.0)] = np.nan #replace z > 50 = np.nan so that nothing exceeds zmax = 50
            self.ztabRsmoo = np.nan_to_num(np.copy(self.zGreaterMatrix), nan = 100)
            
        self.sigmaofRtab_LIM = np.array([HMF_interpolator.sigmaR_int(self.Rtabsmoo_LIM, zz) for zz in self.zintegral]) # sigma_R(z) used for non linear correction

        # EPS factors 
        Nsigmad = 1.0 #how many sigmas we explore
        Nds = 2 #how many deltas
        deltatab_norm = np.linspace(-Nsigmad,Nsigmad,Nds)

        # -----

        # line luminosity variable
        self.rhoL_avg = np.zeros_like(self.zintegral)
        # !!! SL: this is 1D and not 2D as in the SFRD case because LIM is local and we smooth over single R
        # !!! we keep the matrix shape in case we want to have more R
        self.rhoLbar = np.zeros((self.Nzintegral,1))

        # index of rhoL ~ exp(\gamma delta) 
        # this is 1D and not 2D as in the SFRD case because LIM is local and we average over a single R
        self.gammaLIM_index = np.zeros_like(self.rhoLbar)

        ######################################################################################
        ### STEP 1: Recursive routine to compute average rhoL

        zLIMflat = np.geomspace(self.zmin, 50, 128) #extend to z = 50 for extrapolation purposes. Higher in z than self.zintegral  
        zLIM, mArray_LIM = np.meshgrid(zLIMflat, HMF_interpolator.Mhtab, indexing = 'ij', sparse = True)

        rhoL_avg = np.trapz(rhoL_integrand(Line_Parameters,Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray_LIM, zLIM), HMF_interpolator.logtabMh, axis = 1) 
        rhoL_interp = SZ.interpolate.interp1d(zLIMflat, rhoL_avg, kind = 'cubic', bounds_error = False, fill_value = 0,) 

        self.rhoL_avg = rhoL_interp(self.zintegral)

        if(Cosmo_Parameters.Flag_emulate_21cmfast==False):
            self.rhoLbar = rhoL_interp(np.nan_to_num(self.zGreaterMatrix, nan = 100))
            
        elif(Cosmo_Parameters.Flag_emulate_21cmfast==True): 
            mTable_LIM = np.meshgrid(self.zintegral, self.Rtabsmoo_LIM, HMF_interpolator.Mhtab, indexing = 'ij', sparse = True)[-1]
            # !!! SL: note that here the 1 in the axis = 1 direction is the R size. If you change Rtabsmoo_LIM you should change it to len(Rtabsmoo)
            zppTable_LIM = self.zGreaterMatrix.reshape((len(self.zintegral), 1, 1))

            self.rhoLbar = np.trapz(rhoL_integrand(Line_Parameters, Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mTable_LIM, zppTable_LIM), HMF_interpolator.logtabMh, axis = 2)

            self.rhoLbar[np.isnan(self.rhoLbar)] = 0.0


        ######################################################################################
        ### STEP 2: Broadcasted Prescription to Compute gammas
        # ----- same as sfrd.py

        # !!! SL: note that here R has dimension 1
        zArray_LIM, rArray_LIM, mArray_LIM, deltaNormArray_LIM = np.meshgrid(self.zintegral, self.Rtabsmoo_LIM, HMF_interpolator.Mhtab, deltatab_norm, indexing = 'ij', sparse = True)

        rGreaterArray = np.zeros_like(zArray_LIM) + rArray_LIM

        # !!! SL: we keep R dimension in case we want to have more values e.g. z dependnet
        rGreaterArray[Cosmo_Parameters.chiofzint(zArray_LIM) + rArray_LIM >= Cosmo_Parameters.chiofzint(50)] = np.nan

        zGreaterArray = Cosmo_Parameters.zfofRint(Cosmo_Parameters.chiofzint(zArray_LIM) )#+ rGreaterArray) 

        whereNotNans = np.invert(np.isnan(rGreaterArray))

        # -----

        # !!! SL: here we want only 1 R value , which is the 1 in axis = 1
        # modify to len(Rtabsmoo) if you are using array 
        sigmaR_LIM = np.zeros((len(self.zintegral), 1, 1, 1)) 
        sigmaR_LIM[whereNotNans] = HMF_interpolator.sigmaRintlog((np.log(rGreaterArray)[whereNotNans], zGreaterArray[whereNotNans])) 

        # !!! SL: here we want only 1 R value , which is the 1 in axis = 1
        # modify to len(Rtabsmoo) if you are using array 
        sigmaM_LIM = np.zeros((len(self.zintegral), 1, len(HMF_interpolator.Mhtab), 1)) 
        sigmaM_LIM = HMF_interpolator.sigmaintlog((np.log(mArray_LIM), zGreaterArray)) 

        modSigmaSq_LIM = sigmaM_LIM**2 - sigmaR_LIM**2
        indexTooBig = (modSigmaSq_LIM <= 0.0)
        modSigmaSq_LIM[indexTooBig] = np.inf #if sigmaR > sigmaM the halo does not fit in the radius R. Cut the sum
        modSigmaSq_LIM = np.sqrt(modSigmaSq_LIM)

        nu0 = Cosmo_Parameters.delta_crit_ST / sigmaM_LIM # this is needed in the HMF 
        nu0[indexTooBig] = 1.0

        dsigmadMcurr_LIM = HMF_interpolator.dsigmadMintlog((np.log(mArray_LIM),zGreaterArray))  
        dlogSdMcurr_LIM = (dsigmadMcurr_LIM*sigmaM_LIM*2.0)/(modSigmaSq_LIM)

        deltaArray_LIM = deltaNormArray_LIM * sigmaR_LIM

        modd_LIM = Cosmo_Parameters.delta_crit_ST - deltaArray_LIM
        nu = modd_LIM / modSigmaSq_LIM # used in the HMF

        #PS_HMF~ delta/sigma^3 *exp(-delta^2/2sigma^2) * consts(of M including dsigma^2/dm)
        if(Cosmo_Parameters.Flag_emulate_21cmfast==False):
        #Normalized PS(d)/<PS(d)> at each mass. 21cmFAST instead integrates it and does SFRD(d)/<SFRD(d)>
        # last 1+delta product converts from Lagrangian to Eulerian
            EPS_HMF_corr = (nu/nu0) * (sigmaM_LIM/modSigmaSq_LIM)**2.0 * np.exp(-Cosmo_Parameters.a_corr_EPS * (nu**2-nu0**2)/2.0 ) * (1.0 + deltaArray_LIM)

            integrand_LIM = EPS_HMF_corr * rhoL_integrand(Line_Parameters,Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray_LIM, zGreaterArray)
            
        elif(Cosmo_Parameters.Flag_emulate_21cmfast==True): #as 21cmFAST, use PS HMF, integrate and normalize at the end

            PS_HMF_corr = SZ.cosmology.PS_HMF_unnorm(Cosmo_Parameters, HMF_interpolator.Mhtab.reshape(len(HMF_interpolator.Mhtab),1),nu,dlogSdMcurr_LIM) * (1.0 + deltaArray_LIM)

            SFR = SZ.SFR_II(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray_LIM, zGreaterArray, 0.)

            integrand_LIM = PS_HMF_corr * LineLuminosity(SFR, Line_Parameters, Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray_LIM, zGreaterArray) * mArray_LIM
            
        else:
            print("ERROR: Need to set FLAG_EMULATE_21CMFAST at True or False in the self.gammaLIM_index calculation.")

        ########
        # Compute LIM quantities
        self.rhoL_dR = np.trapz(integrand_LIM, HMF_interpolator.logtabMh, axis = 2)

        #compute gammas 
        # !!! note that it is still 2D but the dimension in R is 1 
        self.gammaLIM_index = np.log(self.rhoL_dR[:,:,-1]/self.rhoL_dR[:,:,0]) / (deltaArray_LIM[:,:,0,-1] - deltaArray_LIM[:,:,0,0])
        self.gammaLIM_index[np.isnan(self.gammaLIM_index)] = 0.0

        # ?? no velocity anisotropies and VCB effect ??
        # ?? no LW correection ?? 
                       
        ######################################################################################
        ### STEP 3: Line Intensity Anisotropies
        
        #2D cube, dimensions are (z,R) = (64, 1)

        if Line_Parameters.LINE == 'CO':
            nu_line_rest = -1
        elif Line_Parameters.LINE == 'CII':
            nu_line_rest = Line_Parameters.CII_nu_rest
        else:
            nu_line_rest = -1
            
        if Line_Parameters.OBSERVABLE_LIM == 'Tnu':

            # c1 = uK / Lsun * Mpc^3
            self.coeff1_LIM = (((SZ.constants.c_kms * au.km/au.s)**3 * (1+self.zintegral)**2 / (8*np.pi * (SZ.cosmology.Hub(Cosmo_Parameters, self.zintegral) * au.km/au.s/au.Mpc) * (nu_line_rest * au.Hz)**3 * ac.k_B)).to(au.uK * au.Mpc**3 / au.Lsun )).value

            # c2 = Lbar = Lsun / Mpc^3 
            self.coeff2_LIM = self.rhoLbar

            # --> c1*c2 = uK

        elif Line_Parameters.OBSERVABLE_LIM == 'Inu':

            # nu_rest for CII is in Hz, speed of light in km / s , Hubble in km / s / Mpc --> c1 = cm / sr / Hz
            self.coeff1_LIM = ((SZ.constants.c_kms * au.km/au.s) / (4*np.pi * (SZ.cosmology.Hub(Cosmo_Parameters, self.zintegral) * au.km/au.s/au.Mpc) * nu_line_rest * au.Hz)) * SZ.constants.Mpctocm

            # c2 = Lbar = Lsun / Mpc^3 
            self.coeff2_LIM = self.rhoLbar / SZ.constants.Mpctocm**3

            # --> c1*c2 = Lsun / (cm^2 * sr * Hz) 

        else:
            print('Check Observable for LIM!')
            self.coeff1_LIM = -1
            self.coeff2_LIM = -1


        ######################################################################################
        ### STEP 4: Non-Linear Correction Factors
        #correct for nonlinearities in <(1+d)SFRD>, only if doing nonlinear stuff. We're assuming that (1+d)SFRD ~ exp(gamma*d), so the "Lagrangian" gamma was gamma-1. We're using the fact that for a lognormal variable X = log(Z), with  Z=\gamma \delta, <X> = exp(\gamma^2 \sigma^2/2).

        if(SZ.constants.C2_RENORMALIZATION_FLAG==True):

            if self.Rtabsmoo_LIM >= SZ.constants.MIN_R_NONLINEAR and self.Rtabsmoo_LIM < SZ.constants.MAX_R_NONLINEAR:
                _corrfactorEulerian_LIM = 1.0 + (self.gammaLIM_index-1.0)*self.sigmaofRtab_LIM**2

                self.coeff2_LIM *= _corrfactorEulerian_LIM


        ######################################################################################
        ### STEP 5: Compute the AVERAGE line intensity

        self.Inu_avg = self.coeff1_LIM * self.coeff2_LIM.T[0] # either in units uK or in units Lsun / (cm^2 Hz sr)

    # ?? not needed : electron fraction, reionization , T21 signal ??


# ------------------- here we define the functions needed in the module
    
def rhoL_integrand(Line_Parameters, Astro_Parameters, Cosmo_Parameters, HMF_interpolator, massVector, z):
    "AVERAGE line luminosity density, modelled analogous to the SFRD. Line Parameters is a dictionary that contains information on the line that one wants to model"

    Mh = massVector # in Msun

    SFR = SZ.SFR_II(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, massVector, z, 0.)    

    HMF_curr = np.exp(HMF_interpolator.logHMFint((np.log(Mh), z))) # in Mpc-3 Msun-1 

    Ltab_curr = LineLuminosity(SFR, Line_Parameters, Astro_Parameters, Cosmo_Parameters, HMF_interpolator, Mh, z) 

    integrand_LIM = HMF_curr * Ltab_curr * Mh # in Lsun / Mpc3 

    return integrand_LIM


def LineLuminosity(SFR, Line_Parameters, Astro_Parameters, Cosmo_Parameters, HMF_interpolator, massVector, z):
    "Luminosity for the different lines. Line Parameters is a dictionary that contains information on the line that one wants to model. Units: solar luminosity Lsun"
    
    if SFR is False:
        SFR = SZ.SFR_II(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, massVector, z, 0.)    
        
    # TO BE PROPERLY MODELLED
    if Line_Parameters.LINE == 'CO':
        output = -1

    elif Line_Parameters.LINE == 'CII':

        if Line_Parameters.CII_MODEL == 'Lagache18': 

            alpha_SFR = Line_Parameters.CII_alpha_SFR_0 + Line_Parameters.CII_alpha_SFR_z * z

            beta_SFR = Line_Parameters.CII_beta_SFR_0 + Line_Parameters.CII_beta_SFR_z * z

            try:
                alpha_SFR[alpha_SFR < 0.] = 0. 
            except:
                if alpha_SFR < 0.:
                    alpha_SFR = 0.

            log10_L = alpha_SFR * np.log10(SFR) + beta_SFR     

            # here you can account for stochasticity in the luminosity - SFR relation (on the average, not introducing fluctuations)
            # STILL DEBUGGING
            if Line_Parameters.CII_sigma_LSFR == 0.:
                output = 10.**log10_L
            else:
                
                mu_L = 10.**log10_L
                mu_L[abs(log10_L) == np.inf] = 0.
                
                sigma_L = Line_Parameters.CII_sigma_LSFR 

                Lval = np.logspace(-5,15,203)

                coef = 1/(np.sqrt(2*np.pi)*sigma_L)

                if len(mu_L.shape) == 2:

                    p_logL =  coef * np.exp(- (np.log10(Lval[:,np.newaxis,np.newaxis])-np.log10(mu_L[np.newaxis,:]))**2/(2*(sigma_L)**2))

                    p_logL = np.where(np.isnan(p_logL), 0, p_logL)

                    p_logL[p_logL < 1e-30] = 0.

                    output = SZ.simpson(p_logL / np.log(10) , Lval,axis=0)

                elif len(mu_L.shape) == 4:
                    p_logL = coef *  np.exp(- (np.log10(Lval[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis])-np.log10(mu_L[np.newaxis,:]))**2/(2*(sigma_L)**2))
                    p_logL = np.where(np.isnan(p_logL), 0, p_logL)
                    p_logL[p_logL < 1e-30] = 0.

                    output = SZ.simpson(p_logL / np.log(10) , Lval,axis=0)
               
    else:
        output = -1

    return output