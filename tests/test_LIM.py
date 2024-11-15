"""

Test the LIM part for Zeus21 

Author: Sarah Libanore
BGU - November 2024

"""

import zeus21
from zeus21.LIM import * 

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import pytest

# some inputs are different from the defaults
line_inputs = ['CII',3.,'Lagache18',1.897e12,1.4,-0.07,7.1,-0.07,0.5]

LineParams_input = zeus21.LineParams_Input(LINE= line_inputs[0], _R = line_inputs[1], CII_MODEL = line_inputs[2], CII_nu_rest = line_inputs[3], CII_alpha_SFR_0 = line_inputs[4], CII_alpha_SFR_z = line_inputs[5], CII_beta_SFR_0 = line_inputs[6], CII_beta_SFR_z = line_inputs[7], CII_sigma_LSFR = line_inputs[8])

LineParams = zeus21.Line_Parameters(LineParams_input)

def test_LIM_inputs():

    #make sure all the input parameters are the same as we use throughout
    assert(LineParams.LINE == LineParams_input.LINE)
    assert(LineParams._R == LineParams_input._R and LineParams._R != zeus21.LineParams_Input()._R)
    assert(LineParams.CII_MODEL == LineParams_input.CII_MODEL)
    assert(LineParams.CII_nu_rest == LineParams_input.CII_nu_rest)
    assert(LineParams.CII_alpha_SFR_0 == LineParams_input.CII_alpha_SFR_0)
    assert(LineParams.CII_alpha_SFR_z == LineParams_input.CII_alpha_SFR_z)
    assert(LineParams.CII_beta_SFR_0 == LineParams_input.CII_beta_SFR_0)
    assert(LineParams.CII_beta_SFR_z == LineParams_input.CII_beta_SFR_z)
    assert(LineParams.CII_sigma_LSFR == LineParams_input.CII_sigma_LSFR)

    #make sure the resolution chosen is reasonable.
    assert(LineParams._R >= 0.5) # larger than the minimum value chosen in the 21cm computation? or larger than the NL case?


def test_LIM_functions(with_plots = False):

    CosmoParams_input = zeus21.Cosmo_Parameters_Input(kmax_CLASS = 100.) #to speed up a little
    ClassyCosmo = zeus21.runclass(CosmoParams_input)
    CosmoParams = zeus21.Cosmo_Parameters(CosmoParams_input, ClassyCosmo)
    HMFintclass = zeus21.HMF_interpolator(CosmoParams,ClassyCosmo)

    AstroParams = zeus21.Astro_Parameters(CosmoParams)
    ZMIN = 20.0 #down to which z we compute the evolution

    Coeffs_LIM = zeus21.get_LIM_coefficients(CosmoParams, AstroParams, HMFintclass, LineParams, zmin=ZMIN)

    ztest = 10.
    iztest = min(range(len(Coeffs_LIM.zintegral)), key=lambda i: np.abs(Coeffs_LIM.zintegral[i]-ztest))


    # test line luminosity first
    RHOL = lambda z: LineLuminosity( LineParams, AstroParams, CosmoParams, HMFintclass, HMFintclass.Mhtab, z)

    if with_plots:
        ztest_arr = np.linspace(ztest,SZ.constants.ZMAX_INTEGRAL,10)
        for i in ztest_arr:
            use_rho = RHOL(i)
            plt.loglog(HMFintclass.Mhtab,use_rho)
            plt.xlabel('Mh')
            plt.ylabel('L_{%s}'%LineParams.LINE)
            plt.title('z=%g'%i)
            plt.show()

    sRHOL = RHOL(ztest)
    assert( (0 <= sRHOL).all()) #positive

    # test LIM calculation
    assert( (Coeffs_LIM.ztabRsmoo[iztest] >= Coeffs_LIM.zintegral[iztest]).all())
    assert( (Coeffs_LIM.sigmaofRtab_LIM >= 0.0).all()) 

    assert( (Coeffs_LIM.rhoL_avg >= 0.0).all()) 
    assert( (Coeffs_LIM.rhoLbar >= 0.0).all()) 
    assert( (Coeffs_LIM.Inu_avg >= 0.0).all()) 

    assert( (Coeffs_LIM.gammaLIM_index >= 0.0).all()) #effective biases have to be larger than 0 in reasonable models, since galaxies live in haloes that are more clustered than average matter (in other words, SFRD and Inu grow monotonically with density)

    if with_plots:

        plt.semilogy(Coeffs_LIM.zintegral,Coeffs_LIM.Inu_avg)
        plt.xlabel('z')
        plt.ylabel('Inu [Lsun / (cm^2 Hz sr)]')

        plt.show()


#and test the PS too
def test_LIM_PK(with_plots = False):

    CosmoParams_input = zeus21.Cosmo_Parameters_Input(kmax_CLASS = 100.) #to speed up a little
    ClassyCosmo = zeus21.runclass(CosmoParams_input)
    CosmoParams = zeus21.Cosmo_Parameters(CosmoParams_input, ClassyCosmo)
    HMFintclass = zeus21.HMF_interpolator(CosmoParams,ClassyCosmo)

    AstroParams = zeus21.Astro_Parameters(CosmoParams)
    ZMIN = 20.0 #down to which z we compute the evolution
    
    CorrF_LIM = zeus21.Correlations_LIM(LineParams, CosmoParams, ClassyCosmo)

    Coeffs_LIM = zeus21.get_LIM_coefficients(CosmoParams, AstroParams, HMFintclass, LineParams, zmin=ZMIN)

    Coeffs_T21 = zeus21.get_T21_coefficients(CosmoParams, ClassyCosmo, AstroParams, HMFintclass, zmin=ZMIN)

    PS21 = zeus21.Power_Spectra_LIM(CosmoParams, AstroParams, LineParams, CorrF_LIM, Coeffs_T21, Coeffs_LIM)

    assert((PS21._rs_input_mcfit == CorrF_LIM.rlist_CF).all())
    assert((PS21.klist_PS == CorrF_LIM._klistCF).all())

    ztest = 20.
    iztest = min(range(len(Coeffs_T21.zintegral)), key=lambda i: np.abs(Coeffs_T21.zintegral[i]-ztest))

    assert((PS21.window_LIM[iztest,0] >= PS21.window_LIM[iztest,-1]).all()) #at fixed z it should go down with k
    
    # #make sure all cross correlations are sensible
    #assert( (PS21.Deltasq_dxa[iztest]**2 <= 1.01* PS21.Deltasq_d[iztest] * PS21.Deltasq_xa[iztest]).all())
  
    # #also make sure all Pk(k) < avg^2 for all quantities at some k~0.1
    ktest = 0.1
    iktest = min(range(len(PS21.klist_PS)), key=lambda i: np.abs(PS21.klist_PS[i]-ktest))

    assert( (PS21.Deltasq_LIM[:,iktest] <= 1.01*Coeffs_LIM.rhoL_avg**2 ).all())
    
    assert(CorrF_LIM.xi_RLIM_CF[0][0][1] >= CorrF_LIM.xi_RLIM_CF[1][1][1]) #make sure smoothing goes the right direction
    assert(CorrF_LIM.xi_LIMLIM_CF[0][0][1] >= CorrF_LIM.xi_LIMLIM_CF[1][1][1]) #make sure smoothing goes the right direction

    #windows
    ktestwin = 1e-4
    Rtestwin = 1.0
    assert(CorrF_LIM._WinG(ktestwin,Rtestwin) == pytest.approx(1.0, 0.01))
    assert(CorrF_LIM._WinTH(ktestwin,Rtestwin) == pytest.approx(1.0, 0.01))
    assert(CorrF_LIM._WinTH1D(ktestwin,Rtestwin) == pytest.approx(1.0, 0.01))

    ktestwin = 3.
    assert(CorrF_LIM._WinG(ktestwin,Rtestwin) < 1.0)
    assert(CorrF_LIM._WinTH(ktestwin,Rtestwin) < 1.0)
    assert(CorrF_LIM._WinTH1D(ktestwin,Rtestwin) < 1.0)