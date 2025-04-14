from oLIMpus import * 
from powerbox import get_power
from classy import Class
import os
import pickle
from scipy.optimize import brentq
# 2D binning using scipy
from scipy.stats import binned_statistic_2d



from .fiducial_analysis import * 

folder_out = './analysis_' + str(Lbox_fid) + ',' + str(Nbox_fid) 
if not os.path.exists(folder_out):
    os.makedirs(folder_out)

def run_all_fiducials():

    P_fid = []
    k_fid = []
    r_fid = []
    s_fid = []
    xH_fid = []
    for iz in zvals:
        print('Doing ' + str(iz))
        temp = run_analysis(iz, 
                Lbox_fid, 
                Nbox_fid,
                LineParams_input_1_fid,
                AstroParams_input_fid, 
                CosmoParams_input_fid,
                LineParams2_input = LineParams_input_2_fid,
                store_quantities = True,
                include_label = ''
                )
        P_fid.append(temp[0])
        k_fid.append(temp[1])
        r_fid.append(temp[2])
        s_fid.append(temp[3])
        xH_fid.append(temp[4])

    return P_fid, k_fid, r_fid, s_fid, xH_fid


def run_variations(var_line = False, var_astro = True, var_cosmo = True):

    P_var = []
    k_var = []
    r_var = []
    s_var = []
    xH_var = []
    var_params = []

    if var_line:
        print('What parameters do you want to change in the line model?')
        return -1

    if var_astro:
        var_params_astro = ['epsstar', 'fesc']
        print('Astro params varied: ' + str(var_params_astro))

        P_var_astro = []
        k_var_astro = []
        r_var_astro = []
        s_var_astro = []
        xH_var_astro = []

        for i in range(len(var_params_astro)):
            if var_params_astro[i] == 'epsstar':
                values = values_epsstar
            elif var_params_astro[i] == 'fesc':
                values = values_fesc
            
            P_var_astro.append([])
            k_var_astro.append([])
            r_var_astro.append([])
            s_var_astro.append([])
            xH_var_astro.append([])

            for j in range(len(values)):
                include_label = var_params_astro[i] + '_' + str(round(values[j]))
                P_var_astro[i].append([])
                k_var_astro[i].append([])
                r_var_astro[i].append([])
                s_var_astro[i].append([])
                xH_var_astro[i].append([])
                if var_params_astro[i] == 'epsstar':
                    AstroParams_input_var = {'epsstar':values[j]}
                elif var_params_astro[i] == 'fesc':
                    AstroParams_input_var = {'fesc10':values[j]}
                for iz in zvals:
                    temp = run_analysis(iz, 
                            Lbox_fid, 
                            Nbox_fid,
                            LineParams_input_1_fid,
                            AstroParams_input_var, 
                            CosmoParams_input_fid,
                            LineParams_input_2_fid,
                            store_quantities = True,
                            include_label = include_label
                            )
                    P_var_astro[i][j].append(temp[0])
                    k_var_astro[i][j].append(temp[1])
                    r_var_astro[i][j].append(temp[2])
                    s_var_astro[i][j].append(temp[3])
                    xH_var_astro[i][j].append(temp[4])

        P_var.append(P_var_astro)
        k_var.append(k_var_astro)
        r_var.append(r_var_astro)
        s_var.append(s_var_astro)
        xH_var.append(s_var_astro)
        var_params.append(var_params_astro)

    if var_cosmo:
        var_params_cosmo = ['OmegaC']
        print('Cosmo params varied: ' + str(var_params_cosmo))

        P_var_cosmo = []
        k_var_cosmo = []
        r_var_cosmo = []
        s_var_cosmo = []
        xH_var_cosmo = []

        for i in range(len(var_params_cosmo)):
            if var_params_cosmo[i] == 'OmegaC':
                values = values_OmC
            
            P_var_cosmo.append([])
            k_var_cosmo.append([])
            r_var_cosmo.append([])
            s_var_cosmo.append([])
            xH_var_cosmo.append([])

            for j in range(len(values)):

                include_label = var_params_cosmo[i] + '_' + str(round(values[j]))

                P_var_cosmo[i].append([])
                k_var_cosmo[i].append([])
                r_var_cosmo[i].append([])
                s_var_cosmo[i].append([])
                xH_var_cosmo[i].append([])
                if var_params_cosmo[i] == 'OmegaC':
                    CosmoParams_input_var = {'omegac':values[j]}
                for iz in zvals:
                    temp = run_analysis(iz, 
                            Lbox_fid, 
                            Nbox_fid,
                            LineParams_input_1_fid,
                            AstroParams_input_fid, 
                            CosmoParams_input_var,
                            LineParams_input_2_fid,
                            store_quantities = True,
                            include_label = include_label
                            )
                    P_var_cosmo[i][j].append(temp[0])
                    k_var_cosmo[i][j].append(temp[1])
                    r_var_cosmo[i][j].append(temp[2])
                    s_var_cosmo[i][j].append(temp[3])
                    xH_var_cosmo[i][j].append(temp[4])

        P_var.append(P_var_cosmo)
        k_var.append(k_var_cosmo)
        r_var.append(r_var_cosmo)
        s_var.append(s_var_cosmo)
        xH_var.append(xH_var_cosmo)
        var_params.append(var_params_cosmo)

    return P_var, k_var, r_var, s_var, xH_var, var_params



def run_analysis(z, 
             Lbox, 
             Nbox,
            LineParams_input,
            AstroParams_input, 
            CosmoParams_input,
            LineParams2_input = False,
            foregrounds = False, 
            store_quantities = True,
            include_label = ''
            ):

    CosmoParams_input = Cosmo_Parameters_Input(**CosmoParams_input)
    LineParams_input = LineParams_Input(**LineParams_input) 
    LineParams = Line_Parameters(LineParams_input) 
    if LineParams2_input:
        LineParams2_input = LineParams_Input(**LineParams2_input) 
        LineParams2 = Line_Parameters(LineParams2_input) 

    folder = folder_out + '/z' + str(round(z,1))
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    filename_all = folder + '/slices_' + include_label + str(islice) + '.pkl'
    filename_xH = folder_out + '/xHavg' + include_label + '.dat'
    if os.path.isfile(filename_all) and os.path.isfile(filename_xH):
        with open(filename_all, 'rb') as handle:
            temp =  pickle.load(handle)
            slice_T21 = temp['T21']
            slice_LIM = temp[LineParams_input.LINE]
            if LineParams2_input:
                slice_LIM2 = temp[LineParams2_input.LINE]
        T21_box = False
        LIM_box = False
        if LineParams2_input:
            LIM_box2 = False

        data_reionization = np.loadtxt(filename_xH)
        try:
            z_vals = data_reionization[:, 0]
            xH_vals = data_reionization[:, 1]
            # Find index where z == z0 (within a small tolerance to avoid float issues)
            idx = np.where(np.isclose(z_vals, z, atol=1e-8))[0]
            
            if idx.size == 0:
                raise ValueError(f"z = {z} not found in {filename_xH}") 
            xH_avg = xH_vals[idx[0]]

        except:
            xH_avg = data_reionization[1]


    else:
        # 1) setup the run 
        ClassCosmo = Class()
        ClassCosmo.compute()
        
        ClassyCosmo = runclass(CosmoParams_input) 
        CosmoParams = Cosmo_Parameters(CosmoParams_input, ClassyCosmo) 
        HMFintclass = HMF_interpolator(CosmoParams,ClassyCosmo)
        AstroParams = Astro_Parameters(CosmoParams,**AstroParams_input)

        # 2) compute zeus21 quantities
        corr_21 = Correlations(CosmoParams, ClassyCosmo)   
        coef_21 = get_T21_coefficients(CosmoParams, ClassyCosmo, AstroParams, HMFintclass, zmin=ZMIN)
        PS21 = Power_Spectra(CosmoParams, AstroParams, ClassyCosmo, corr_21, coef_21, RSD_MODE = RSDMODE)
        BMF_use = BMF(coef_21, HMFintclass,CosmoParams,AstroParams)

        boxes_zeus21 = maps.T21_bubbles(coef_21,PS21,z,Lbox,Nbox,seed,corr_21,CosmoParams,AstroParams,ClassyCosmo,BMF_use)

        # boxes of density, ionization, 21cm
        delta_box = boxes_zeus21.deltamap
        xHI_box = boxes_zeus21.xHI_map
        T21_box = boxes_zeus21.T21map

        # 3) compute oLIMpus quantities
        corr_LIM = Correlations_LIM(LineParams, CosmoParams, ClassyCosmo)
        coef_LIM = get_LIM_coefficients(CosmoParams,  AstroParams, HMFintclass, LineParams, zmin=ZMIN)
        PSLIM = Power_Spectra_LIM(CosmoParams, AstroParams, LineParams, corr_LIM, coef_LIM, RSD_MODE = RSDMODE)

        boxes_oLIMpus = CoevalBoxes_percell('full',coef_LIM,corr_LIM,PSLIM,coef_21,z,LineParams,AstroParams,HMFintclass,CosmoParams,Lbox,Nbox,seed)

        # boxes of SFRD and line intensity
        SFRD_box = boxes_oLIMpus.SFRD_box
        LIM_box = boxes_oLIMpus.Inu_box

        if LineParams2_input:
            corr_LIM2 = Correlations_LIM(LineParams2, CosmoParams, ClassyCosmo)
            coef_LIM2 = get_LIM_coefficients(CosmoParams,  AstroParams, HMFintclass, LineParams2, zmin=ZMIN)
            PSLIM2 = Power_Spectra_LIM(CosmoParams, AstroParams, LineParams2, corr_LIM2, coef_LIM2, RSD_MODE = RSDMODE)

            boxes_oLIMpus2 = CoevalBoxes_percell('full',coef_LIM2,corr_LIM2,PSLIM2,coef_21,z,LineParams2,AstroParams,HMFintclass,CosmoParams,Lbox,Nbox,seed)

            # box of intensity
            LIM_box2 = boxes_oLIMpus2.Inu_box

        # 4) get 21cm and LIM slices
        slice_T21 = T21_box[islice]
        slice_LIM = LIM_box[islice]
        if LineParams2_input:
            slice_LIM2 = LIM_box2[islice]

        xH_avg = np.mean(xHI_box)

        # 5) store quantities for animation and lightcone
        if store_quantities:

            with open(filename_all, 'wb') as handle:
                if LineParams2_input:
                    pickle.dump({'delta': delta_box[islice], 'T21': slice_T21, 'SFRD': SFRD_box[islice], \
                    'xHI': xHI_box[islice], LineParams.LINE: slice_LIM, LineParams2.LINE: slice_LIM2}, handle)
                else:
                    pickle.dump({'delta': delta_box[islice], 'T21': slice_T21, 'SFRD': SFRD_box[islice], \
                    'xHI': xHI_box[islice], LineParams.LINE: slice_LIM}, handle)

            if not os.path.exists(filename_xH):
                with open(filename_xH, 'w') as f:
                    f.write("# z xH\n")  # optional header
                # Append the new (z, xH) value
                with open(filename_xH, 'a') as f:
                    f.write(f"{z:.6e} {xH_avg:.6e}\n")

    # 6) compute correlations
    Pear = Pearson(slice_LIM, slice_T21, foregrounds)
    k, ratio_cross = ratio_pk(LIM_box, T21_box, Lbox, Nbox, LineParams.LINE, foregrounds, store_quantities, folder, include_label)
    squared_cross_val = squared_cross(LIM_box, T21_box, Lbox, Nbox, LineParams.LINE, store_quantities, folder, include_label)
    if LineParams2_input:
        Pear2 = Pearson(slice_LIM2, slice_T21, foregrounds)
        k2, ratio_cross2 = ratio_pk(LIM_box2, T21_box, Lbox, Nbox, LineParams2.LINE, foregrounds, store_quantities, folder, include_label)
        squared_cross_val2 = squared_cross(LIM_box2, T21_box, Lbox, Nbox, LineParams2.LINE, store_quantities, folder, include_label)

    if LineParams2:
        return [Pear, Pear2], [k,k2], [ratio_cross, ratio_cross2], [squared_cross_val, squared_cross_val2], [xH_avg, xH_avg]
    else:
        return Pear, k, ratio_cross, squared_cross_val, xH_avg


def Pearson(slice_LIM, slice_T21, foregrounds):

    cross_TLIM1 = np.corrcoef((slice_T21.flatten()), slice_LIM.flatten())[0, 1]
    if foregrounds:
        print('Foregrounds not yet implemented')

    return cross_TLIM1


def ratio_pk(box_LIM, box_T21, Lbox, Nbox, line, foregrounds, store_quantities, folder, include_label):

    filename_all = folder + '/powerspectra_' + include_label + line + '.pkl'

    if os.path.isfile(filename_all):
        with open(filename_all, 'rb') as handle:
            temp =  pickle.load(handle)
            k_cross = temp['k']
            pk_21 = temp['Pk_21']
            pk_LIM = temp['Pk_line']
            pk_cross = temp['Pk_cross']

    else:
        if box_LIM is False or box_T21 is False:
            print('Warning! The function ratio_pk requires boxes to run, since there are no spectra stored')
            return -1
        
        kmin = 2*np.pi/Lbox # 1/Mpc
        k_max = kmin*Nbox # 1/Mpc
        k_factor = 1.5
        k_bins_edges = []
        k_ceil = kmin
        while (k_ceil < k_max):
            k_bins_edges.append(k_ceil)
            k_ceil *= k_factor

        k_bins_edges = np.array(k_bins_edges)

        res = get_power(
            deltax = box_T21,
            boxlength=Lbox,
            bins = k_bins_edges,
            dimensionless=False,
        )
        pk_21 = list(res)[0]

        res = get_power(
            deltax = box_LIM,
            boxlength=Lbox,
            bins = k_bins_edges,
            dimensionless=False
        )
        pk_LIM = list(res)[0]

        res = get_power(
            deltax = box_T21,
            boxlength=Lbox,
            deltax2 = box_LIM,
            bins = k_bins_edges,
            dimensionless=False
        )
        pk_cross, k_cross = list(res)

        if foregrounds:
            print('Foregrounds not yet implemented')

        if store_quantities:
            with open(filename_all, 'wb') as handle:
                pickle.dump({'k': k_cross, 'Pk_21': pk_21, 'Pk_line': pk_LIM, 'Pk_cross': pk_cross}, handle)

    ratio = pk_cross / np.sqrt(pk_LIM * pk_21) 

    return k_cross, ratio


def squared_cross(box_LIM, box_T21, Lbox, Nbox, line, store_quantities, folder, include_label):

    filename_all = folder + '/powerspectra_' + include_label + line + '.pkl'

    if os.path.isfile(filename_all):
        with open(filename_all, 'rb') as handle:
            temp =  pickle.load(handle)
            k_cross = temp['k']
            pk_21 = temp['Pk_21']
            pk_LIM = temp['Pk_line']
            pk_cross = temp['Pk_cross']

    else:
        if not box_LIM or not box_T21:
            print('Warning! The function ratio_pk requires boxes to run, since there are no spectra stored')
            return -1

    kx = np.fft.fftshift(np.fft.fftfreq(Nbox, d=Lbox / Nbox)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(Nbox, d=Lbox / Nbox)) * 2 * np.pi
    kz = np.fft.fftshift(np.fft.fftfreq(Nbox, d=Lbox / Nbox)) * 2 * np.pi

    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing='ij')
    k_perp = np.sqrt(kx3d**2 + ky3d**2)
    k_par = np.abs(kz3d)

    kperp_flat = k_perp.ravel()
    kpar_flat = k_par.ravel()
    power_flat_21 = pk_21.ravel()
    power_flat_LIM = pk_LIM.ravel()
    power_flat_cross = pk_cross.ravel()

    # Choose bin edges
    n_bins = 40
    kperp_bins = np.linspace(0, np.max(k_perp), n_bins + 1)
    kpar_bins = np.linspace(0, np.max(k_par), n_bins + 1)


    P_kperp_kpar_21, _, _, _ = binned_statistic_2d(
        kperp_flat, kpar_flat, power_flat_21,
        statistic='mean', bins=[kperp_bins, kpar_bins]
    )
    P_kperp_kpar_LIM, _, _, _ = binned_statistic_2d(
        kperp_flat, kpar_flat, power_flat_LIM,
        statistic='mean', bins=[kperp_bins, kpar_bins]
    )
    P_kperp_kpar_cross, _, _, _ = binned_statistic_2d(
        kperp_flat, kpar_flat, power_flat_cross,
        statistic='mean', bins=[kperp_bins, kpar_bins]
    )

    # Bin centers
    kperp_centers = 0.5 * (kperp_bins[1:] + kperp_bins[:-1])
    kpar_centers = 0.5 * (kpar_bins[1:] + kpar_bins[:-1])

    # compute C and the cross ratio ...


    return 