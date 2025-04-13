from oLIMpus import * 
from powerbox import get_power
from classy import Class
import os
import pickle

from .analysis_fiducial import * 

def run_all_fiducials():

    P_fid = []
    k_fid = []
    r_fid = []
    s_fid = []
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

    return P_fid, r_fid, s_fid


def run_variations(var_line = False, var_astro = True, var_cosmo = True):

    P_var = []
    k_var = []
    r_var = []
    s_var = []

    if var_line:
        print('What parameters do you want to change in the line model?')
        return -1

    if var_astro:
        var_params = ['epsstar', 'fesc']
        print('Astro params varied: ' + str(var_params))

        P_var_astro = []
        k_var_astro = []
        r_var_astro = []
        s_var_astro = []

        for i in range(len(var_params)):
            if var_params[i] == 'fstar':
                values = np.linspace(0.01,1,10)
            elif var_params[i] == 'fesc':
                values = np.linspace(0.01,1,10)
            
            P_var_astro.append([])
            k_var_astro.append([])
            r_var_astro.append([])
            s_var_astro.append([])

            for j in range(len(values)):
                include_label = var_params[i] + '_' + str(round(values[j]))
                P_var_astro[j].append([])
                k_var_astro[j].append([])
                r_var_astro[j].append([])
                s_var_astro[j].append([])
                if var_params[i] == 'epsstar':
                    AstroParams_input_var = {'epsstar':values[j]}
                elif var_params[i] == 'fesc':
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

        P_var.append(P_var_astro)
        k_var.append(P_var_astro)
        r_var.append(P_var_astro)
        s_var.append(P_var_astro)
            
    if var_cosmo:
        var_params = ['OmegaC']
        print('Astro params varied: ' + str(var_params))

        P_var_cosmo = []
        k_var_cosmo = []
        r_var_cosmo = []
        s_var_cosmo = []

        for i in range(len(var_params)):
            if var_params[i] == 'OmegaC':
                values = np.linspace(0.01,1,10)
            
            P_var_cosmo.append([])
            k_var_cosmo.append([])
            r_var_cosmo.append([])
            s_var_cosmo.append([])

            for j in range(len(values)):

                include_label = var_params[i] + '_' + str(round(values[j]))

                P_var_cosmo[j].append([])
                k_var_cosmo[j].append([])
                r_var_cosmo[j].append([])
                s_var_cosmo[j].append([])
                if var_params[i] == 'OmegaC':
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

        P_var.append(P_var_cosmo)
        k_var.append(P_var_cosmo)
        r_var.append(P_var_cosmo)
        s_var.append(P_var_cosmo)
            

    return P_var, k_var, r_var, s_var


def run_analysis(z, 
             Lbox, 
             Nbox,
            LineParams_input,
            AstroParams_input, 
            CosmoParams_input,
            LineParams2_input = False,
            store_quantities = True,
            include_label = ''
            ):

    CosmoParams_input = Cosmo_Parameters_Input(**CosmoParams_input)
    LineParams_input = LineParams_Input(**LineParams_input) 
    LineParams = Line_Parameters(LineParams_input) 
    if LineParams2_input:
        LineParams2_input = LineParams_Input(**LineParams2_input) 
        LineParams2 = Line_Parameters(LineParams2_input) 

    folder_out = './analysis_' + str(Lbox) + ',' + str(Nbox) 
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    folder = folder_out + '/z' + str(round(z,1))
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    filename_all = folder + '/slices_' + include_label + str(islice) + '.pkl'
    if os.path.isfile(filename_all):
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

        # 5) store quantities for animation and lightcone
        if store_quantities:

            with open(filename_all, 'wb') as handle:
                if LineParams2_input:
                    pickle.dump({'delta': delta_box[islice], 'T21': slice_T21, 'SFRD': SFRD_box[islice], \
                    'xHI': xHI_box[islice], LineParams.LINE: slice_LIM, LineParams2.LINE: slice_LIM2}, handle)
                else:
                    pickle.dump({'delta': delta_box[islice], 'T21': slice_T21, 'SFRD': SFRD_box[islice], \
                    'xHI': xHI_box[islice], LineParams.LINE: slice_LIM}, handle)

    # 6) compute correlations
    Pear = Pearson(slice_LIM, slice_T21)
    k, ratio_cross = ratio_pk(LIM_box, T21_box, Lbox, Nbox, LineParams.LINE, store_quantities, folder, include_label)
    squared_cross_val = squared_cross(LIM_box, T21_box, Lbox, Nbox, LineParams.LINE, store_quantities, folder, include_label)
    if LineParams2_input:
        Pear2 = Pearson(slice_LIM2, slice_T21)
        k2, ratio_cross2 = ratio_pk(LIM_box2, T21_box, Lbox, Nbox, LineParams2.LINE, store_quantities, folder, include_label)
        squared_cross_val2 = squared_cross(LIM_box2, T21_box, Lbox, Nbox, LineParams2.LINE, store_quantities, folder, include_label)

    if LineParams2:
        return [Pear, Pear2], [k,k2], [ratio_cross, ratio_cross2], [squared_cross_val, squared_cross_val2]
    else:
        return Pear, k, ratio_cross, squared_cross_val


def Pearson(slice_LIM, slice_T21):

    cross_TLIM1 = np.corrcoef((slice_T21.flatten()), slice_LIM.flatten())[0, 1]

    return cross_TLIM1


def ratio_pk(box_LIM, box_T21, Lbox, Nbox, line, store_quantities, folder, include_label):

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


    return 