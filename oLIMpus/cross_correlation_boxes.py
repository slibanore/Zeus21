from oLIMpus import * 
from powerbox import get_power
from classy import Class
import os
import pickle
from scipy.optimize import brentq
# 2D binning using scipy
from scipy.stats import binned_statistic_2d
from tqdm import tqdm
from oLIMpus.maps_LIM import build_lightcone

from functools import partial
import concurrent.futures

from oLIMpus.fiducial_analysis import * 

save_path = os.path.join(os.getcwd(), "oLIMpus")
folder_out = os.path.abspath(os.path.join(save_path, "..", 'analysis_' + str(Lbox_fid) + ',' + str(Nbox_fid) ))
if not os.path.exists(folder_out):
    os.makedirs(folder_out)


CosmoParams_input_fid = Cosmo_Parameters_Input(**CosmoParams_input_fid)
ClassyCosmo_fid = runclass(CosmoParams_input_fid) 
CosmoParams_fid = Cosmo_Parameters(CosmoParams_input_fid, ClassyCosmo_fid) 
HMFintclass_fid = HMF_interpolator(CosmoParams_fid,ClassyCosmo_fid)
AstroParams_fid = Astro_Parameters(CosmoParams_fid,**AstroParams_input_fid)

corr_21_fid = Correlations(CosmoParams_fid, ClassyCosmo_fid)   
coef_21_fid = get_T21_coefficients(CosmoParams_fid, ClassyCosmo_fid, AstroParams_fid, HMFintclass_fid, zmin=ZMIN)
PS21_fid = Power_Spectra(CosmoParams_fid, AstroParams_fid, ClassyCosmo_fid, corr_21_fid, coef_21_fid, RSD_MODE = RSDMODE)
BMF_fid = BMF(coef_21_fid, HMFintclass_fid,CosmoParams_fid,AstroParams_fid)

LineParams_input_1_fid = LineParams_Input(**LineParams_input_1_fid) 
LineParams1_fid = Line_Parameters(LineParams_input_1_fid) 

LineParams2_input_2_fid = LineParams_Input(**LineParams_input_2_fid) 
LineParams2_fid = Line_Parameters(LineParams2_input_2_fid) 

corr_LIM_1_fid = Correlations_LIM(LineParams1_fid, CosmoParams_fid, ClassyCosmo_fid)
coef_LIM_1_fid = get_LIM_coefficients(CosmoParams_fid,  AstroParams_fid, HMFintclass_fid, LineParams1_fid, zmin=ZMIN)
PSLIM1_fid = Power_Spectra_LIM(CosmoParams_fid, AstroParams_fid, LineParams1_fid, corr_LIM_1_fid, coef_LIM_1_fid, RSD_MODE = RSDMODE)

corr_LIM_2_fid = Correlations_LIM(LineParams2_input_2_fid, CosmoParams_fid, ClassyCosmo_fid)
coef_LIM_2_fid = get_LIM_coefficients(CosmoParams_fid,  AstroParams_fid, HMFintclass_fid, LineParams2_input_2_fid, zmin=ZMIN)
PSLIM_2_fid = Power_Spectra_LIM(CosmoParams_fid, AstroParams_fid, LineParams2_input_2_fid, corr_LIM_2_fid, coef_LIM_2_fid, RSD_MODE = RSDMODE)


def run_all_lighcones():

    which_lightcone = ['density','SFRD','xHI','T21','LIM']
    for i in which_lightcone:
        build_lightcone(i,
                    zvals,
                    Lbox_fid, 
                    Nbox_fid, 
                    seed, 
                    RSDMODE, 
                    False, 
                    LineParams_input_1_fid, 
                    AstroParams_fid, 
                    ClassyCosmo_fid, 
                    HMFintclass_fid, 
                    CosmoParams_input_fid,      
                    ZMIN,    
                    include_label = ''    
                    )

    build_lightcone('LIM',
                zvals,
                Lbox_fid, 
                Nbox_fid, 
                seed, 
                RSDMODE, 
                False, 
                LineParams_input_2_fid, 
                AstroParams_fid, 
                ClassyCosmo_fid, 
                HMFintclass_fid, 
                CosmoParams_input_fid,      
                ZMIN,    
                include_label = ''    
                )

    return 


def run_all_fiducials(ncore=4):

    P_fid = []
    k_fid = []
    r_fid = []
    s_fid = []
    xH_fid = []
    print('Doing fiducials')

    funct = partial(run_analysis,Lbox=Lbox_fid, 
                Nbox=Nbox_fid,
                corr_21=corr_21_fid,
                coef_21=coef_21_fid,
                PS21=PS21_fid,
                BMF_use=BMF_fid,
                corr_LIM=corr_LIM_1_fid,
                coef_LIM=coef_LIM_1_fid,
                PSLIM=PSLIM1_fid,
                LineParams1=LineParams1_fid,
                AstroParams=AstroParams_fid, 
                CosmoParams=CosmoParams_fid,
                HMFintclass=HMFintclass_fid,
                LineParams2=LineParams2_fid,
                corr_LIM2=corr_LIM_2_fid,
                coef_LIM2=coef_LIM_2_fid,
                PSLIM2=PSLIM_2_fid,
                foregrounds = False, 
                store_quantities = True,
                include_label = '')
    
    for iz in tqdm(range(len(zvals))):
        temp = funct(zvals[iz])
        P_fid.append(temp[0])
        k_fid.append(temp[1])
        r_fid.append(temp[2])
        s_fid.append(temp[3])
        xH_fid.append(temp[4])

    #with concurrent.futures.ProcessPoolExecutor(max_workers=ncore) as executor:
    #    results = list(tqdm(executor.map(funct, zvals), total=len(zvals)))
    #P_fid, k_fid, r_fid, s_fid, xH_fid = zip(*results)

    return P_fid, k_fid, r_fid, s_fid, xH_fid


def run_variations(var_line = False, var_astro = True, var_cosmo = True,
                    var_params_astro = ['epsstar', 'fesc'],
                    var_params_cosmo = ['OmegaC'],
                    ncore=4,):

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

            P_var.append([])
            k_var.append([])
            r_var.append([])
            s_var.append([])
            xH_var.append([])
            var_params.append([])

            for j in range(len(values)):
                print(var_params_astro[i] + ' = ' + str(round(values[j],2)))

                include_label = var_params_astro[i] + '_' + str(round(values[j],2))
                P_var_astro[i].append([])
                k_var_astro[i].append([])
                r_var_astro[i].append([])
                s_var_astro[i].append([])
                xH_var_astro[i].append([])
                if var_params_astro[i] == 'epsstar':
                    AstroParams_input_var = {'epsstar':values[j]}
                elif var_params_astro[i] == 'fesc':
                    AstroParams_input_var = {'fesc10':values[j]}

                folder = folder_out + '/z' + str(round(zvals[0],1))
                filename_all = folder + '/slices_' + include_label + str(islice) + '.pkl'
                if not os.path.isfile(filename_all):

                    AstroParams_var = Astro_Parameters(CosmoParams_fid,**AstroParams_input_var)

                    coef_21_var = get_T21_coefficients(CosmoParams_fid, ClassyCosmo_fid, AstroParams_var, HMFintclass_fid, zmin=ZMIN)
                    PS21_var = Power_Spectra(CosmoParams_fid, AstroParams_var, ClassyCosmo_fid, corr_21_fid, coef_21_var, RSD_MODE = RSDMODE)
                    BMF_var = BMF(coef_21_var, HMFintclass_fid,CosmoParams_fid,AstroParams_var)

                    coef_LIM_1_var = get_LIM_coefficients(CosmoParams_fid,  AstroParams_var, HMFintclass_fid, LineParams1_fid, zmin=ZMIN)
                    PSLIM1_var = Power_Spectra_LIM(CosmoParams_fid, AstroParams_var, LineParams1_fid, corr_LIM_1_fid, coef_LIM_1_var, RSD_MODE = RSDMODE)

                    coef_LIM_2_var = get_LIM_coefficients(CosmoParams_fid,  AstroParams_var, HMFintclass_fid, LineParams2_input_2_fid, zmin=ZMIN)
                    PSLIM_2_var = Power_Spectra_LIM(CosmoParams_fid, AstroParams_var, LineParams2_input_2_fid, corr_LIM_2_fid, coef_LIM_2_var, RSD_MODE = RSDMODE)
                else:
                    coef_21_var=coef_21_fid
                    PS21_var=PS21_fid
                    BMF_var=BMF_fid
                    coef_LIM_1_var=coef_LIM_1_fid
                    PSLIM1_var=PSLIM1_fid
                    AstroParams_var=AstroParams_fid
                    coef_LIM_2_var=coef_LIM_2_fid
                    PSLIM_2_var=PSLIM_2_fid

                funct = partial(run_analysis,Lbox=Lbox_fid, 
                    Nbox=Nbox_fid,
                    corr_21=corr_21_fid,
                    coef_21=coef_21_var,
                    PS21=PS21_var,
                    BMF_use=BMF_var,
                    corr_LIM=corr_LIM_1_fid,
                    coef_LIM=coef_LIM_1_var,
                    PSLIM=PSLIM1_var,
                    LineParams1=LineParams1_fid,
                    AstroParams=AstroParams_var, 
                    CosmoParams=CosmoParams_fid,
                    HMFintclass=HMFintclass_fid,
                    LineParams2=LineParams2_fid,
                    corr_LIM2=corr_LIM_2_fid,
                    coef_LIM2=coef_LIM_2_var,
                    PSLIM2=PSLIM_2_var,
                    foregrounds = False, 
                    store_quantities = True,
                    include_label = include_label)
                
                # with concurrent.futures.ProcessPoolExecutor(max_workers=ncore) as executor:
                #     results = list(tqdm(executor.map(funct, zvals), total=len(zvals)))

                # P_list, k_list, r_list, s_list, xH_list = zip(*results)
                # P_var_astro[i][j].append(list(P_list))
                # k_var_astro[i][j].append(list(k_list))
                # r_var_astro[i][j].append(list(r_list))
                # s_var_astro[i][j].append(list(s_list))
                # xH_var_astro[i][j].append(list(xH_list))

                for iz in tqdm(range(len(zvals))):
                    temp = funct(zvals[iz])
                    P_var_astro[i][j].append(temp[0])
                    k_var_astro[i][j].append(temp[1])
                    r_var_astro[i][j].append(temp[2])
                    s_var_astro[i][j].append(temp[3])
                    xH_var_astro[i][j].append(temp[4])

            P_var[i].append(P_var_astro[i])
            k_var[i].append(k_var_astro[i])
            r_var[i].append(r_var_astro[i])
            s_var[i].append(s_var_astro[i])
            xH_var[i].append(xH_var_astro[i])
            var_params[i].append(var_params_astro[i])

    if var_cosmo:
        print('Cosmo params varied: ' + str(var_params_cosmo))

        P_var_cosmo = []
        k_var_cosmo = []
        r_var_cosmo = []
        s_var_cosmo = []
        xH_var_cosmo = []

        P_var.append([])
        k_var.append([])
        r_var.append([])
        s_var.append([])
        xH_var.append([])
        var_params.append([])

        for i in range(len(var_params_cosmo)):
            if var_params_cosmo[i] == 'OmegaC':
                values = values_OmC
            P_var_cosmo.append([])
            k_var_cosmo.append([])
            r_var_cosmo.append([])
            s_var_cosmo.append([])
            xH_var_cosmo.append([])

            for j in range(len(values)):
                print(var_params_cosmo[i] + ' = ' + str(round(values[j],2)))
                include_label = var_params_cosmo[i] + '_' + str(round(values[j],2))

                P_var_cosmo[i].append([])
                k_var_cosmo[i].append([])
                r_var_cosmo[i].append([])
                s_var_cosmo[i].append([])
                xH_var_cosmo[i].append([])
                if var_params_cosmo[i] == 'OmegaC':
                    CosmoParams_input_var = {'omegac':values[j]}


                folder = folder_out + '/z' + str(round(zvals[0],1))
                filename_all = folder + '/slices_' + include_label + str(islice) + '.pkl'
                if not os.path.isfile(filename_all):
                    CosmoParams_input_var = Cosmo_Parameters_Input(**CosmoParams_input_var)
                    ClassyCosmo_var = runclass(CosmoParams_input_var) 
                    CosmoParams_var = Cosmo_Parameters(CosmoParams_input_var, ClassyCosmo_var) 
                    HMFintclass_var = HMF_interpolator(CosmoParams_var,ClassyCosmo_var)
                    AstroParams_var = Astro_Parameters(CosmoParams_var,**AstroParams_input_fid)

                    corr_21_var = Correlations(CosmoParams_var, ClassyCosmo_var)   
                    coef_21_var = get_T21_coefficients(CosmoParams_var, ClassyCosmo_var, AstroParams_var, HMFintclass_var, zmin=ZMIN)
                    PS21_var = Power_Spectra(CosmoParams_var, AstroParams_var, ClassyCosmo_var, corr_21_var, coef_21_var, RSD_MODE = RSDMODE)
                    BMF_var = BMF(coef_21_var, HMFintclass_var,CosmoParams_var,AstroParams_var)

                    corr_LIM_1_var = Correlations_LIM(LineParams1_fid, CosmoParams_var, ClassyCosmo_var)
                    coef_LIM_1_var = get_LIM_coefficients(CosmoParams_var,  AstroParams_var, HMFintclass_var, LineParams1_fid, zmin=ZMIN)
                    PSLIM1_var = Power_Spectra_LIM(CosmoParams_var, AstroParams_var, LineParams1_fid, corr_LIM_1_var, coef_LIM_1_var, RSD_MODE = RSDMODE)

                    corr_LIM_2_var = Correlations_LIM(LineParams2_input_2_fid, CosmoParams_var, ClassyCosmo_var)
                    coef_LIM_2_var = get_LIM_coefficients(CosmoParams_var,  AstroParams_var, HMFintclass_var, LineParams2_input_2_fid, zmin=ZMIN)
                    PSLIM_2_var = Power_Spectra_LIM(CosmoParams_var, AstroParams_var, LineParams2_input_2_fid, corr_LIM_2_var, coef_LIM_2_var, RSD_MODE = RSDMODE)
                else:
                    corr_21_var=corr_21_fid
                    coef_21_var=coef_21_fid
                    PS21_var=PS21_fid
                    BMF_var=BMF_fid
                    corr_LIM_1_var=corr_LIM_1_fid
                    coef_LIM_1_var=coef_LIM_1_fid
                    PSLIM1_var=PSLIM1_fid
                    AstroParams_var=AstroParams_fid 
                    CosmoParams_var=CosmoParams_fid
                    HMFintclass_var=HMFintclass_fid
                    corr_LIM_2_var=corr_LIM_2_fid
                    coef_LIM_2_var=coef_LIM_2_fid
                    PSLIM_2_var=PSLIM_2_fid

                funct = partial(run_analysis,Lbox=Lbox_fid, 
                            Nbox=Nbox_fid,
                            corr_21=corr_21_var,
                            coef_21=coef_21_var,
                            PS21=PS21_var,
                            BMF_use=BMF_var,
                            corr_LIM=corr_LIM_1_var,
                            coef_LIM=coef_LIM_1_var,
                            PSLIM=PSLIM1_var,
                            LineParams1=LineParams1_fid,
                            AstroParams=AstroParams_var, 
                            CosmoParams=CosmoParams_var,
                            HMFintclass=HMFintclass_var,
                            LineParams2=LineParams2_fid,
                            corr_LIM2=corr_LIM_2_var,
                            coef_LIM2=coef_LIM_2_var,
                            PSLIM2=PSLIM_2_var,
                            foregrounds = False, 
                            store_quantities = True,
                            include_label = include_label)
                
                # with concurrent.futures.ProcessPoolExecutor(max_workers=ncore) as executor:
                #     results = list(tqdm(executor.map(funct, zvals), total=len(zvals)))
                # P_list, k_list, r_list, s_list, xH_list = zip(*results)
                # P_var_cosmo[i][j].append(list(P_list))
                # k_var_cosmo[i][j].append(list(k_list))
                # r_var_cosmo[i][j].append(list(r_list))
                # s_var_cosmo[i][j].append(list(s_list))
                # xH_var_cosmo[i][j].append(list(xH_list))

                for iz in tqdm(range(len(zvals))):
                    temp = funct(zvals[iz])
                    P_var_cosmo[i][j].append(temp[0])
                    k_var_cosmo[i][j].append(temp[1])
                    r_var_cosmo[i][j].append(temp[2])
                    s_var_cosmo[i][j].append(temp[3])
                    xH_var_cosmo[i][j].append(temp[4])

            P_var[i].append(P_var_cosmo[i])
            k_var[i].append(k_var_cosmo[i])
            r_var[i].append(r_var_cosmo[i])
            s_var[i].append(s_var_cosmo[i])
            xH_var[i].append(xH_var_cosmo[i])
            var_params[i].append(var_params_cosmo[i])

    return P_var, k_var, r_var, s_var, xH_var, var_params



def run_analysis(z, 
             Lbox, 
             Nbox,
            corr_21,
            coef_21,
            PS21,
            BMF_use,
            corr_LIM,
            coef_LIM,
            PSLIM,
            LineParams1,
            AstroParams, 
            CosmoParams,
            HMFintclass,
            LineParams2,
            corr_LIM2,
            coef_LIM2,
            PSLIM2,
            foregrounds = False, 
            store_quantities = True,
            include_label = ''
            ):

    folder = folder_out + '/z' + str(round(z,1))
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    filename_all = folder + '/slices_' + include_label + str(islice) + '.pkl'
    filename_xH = folder_out + '/xHavg' + include_label + '.dat'
    if os.path.isfile(filename_all) and os.path.isfile(filename_xH):
        #print('Importing ' + filename_all)
        with open(filename_all, 'rb') as handle:
            temp =  pickle.load(handle)
            slice_T21 = temp['T21']
            slice_LIM = temp[LineParams1.LINE]
            if LineParams2:
                slice_LIM2 = temp[LineParams2.LINE]
        T21_box = False
        LIM_box = False
        if LineParams2:
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


        boxes_zeus21 = maps.T21_bubbles(coef_21,PS21,z,Lbox,Nbox,seed,corr_21,CosmoParams,AstroParams,BMF_use)

        # boxes of density, ionization, 21cm
        delta_box = boxes_zeus21.deltamap
        xHI_box = boxes_zeus21.xHI_map
        T21_box = boxes_zeus21.T21map

        boxes_oLIMpus = CoevalBoxes_percell('full',coef_LIM,corr_LIM,PSLIM,coef_21,z,LineParams1,AstroParams,HMFintclass,CosmoParams,Lbox,Nbox,seed)

        # boxes of SFRD and line intensity
        SFRD_box = boxes_oLIMpus.SFRD_box
        LIM_box = boxes_oLIMpus.Inu_box

        if LineParams2:
            # corr_LIM2 = Correlations_LIM(LineParams2, CosmoParams, ClassyCosmo)
            # coef_LIM2 = get_LIM_coefficients(CosmoParams,  AstroParams, HMFintclass, LineParams2, zmin=ZMIN)
            # PSLIM2 = Power_Spectra_LIM(CosmoParams, AstroParams, LineParams2, corr_LIM2, coef_LIM2, RSD_MODE = RSDMODE)

            boxes_oLIMpus2 = CoevalBoxes_percell('full',coef_LIM2,corr_LIM2,PSLIM2,coef_21,z,LineParams2,AstroParams,HMFintclass,CosmoParams,Lbox,Nbox,seed)

            # box of intensity
            LIM_box2 = boxes_oLIMpus2.Inu_box

        # 4) get 21cm and LIM slices
        slice_T21 = T21_box[islice]
        slice_LIM = LIM_box[islice]
        if LineParams2:
            slice_LIM2 = LIM_box2[islice]

        xH_avg = np.mean(xHI_box)

        # 5) store quantities for animation and lightcone
        if store_quantities:

            with open(filename_all, 'wb') as handle:
                if LineParams2:
                    pickle.dump({'delta': delta_box[islice], 'T21': slice_T21, 'SFRD': SFRD_box[islice], \
                    'xHI': xHI_box[islice], LineParams1.LINE: slice_LIM, LineParams2.LINE: slice_LIM2}, handle)
                else:
                    pickle.dump({'delta': delta_box[islice], 'T21': slice_T21, 'SFRD': SFRD_box[islice], \
                    'xHI': xHI_box[islice], LineParams1.LINE: slice_LIM}, handle)

            if not os.path.exists(filename_xH):
                with open(filename_xH, 'w') as f:
                    f.write("# z xH\n")  # optional header
            # Append the new (z, xH) value
            with open(filename_xH, 'a') as f:
                f.write(f"{z:.6e} {xH_avg:.6e}\n")

    # 6) compute correlations
    Pear = Pearson(slice_LIM, slice_T21, foregrounds)
    k, ratio_cross = ratio_pk(LIM_box, T21_box, Lbox, Nbox, LineParams1.LINE, foregrounds, store_quantities, folder, include_label)
    squared_cross_val = squared_cross(LIM_box, T21_box, Lbox, Nbox, LineParams1.LINE, store_quantities, folder, include_label)
    if LineParams2:
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

    # filename_all = folder + '/powerspectra_' + include_label + line + '.pkl'

    # if os.path.isfile(filename_all):
    #     with open(filename_all, 'rb') as handle:
    #         temp =  pickle.load(handle)
    #         k_cross = temp['k']
    #         pk_21 = temp['Pk_21']
    #         pk_LIM = temp['Pk_line']
    #         pk_cross = temp['Pk_cross']

    # else:
    #     if not box_LIM or not box_T21:
    #         print('Warning! The function ratio_pk requires boxes to run, since there are no spectra stored')
    #         return -1

    # kx = np.fft.fftshift(np.fft.fftfreq(Nbox, d=Lbox / Nbox)) * 2 * np.pi
    # ky = np.fft.fftshift(np.fft.fftfreq(Nbox, d=Lbox / Nbox)) * 2 * np.pi
    # kz = np.fft.fftshift(np.fft.fftfreq(Nbox, d=Lbox / Nbox)) * 2 * np.pi

    # kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing='ij')
    # k_perp = np.sqrt(kx3d**2 + ky3d**2)
    # k_par = np.abs(kz3d)

    # kperp_flat = k_perp.ravel()
    # kpar_flat = k_par.ravel()
    # power_flat_21 = pk_21.ravel()
    # power_flat_LIM = pk_LIM.ravel()
    # power_flat_cross = pk_cross.ravel()

    # # Choose bin edges
    # n_bins = 40
    # kperp_bins = np.linspace(0, np.max(k_perp), n_bins + 1)
    # kpar_bins = np.linspace(0, np.max(k_par), n_bins + 1)


    # P_kperp_kpar_21, _, _, _ = binned_statistic_2d(
    #     kperp_flat, kpar_flat, power_flat_21,
    #     statistic='mean', bins=[kperp_bins, kpar_bins]
    # )
    # P_kperp_kpar_LIM, _, _, _ = binned_statistic_2d(
    #     kperp_flat, kpar_flat, power_flat_LIM,
    #     statistic='mean', bins=[kperp_bins, kpar_bins]
    # )
    # P_kperp_kpar_cross, _, _, _ = binned_statistic_2d(
    #     kperp_flat, kpar_flat, power_flat_cross,
    #     statistic='mean', bins=[kperp_bins, kpar_bins]
    # )

    # # Bin centers
    # kperp_centers = 0.5 * (kperp_bins[1:] + kperp_bins[:-1])
    # kpar_centers = 0.5 * (kpar_bins[1:] + kpar_bins[:-1])

    # compute C and the cross ratio ...


    return None