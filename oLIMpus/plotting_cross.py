from oLIMpus.cross_correlation_boxes import *


def get_crossings(Pearson1,Pearson2):

    first_zero_crossing_val_1 = 0
    second_zero_crossing_val_1 = 0
    first_zero_crossing_val_2 = 0
    second_zero_crossing_val_2 = 0

    for zi in range(len(zvals)):
        if zi > 0:

            if Pearson1[zi-1] < 0 and Pearson1[zi] > 0 and first_zero_crossing_val_1 == 0: 
                first_zero_crossing_val_1_funct = interp1d([zvals[zi-1], zvals[zi]], [Pearson1[zi-1], Pearson1[zi]]) 
                first_zero_crossing_val_1 = brentq(first_zero_crossing_val_1_funct, zvals[zi-1], zvals[zi])
            if Pearson2[zi-1] < 0 and Pearson2[zi] > 0 and first_zero_crossing_val_2 == 0: 
                first_zero_crossing_val_2_funct = interp1d([zvals[zi-1], zvals[zi]], [Pearson2[zi-1], Pearson2[zi]])
                first_zero_crossing_val_2 = brentq(first_zero_crossing_val_2_funct, zvals[zi-1], zvals[zi])
        
            if Pearson1[zi-1] > 0 and Pearson1[zi] < 0 and second_zero_crossing_val_1 == 0: 
                second_zero_crossing_val_1_funct = interp1d([zvals[zi-1], zvals[zi]], [Pearson1[zi-1], Pearson1[zi]])
                second_zero_crossing_val_1 = brentq(second_zero_crossing_val_1_funct, zvals[zi-1], zvals[zi])

            if Pearson2[zi-1] > 0 and Pearson2[zi] < 0 and second_zero_crossing_val_2 == 0: 
                second_zero_crossing_val_2_funct = interp1d([zvals[zi-1], zvals[zi]], [Pearson2[zi-1], Pearson2[zi]])
                second_zero_crossing_val_2 = brentq(second_zero_crossing_val_2_funct, zvals[zi-1], zvals[zi])

    return first_zero_crossing_val_1, second_zero_crossing_val_1, first_zero_crossing_val_2, second_zero_crossing_val_2


def plot_Pearson(var_line = False, var_astro = True, var_cosmo = True):

    P_fid, k_fid, r_fid, s_fid, xH_fid =  run_all_fiducials()
    P_var, k_var, r_var, s_var, xH_var, var_params = run_variations(var_line, var_astro, var_cosmo)

    flags = len(var_params)

    for f in range(flags):
        for p in range(len(var_params[f])):

            par = var_params[f][p]
            plt.figure(figsize=(12,8))
            if par == 'epsstar':
                array = values_epsstar
                fid_value = AstroParams_input_fid['epsstar']
            elif par == 'fesc':
                array = values_fesc
                fid_value = AstroParams_input_fid['fesc10']
            elif par == 'OmegaC':
                array = values_OmC
                fid_value = CosmoParams_input_fid['omegac']
            else:
                print('Check parameter!')
                return -1 
            
            zero_crossing_1 = np.zeros(len(array))
            distance_zeros_1 = np.zeros(len(array))
            zero_crossing_2 = np.zeros(len(array))
            distance_zeros_2 = np.zeros(len(array))

            for i in range(len(array)):

                P_OIII = np.asarray(P_var[f][p][i]).T[0]
                P_Ha = np.asarray(P_var[f][p][i]).T[1]
                first_zero_crossing_val_1, second_zero_crossing_val_1, first_zero_crossing_val_2, second_zero_crossing_val_2 = get_crossings(P_OIII,P_Ha)

                xH_OIII_value = np.asarray(xH_var[f][p][i]).T[0]
                xH_Ha_value = np.asarray(xH_var[f][p][i]).T[1]

                label = r'$\Omega_{\rm c} = %g$'%array[i] if par == 'OmegaC' else r'$\epsilon_{*} = %g$'%array[i] if par == 'epsstar' else r'$f_{\rm esc} = %g$'%array[i] 

                plt.subplot(221)
                plt.plot(zvals,P_OIII,label=label,color=colors[i+1])
                plt.plot(zvals,P_Ha,ls='--',color=colors[i+1])

                zero_crossing_1[i] = second_zero_crossing_val_1 
                distance_zeros_1[i] = first_zero_crossing_val_1 - second_zero_crossing_val_1
                zero_crossing_2[i] = second_zero_crossing_val_2 
                distance_zeros_2[i] = first_zero_crossing_val_2 - second_zero_crossing_val_2

                plt.subplot(222)
                plt.plot(zvals,1.-xH_OIII_value,label=label,color=colors[i+1])
                plt.plot(zvals,1.-xH_Ha_value,color=colors[i+1],ls='--')
                # plt.axvline(zero_crossing_1[i],color=colors[i+1])
                # plt.axvline(zero_crossing_2[i],color=colors[i+1],ls='--')
                
                plt.subplot(223)
                plt.plot(array,zero_crossing_1,label=label,color=colors[i+1],)
                plt.plot(array,zero_crossing_2,ls='--',color=colors[i+1],)

                plt.subplot(224)
                plt.plot(array,distance_zeros_1,label=label,color=colors[i+1],)
                plt.plot(array,distance_zeros_2,ls='--',color=colors[i+1],)

            P_OIII_fid = np.asarray(P_fid).T[0]
            P_Ha_fid = np.asarray(P_fid).T[1]
            first_zero_crossing_val_1, second_zero_crossing_val_1, first_zero_crossing_val_2, second_zero_crossing_val_2 = get_crossings(P_OIII_fid,P_Ha_fid)

            xH_OIII_fid = np.asarray(xH_fid).T[0]
            xH_Ha_fid = np.asarray(xH_fid).T[1]

            plt.subplot(221)
            plt.plot(zvals,P_OIII_fid,label=r'$\rm fiducial$',color=colors[0])
            plt.plot(zvals,P_Ha_fid,ls='--')

            zero_crossing_1_fid = second_zero_crossing_val_1 
            distance_zeros_1_fid = first_zero_crossing_val_1 - second_zero_crossing_val_1
            zero_crossing_2_fid = second_zero_crossing_val_2 
            distance_zeros_2_fid = first_zero_crossing_val_2 - second_zero_crossing_val_2

            plt.subplot(222)
            plt.plot(zvals,1.-xH_OIII_fid,label=r'$\rm fiducial$',color=colors[0])
            plt.plot(zvals,1.-xH_Ha_fid,color=colors[0],ls='--')
            # plt.axvline(zero_crossing_1_fid,color=colors[0])
            # plt.axvline(zero_crossing_2_fid,color=colors[0],ls='--')

            plt.subplot(223)
            plt.scatter([fid_value],zero_crossing_1_fid,label=r'$\rm fiducial$',color=colors[0],marker='D')
            plt.scatter([fid_value],zero_crossing_2_fid,marker='*',color=colors[0])

            plt.subplot(224)
            plt.scatter([fid_value],distance_zeros_1_fid,label=r'$\rm fiducial$',color=colors[0],marker='D')
            plt.scatter([fid_value],distance_zeros_2_fid,marker='*',color=colors[0])

            fontsize= 20 
            plt.subplot(221)        
            plt.xlabel(r'$z$',fontsize=fontsize)
            plt.ylabel(r'$P$',fontsize=fontsize)
            #plt.legend(fontsize=fontsize)
            plt.axhline(0,linewidth=0.5)
            plt.ylim(-1,1)

            plt.subplot(222)
            plt.xlabel(r'$z$',fontsize=fontsize)
            plt.ylabel(r'$x_{\rm HI}$',fontsize=fontsize)
            plt.legend(fontsize=fontsize-5)
            plt.xlim(6,20)

            plt.subplot(223)
            xlab = r'$\Omega_{\rm c}$' if par == 'OmegaC' else r'$\epsilon_{*}$' if par == 'epsstar' else r'$f_{\rm esc}$'
            plt.xlabel(xlab,fontsize=fontsize)
            plt.ylabel(r'${\rm Low}-z\,{\rm crossing}$',fontsize=fontsize)

            plt.subplot(224)
            xlab = r'$\Omega_{\rm c}$' if par == 'OmegaC' else r'$\epsilon_{*}$' if par == 'epsstar' else r'$f_{\rm esc}$'
            plt.xlabel(xlab,fontsize=fontsize)
            plt.ylabel(r'${\rm Crossing\, distance}$',fontsize=fontsize)

            plt.tight_layout()

            folder_plot = folder_out + '/plots' 
            if not os.path.exists(folder_plot):
                os.makedirs(folder_plot)
            plt.savefig(folder_plot + '/var_' + par + '.png')
            plt.show()

    return
