from oLIMpus.cross_correlation_boxes import *

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
            elif par == 'fesc':
                array = values_fesc
            elif par == 'OmegaC':
                array = values_OmC
            else:
                print('Check parameter!')
                return -1 
            
            zero_crossing_1 = np.zeros(len(array))
            distance_zeros_1 = np.zeros(len(array))
            zero_crossing_2 = np.zeros(len(array))
            distance_zeros_2 = np.zeros(len(array))

            for i in range(len(array)):

                first_zero_crossing_val_1 = 0
                second_zero_crossing_val_1 = 0
                first_zero_crossing_val_2 = 0
                second_zero_crossing_val_2 = 0

                P_OIII = np.asarray(P_var[f][p][i]).T[0]
                xH_OIII_value = np.asarray(xH_var[f][p][i]).T[0]
                P_Ha = np.asarray(P_var[f][p][i]).T[1]
                xH_Ha_value = np.asarray(xH_var[f][p][i]).T[1]
                for zi in range(len(zvals)):
                    if zi > 0:

                        if P_OIII[zi-1] < 0 and P_OIII[zi] > 0 and first_zero_crossing_val_1 == 0: 
                            first_zero_crossing_val_1_funct = interp1d([zvals[zi-1], zvals[zi]], [P_OIII[zi-1], P_OIII[zi]]) 
                            first_zero_crossing_val_1 = brentq(first_zero_crossing_val_1_funct, zvals[zi-1], zvals[zi])
                        if P_Ha[zi-1] < 0 and P_Ha[zi] > 0 and first_zero_crossing_val_2 == 0: 
                            first_zero_crossing_val_2_funct = interp1d([zvals[zi-1], zvals[zi]], [P_Ha[zi-1], P_Ha[zi]])
                            first_zero_crossing_val_2 = brentq(first_zero_crossing_val_2_funct, zvals[zi-1], zvals[zi])
                    
                        if P_OIII[zi-1] > 0 and P_OIII[zi] < 0 and second_zero_crossing_val_1 == 0: 
                            second_zero_crossing_val_1_funct = interp1d([zvals[zi-1], zvals[zi]], [P_OIII[zi-1], P_OIII[zi]])
                            second_zero_crossing_val_1 = brentq(second_zero_crossing_val_1_funct, zvals[zi-1], zvals[zi])

                        if P_Ha[zi-1] > 0 and P_Ha[zi] < 0 and second_zero_crossing_val_2 == 0: 
                            second_zero_crossing_val_2_funct = interp1d([zvals[zi-1], zvals[zi]], [P_Ha[zi-1], P_Ha[zi]])
                            second_zero_crossing_val_2 = brentq(second_zero_crossing_val_2_funct, zvals[zi-1], zvals[zi])

                label = r'$\Omega_{\rm c} = %g$'%array[i] if par == 'Omegac' else r'$\epsilon_{*} = %g$'%array[i] if par == 'epsstar' else r'$f_{\rm esc} = %g$'%array[i] 

                plt.subplot(221)
                plt.plot(zvals,P_OIII,label=label,marker='D',color=colors[i])
                plt.plot(zvals,P_Ha,ls='--',marker='D')

                plt.subplot(222)
                plt.plot(zvals,1-xH_OIII_value,label=label,color=colors[i],marker='D')
                plt.plot(zvals,1-xH_Ha_value,label=label,color=colors[i],marker='D',ls='--')
                plt.axvline(zero_crossing_2[i],color=colors[i])
                plt.axvline(zero_crossing_2[i],color=colors[i],ls='--')

                zero_crossing_1[i] = second_zero_crossing_val_1 
                distance_zeros_1[i] = first_zero_crossing_val_1 - second_zero_crossing_val_1
                zero_crossing_2[i] = second_zero_crossing_val_2 
                distance_zeros_2[i] = first_zero_crossing_val_2 - second_zero_crossing_val_2
                
            plt.subplot(223)
            plt.plot(array,zero_crossing_1,label=label,color=colors[i],marker='D')
            plt.plot(array,zero_crossing_2,ls='--',label=label,color=colors[i],marker='D')

            plt.subplot(224)
            plt.plot(array,distance_zeros_1,label=label,color=colors[i],marker='D')
            plt.plot(array,distance_zeros_2,ls='--',label=label,color=colors[i],marker='D')

            plt.subplot(221)        
            plt.xlabel(r'$z$')
            plt.ylabel(r'$P$')
            plt.legend()
            plt.axhline(0,linewidth=0.5)
            plt.ylim(-1,1)

            plt.subplot(222)
            plt.xlabel(r'$z$')
            plt.ylabel(r'$x_{\rm HI}$')
            plt.legend()
            plt.xlim(6,20)

            plt.subplot(223)
            xlab = r'$\Omega_{\rm c}$' if par == 'OmegaC' else r'$\epsilon_{*}$' if par == 'epsstar' else r'$f_{\rm esc}$'
            plt.xlabel(xlab)
            plt.ylabel(r'${\rm Low}-z\,{\rm crossing}$')
            plt.ylim(6,20)

            plt.subplot(224)
            xlab = r'$\Omega_{\rm c}$' if par == 'OmegaC' else r'$\epsilon_{*}$' if par == 'epsstar' else r'$f_{\rm esc}$'
            plt.xlabel(xlab)
            plt.ylabel(r'${\rm Crossing distance}$')
            plt.ylim(0,10)

            plt.tight_layout()

            folder_plot = folder_out + '/plots' 
            if not os.path.exists(folder_plot):
                os.makedirs(folder_plot)
            plt.savefig(folder_plot + '/var_' + par + '.png')
            plt.show()

    return
