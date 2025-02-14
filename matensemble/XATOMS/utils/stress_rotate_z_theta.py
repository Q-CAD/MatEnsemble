import numpy as np
import os

"""

Syntax: python <code> theta tau_res

Assumtions:
          Y-axis is the Burger's vector or [111]
          Rotation axis is Z-axis, the slip plane normal [101]
"""

def stress_rotate_z_theta(theta_deg, tau_res, outfile, lmp_stress_var):
    try:
        os.remove(outfile)
    except OSError as error:
        pass

    theta = theta_deg*np.pi/180

    ############ Form Local stress tensor ###############
    stress_local = np.zeros((3,3))
    stress_local[1,2] = tau_res
    stress_local[2,1] = tau_res
    stress_local[0,0] = 0
    stress_local[2,2] = 0
    ############## Define Rotational matrix (about Z) ##############
    R = np.zeros((3,3))
    R[2,2] = 1.
    R[0,0] = np.cos(theta)
    R[0,1] = np.sin(theta)
    R[1,0] = np.sin(-theta)
    R[1,1] = np.cos(theta)

############## Golbal stress tensor ################
    stress_global = np.dot(np.dot(R, stress_local), R.T)

############## Write stress file ###################
    with open(outfile, 'a') as file:
        file.write('stress (GPa)\n')
        np.savetxt(file, stress_global)


    with open(lmp_stress_var, 'w') as file:
        file.write('variable tau_xx equal '+str(stress_global[0,0]*1e4)+'\n')
        file.write('variable tau_xy equal '+str(stress_global[0,1]*1e4)+'\n')
        file.write('variable tau_xz equal '+str(stress_global[0,2]*1e4)+'\n')
        file.write('variable tau_yy equal '+str(stress_global[1,1]*1e4)+'\n')
        file.write('variable tau_yz equal '+str(stress_global[1,2]*1e4)+'\n')
        file.write('variable tau_zz equal '+str(stress_global[2,2]*1e4)+'\n')
