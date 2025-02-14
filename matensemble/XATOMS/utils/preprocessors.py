import numpy as np
import os
from .stress_rotate_z_theta import *


def create_lammps_data():
        pass

def generate_lammps_input(input_params):
        """
        to run NPT under T=temperature K, and stresse tensor (in GPa)

        """

        if input_params['rotate_stress']:
                stress_rotate_z_theta(input_params['dipole_orientation'], 
                                      input_params['rss'], 
                                      outfile=input_params['stress_info'], lmp_stress_var='.variable_stress.lmp')

        init_commands = f"""
# ------------------------ INITIALIZATION ----------------------------
processors    * * *
units         metal
dimension    3
boundary    p    p    p
atom_style   atomic

#--------------------------- LAMMPS Data File--------------------------
read_data     {input_params['structure']}
change_box    all triclinic
# reset_atom_ids 
include       {'.variable_stress.lmp'}
variable      temp  equal {input_params["temperature"]}
"""
        
        potential_commands = f"""

# ------------------------ FORCE FIELDS ------------------------------
pair_style    {input_params["potential_style"]}
pair_coeff    * * {input_params["potential_file"]} {' '.join(input_params["species"])}
#----------------------------------------------------------------------
"""
        dynamics_commands = """

# Thermal equilibration

timestep 0.001 # 1 fs 
variable      temp_init equal ${temp}*2
velocity all create ${temp_init} 12345 mom yes rot no

# Display thermo
thermo     100
thermo_modify flush yes
thermo_style    custom step temp press pxx pyy pzz pxz pyz pxy 
fix 11 all npt temp ${temp} ${temp} 10 x ${tau_xx} ${tau_xx} 1 y ${tau_yy} ${tau_yy} 1 z ${tau_zz} ${tau_zz} 1 yz ${tau_yz} ${tau_yz} 1 xz ${tau_xz} ${tau_xz} 1 xy ${tau_xy} ${tau_xy} 1

######################################

        """
        cwd = os.getcwd()
        try:
                with open(cwd+'/md_metadata/in.mob', 'w') as file:
                        file.write(init_commands+dynamics_commands)
        except FileNotFoundError:
                os.mkdir(cwd+'/md_metadata')
                with open(cwd+'/md_metadata/in.mob', 'w') as file:
                        file.write(init_commands+dynamics_commands)


        return init_commands+potential_commands+dynamics_commands
