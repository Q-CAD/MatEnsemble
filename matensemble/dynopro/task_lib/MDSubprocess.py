
import lammps
import numpy as np
import math
from matensemble.dynopro.utils import preprocessors
import os

def MDSubprocess(split, comm, input_params):

        try:
            gpus_per_node = os.environ.get('SLURM_GPUS_PER_NODE')
        except:
            gpus_per_node = 8

        me = comm.Get_rank()
        nprocs = comm.Get_size()

        
        import lammps
        if input_params['run_on_gpus']:

                lmp = lammps.lammps(comm=split,cmdargs=['-k', 'on', 'g', str(gpus_per_node), '-sf','kk','-pk', 'kokkos','neigh','half','newton', 'off'])
        else:
                lmp = lammps.lammps(comm=split) #,cmdargs=['-screen', 'off'])

        
        # Have to activate mliappy if running with ML-IAP 
        if 'mliap' in input_params.keys():
               if input_params['mliap']:
                        import lammps
                        import lammps.mliap
                        lammps.mliap.activate_mliappy(lmp)


        # initialize a LAMMPS simulation from 'lammps_input' file

        if 'lammps_input' in input_params.keys():
                
                try:
                        lines = open(input_params['lammps_input'],'r').readlines()
                        for line in lines: lmp.command(line)
                except:
                        comm.Abort(1) 

        else:
                try:
                        lmp_input = preprocessors.generate_lammps_input(input_params)
                        lmp.commands_string(lmp_input)
                except:
                        comm.Abort(1)

        box_lo, box_hi, xy, yz, xz, _,_ = lmp.extract_box()
 
        if 'lattice_scale' in input_params.keys():
                lmp.command(f"change_box all x scale {input_params['lattice_scale']} y scale {input_params['lattice_scale']} remap units box ")
        
        if 'shear_scale' in input_params.keys():
                
                xy_final = xy*float(input_params['shear_scale'])
                lmp.command(f"change_box all xy final {xy_final} remap units box ")

        if 'strain_tensor' in input_params.keys():


                xy_final = xy*float(input_params['strain_tensor']['xy'])
                yz_final = yz*float(input_params['strain_tensor']['yz'])
                xz_final = xz*float(input_params['strain_tensor']['xz'])
                
                lmp.command(f"change_box all x scale {input_params['strain_tensor']['xx']} y scale {input_params['strain_tensor']['yy']} z scale {input_params['strain_tensor']['yy']} xy final {xy_final} \
                            yz final {yz_final} xz final {xz_final} remap units box ")


        if 'heat' in input_params.keys():

                T_heat =  input_params['heat']['T_heat']
                lmp.command(f"fix heat all langevin {T_heat} {T_heat} 50 12345")
                lmp.command("fix ensemble all nve")
                try:
                        if 'verlet_delta_t' in input_params.keys():
                                lmp.command(f"timestep {input_params['verlet_delta_t']}")
                        else:
                                print ("No Verlet Time-Step (delta_t) is specified). Will be using default value of 0.1. But this could be spurious--> so please specify!")

                        lmp.command(f"run {input_params['heat']['heat_timesteps']}")
                        lmp.command("unfix heat")
                except:
                        pass # this is the "interesting dynamics" for on-the-fly analysis!

        
        if 'quench' in input_params.keys():
                lmp.command(f"fix quench all langevin {input_params['heat']['T_heat']} {input_params['quench']['T_quench']} 50 12345")
                try:
                        lmp.command(f"run {input_params['quench']['quench_timesteps']}")
                except:
                        pass # this is the "interesting dynamics" for on-the-fly analysis!


        """
        resetting timestep to zero assuming 
        that "interesting" dynamic simulations begin now
        """
        lmp.command("reset_timestep 0")

        MAX_STRIDES = math.ceil(input_params["total_number_of_timesteps"]/input_params["i_o_freq"]/(nprocs - input_params['md_procs']))
        stride = 0

        while stride<MAX_STRIDES:

                md_subtask = 1
                
                while md_subtask<=nprocs-input_params['md_procs']:

                        if stride==0 and md_subtask==1:
                                try:
                                        lmp.command(f"run 0 start 0 stop {input_params['total_number_of_timesteps']}")
                                except:
                                        comm.Abort(1)

                        elif int(lmp.get_thermo('step'))<input_params["total_number_of_timesteps"]:
                                try:
                                        lmp.command(f"run {input_params['i_o_freq']} start 0 stop {input_params['total_number_of_timesteps']}")
                                except:
                                        comm.Abort(1)
                                        
                        else:
                                lmp.close()
                                return

                        lmp_snapshot = extract_lammps_attr(lmp)
                        
                        if me==0:
                                if lmp_snapshot['timestep']<=input_params["total_number_of_timesteps"]:
                                        analysis_rank = md_subtask -1 + input_params['md_procs'] 
                                        comm.send(lmp_snapshot, dest=analysis_rank)    
                        
                        md_subtask+=1    
                        
                stride +=1

        lmp.close()
        return




def extract_lammps_attr(lmp):

        # Extract LAMMPS instance attributes
        x = lmp.gather_atoms("x",1,3)
        coords = np.ctypeslib.as_array(x)

        t = lmp.gather_atoms('type',0,1)
        types = np.ctypeslib.as_array(t)

        Natoms = int(len(x)/3)
        coords.shape = (Natoms, 3)
        types.shape = (Natoms,)
        box_info = lmp.extract_box()
        dim = lmp.extract_setting('dimension')
        timestep = int(lmp.get_thermo('step'))
        
        lmp_snapshot = {'coords': coords, 'box_info': box_info, 'dim': dim, 'timestep':timestep, "types": types}

        return lmp_snapshot


def adaptive_stride(comm, input_params):
        #  timestep = (stride*(nprocs-input_params["md_procs"]) + (rank-input_params["md_procs"]))*input_params["i_o_freq"]
        # HAVE TO IMPLEMENT
        pass
