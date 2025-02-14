from XATOMS.postprocessors.bispectrum_calculator import SNAP  
import math

def AnalysisSubprocess(comm, input_params):

        nprocs = comm.Get_size()
        rank = comm.Get_rank()
        MAX_STRIDES = math.ceil(input_params["total_number_of_timesteps"]/input_params["i_o_freq"]/(nprocs - input_params['md_procs']))
        stride = 0 #

        while stride<MAX_STRIDES:
                
       # receive LAMMPS Snapshot


                timestep = (stride*(nprocs-input_params["md_procs"]) + (rank-input_params["md_procs"]))*input_params["i_o_freq"]

                # print (f'currently at rank {rank} and stride {stride} at time step: {timestep} BEFORE receving')
                
                if timestep<=input_params["total_number_of_timesteps"]:
                        lmp_snapshot = comm.recv(source=0)
                        # print (f'currently at rank {rank} and stride {stride} AFTER receving')
                else:
                        return
                
                # Run analysis for each snapshot
                data = OvitoCalculators(lmp_snapshot=lmp_snapshot)

                if input_params["dxa_analysis"]:
                        data.calculate_dxa(export_formats=input_params["dxa_output_formats"], dxa_line_sep=input_params["dxa_line_sep"], lattice="lattice")
                if input_params["full_trajectory_dump"]:
                                data.dump_trajectories(export_formats=input_params["trajectory_output_format"])
                
                stride+=1


        return