import numpy as np
import json

from matensemble.pipeline import Pipeline
from matnesemble.dynopro.task_lib.MDSubprocess import MDSubprocess
from matnesemble.dynopro.task_lib.AnalysisSubprocess import AnalysisSubprocess

pipe = Pipeline()


@pipe.chore()
def md_subprocess(comm, split, input_params):
    MDSubprocess(comm, split, input_params)


@pipe.chore()
def anaylsis_subprocess(comm, input_params):
    AnalysisSubprocess(comm, input_params)


INITIAL_PARAMTERS = {
    "species": ["Bi", "Se"],
    "lammps_input": "/lustre/orion/world-shared/lrn090/Demo_workflows_Feb2026/Automated_XRD/Bi-Se/in.recryst.npt",
    "verlet_delta_t": 0.001,
    "heat": {"T_heat": 1500, "heat_timesteps": 100},
    "lattice_scale": 1.025,
    "quench": {"T_quench": 10},
    "run_on_gpus": True,
    "nnodes": 3,
    "gpus_per_node": 8,
    "total_procs": 168,
    "md_procs": 24,
    "i_o_freq": 100,
    "total_number_of_timesteps": 1000,
    "compute_xrd": True,
    "compute_rdf": {"cutoff": 10.0, "number_of_bins": 100, "z_min": 0},
    "full_trajectory_dump": True,
    "trajectory_output_format": ["lammps/dump"],
}


if __name__ == "__main__":
    n_param = 15
    temperature_data = np.linspace(1500, 3000, n_param)
    params_dict = []

    import sys

    partial_samples_init = int(sys.argv[1])
    partial_samples_final = int(sys.argv[2])

    for ic in range(partial_samples_init, partial_samples_final):
        params_dict.append({"Temp_K": temperature_data[ic]})

    N_sim = len(params_dict)
    sim_list = list(np.arange(N_sim))

    for i in range(N_sim):
        INITIAL_PARAMTERS["heat"]["T_heat"] = params_dict[i]["Temp_K"]
        namespace = f"{params_dict[i]['Temp_K']}_K"

        pipe.dynopro(
            nnodes=3,
            gpus_per_node=8,
            cores_per_node=64,
            gpu_subprocess="md_subprocess",
            cpu_subprocess="anaylsis_subprocess",
            gpu_args=(json.dumps(INITIAL_PARAMTERS),),
            cpu_args=(json.dumps(INITIAL_PARAMTERS),),
            name=namespace,
            num_tasks=168,
        )

    pipe.submit()
