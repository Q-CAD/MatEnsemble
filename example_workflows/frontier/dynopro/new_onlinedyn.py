import numpy as np
import json
import copy

from matensemble.pipeline import Pipeline

DEFAULT_INITIAL_PARAMETERS = {
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


def _load_initial_parameters(initial_parameters_file=None):
    if initial_parameters_file is None:
        return copy.deepcopy(DEFAULT_INITIAL_PARAMETERS)

    with open(initial_parameters_file, "r") as file:
        return json.load(file)


def _register_dynopro_chores(pipe):
    @pipe.chore(name="md_subprocess")
    def md_subprocess(input_params, *, comm, split):
        from matensemble.dynopro.task_lib.MDSubprocess import MDSubprocess

        MDSubprocess(split, comm, input_params=json.loads(input_params))

    @pipe.chore(name="analysis_subprocess")
    def analysis_subprocess(input_params, *, comm):
        from matensemble.dynopro.task_lib.AnalysisSubprocess import AnalysisSubprocess

        AnalysisSubprocess(comm, input_params=json.loads(input_params))


def onlineMD(candidate_parameters, initial_parameters_file=None):
    input_template = _load_initial_parameters(initial_parameters_file)
    pipe = Pipeline()
    _register_dynopro_chores(pipe)

    for candidate in candidate_parameters:
        input_params = copy.deepcopy(input_template)
        input_params["heat"]["T_heat"] = candidate["Temp_K"]
        namespace = f"{candidate['Temp_K']}_K"
        input_params_json = json.dumps(input_params)

        pipe.dynopro(
            nnodes=input_params["nnodes"],
            gpus_per_node=input_params["md_procs"],
            cores_per_node=input_params["total_procs"],
            gpu_subprocess="md_subprocess",
            cpu_subprocess="analysis_subprocess",
            gpu_args=(input_params_json,),
            cpu_args=(input_params_json,),
            name=namespace,
            num_tasks=input_params["total_procs"],
        )

    future = pipe.submit(buffer_time=0.5, adaptive=True, dynopro=True)
    future.result()
    return


if __name__ == "__main__":
    n_param = 15
    temperature_data = np.linspace(1500, 3000, n_param)
    params_dict = []

    import sys

    partial_samples_init = int(sys.argv[1])
    partial_samples_final = int(sys.argv[2])

    for ic in range(partial_samples_init, partial_samples_final):
        params_dict.append({"Temp_K": temperature_data[ic]})

    onlineMD(
        candidate_parameters=params_dict,
        initial_parameters_file="input_parameters.json",
    )
