import numpy as np

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


if __name__ == "__main__":
    n_param = 15
    temperature_data = np.linspace(1500, 3000, n_param)
    #    lattice_strain_data = np.linspace(1.00,1.05, n_param)
    #    shear_strain_data = np.linspace(1.00, 1.05, n_param)
    # grid = np.array(np.meshgrid(temperature_data, lattice_strain_data, shear_strain
    params_dict = []
    # num_samples = len(grid)

    import sys

    partial_samples_init = int(sys.argv[1])
    partial_samples_final = int(sys.argv[2])

    for ic in range(partial_samples_init, partial_samples_final):
        params_dict.append(
            {"Temp_K": temperature_data[ic]}
        )  # , 'Strain': grid[ic,1], 'Shear_Strain': grid[ic,2]}))

        pipe.dynopro(
            gpu_subprocess="md_subprocess",
            cpu_subprocess="anaylsis_subprocess",
            nnodes=3,
            gpus_per_node=8,
            cores_per_node=64,
            num_tasks=168,
        )
