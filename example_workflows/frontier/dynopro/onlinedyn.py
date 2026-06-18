from matensemble.dynopro.ensemble import EnsembleDynamicsRunner

# from lossfunctions import *
import numpy as np
import pandas as pd
import os


def onlineMD(candidate_parameters, initial_parameters_file=None):
    # create a list of task indicators; In the following I use integers as task-IDs.

    N_sim = len(candidate_parameters)
    sim_list = list(np.arange(N_sim))

    # Input argument list specific to each task in the sorted as task_ids
    import json

    if initial_parameters_file != None:
        try:
            with open(initial_parameters_file, "r") as file:
                input_params = json.load(file)
        except:
            print(f"Have to sepcify initial_paramters file to setup MD")

    sim_args_list = []
    sim_dir_list = []

    # initiate and launch online streaming service
    # from matensemble.redis.service import RedisService
    # rds = RedisService()
    # rds.launch()

    for i in range(N_sim):
        input_params["heat"]["T_heat"] = candidate_parameters[i]["Temp_K"]
        namespace = f"{candidate_parameters[i]['Temp_K']}_K"
        # input_params['stream'] = {}
        # input_params['stream']['host'] = rds.host
        # input_params['stream']['port'] = rds.port
        # input_params['stream']['namespace'] = namespace

        sim_args_list.append(json.dumps(input_params))
        sim_dir_list.append(namespace)

    edr = EnsembleDynamicsRunner(
        sim_list=sim_list,
        sim_args_list=sim_args_list,
        sim_dir_list=sim_dir_list,
        tasks_per_job=input_params["total_procs"],
        cores_per_task=1,
        gpus_per_task=0,
        nnodes=input_params["nnodes"],
        gpus_per_node=input_params["gpus_per_node"],
        write_restart_freq=10,
        buffer_time=0.5,
        adaptive=True,
    )

    edr.run()
    return


def get_traj_data(rds, namespace, key, time_window):

    traj_data = rds.extract_from_stream(namespace=namespace, key=key)
    start_time, end_time = time_window

    window_traj_data = traj_data[
        (traj_data["timestep"] >= start_time) & (traj_data["timestep"] <= end_time)
    ].copy()
    return window_traj_data


def compute_md_loss(
    params_dict, initial_parameters_file, observable_key, time_window, target
):

    rds, namespace_list = onlineMD(
        candidate_parameters=params_dict,
        initial_parameters_file=initial_parameters_file,
    )
    target_x_grid = target[:, 0]
    target_pdf = target[:, 1]

    import wasserstein as ws

    mean_pdf_series = []

    for ic, ns in enumerate(namespace_list):
        traj_data = get_traj_data(rds, ns, observable_key, time_window=time_window)
        traj_data.to_csv(f"check_redis_twist_output_for_{ns}.csv", sep=" ", index=None)

        pdf_series = []

        for _, row in traj_data.iterrows():
            data = row[observable_key]
            timestep = row["timestep"]

            pdf = ws.get_pdf_from_data(target_x_grid, data)
            pdf_series.append(pdf)

        pdf_series = np.array(pdf_series)

        _, mean_pdf = ws.barycenter_1d(target_x_grid, pdf_series)
        mean_pdf_series.append(mean_pdf)

    mean_pdf_series = np.array(mean_pdf_series)
    w2 = ws.w2_distance(target_x_grid, mean_pdf_series, target_pdf)
    print(f"w2 loss vector :", w2)
    rds.shutdown()

    return w2


if __name__ == "__main__":
    n_param = 15
    temperature_data = np.linspace(1500, 3000, n_param)
    #    lattice_strain_data = np.linspace(1.00,1.05, n_param)
    #    shear_strain_data = np.linspace(1.00, 1.05, n_param)
    # grid = np.array(np.meshgrid(temperature_data, lattice_strain_data, shear_strain_data)).T.reshape(-1,3)
    params_dict = []
    # num_samples = len(grid)

    import sys

    partial_samples_init = int(sys.argv[1])
    partial_samples_final = int(sys.argv[2])

    for ic in range(partial_samples_init, partial_samples_final):
        params_dict.append(
            {"Temp_K": temperature_data[ic]}
        )  # , 'Strain': grid[ic,1], 'Shear_Strain': grid[ic,2]}))

    onlineMD(
        candidate_parameters=params_dict,
        initial_parameters_file="input_parameters.json",
    )
