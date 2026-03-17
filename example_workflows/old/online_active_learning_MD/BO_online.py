
import subprocess
import sys
from typing import Tuple


'''Check which packages are already installed/ need to be installed and install
them automatically.'''

def install_package(package: str) -> None:
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_and_install_packages(required_packages: dict) -> None:
    """Check and install missing packages."""
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"{package} is already installed.")
        except ImportError:
            print(f"{package} is not installed. Installing now...")
            install_package(package)

# Import required packages
import torch
import pandas as pd
import numpy as np
import gpytorch
from sklearn.preprocessing import MinMaxScaler
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound, \
    qProbabilityOfImprovement, qSimpleRegret
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import GenericMCObjective
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
#import imageio
import random
import seaborn as sns



# MD_PROCS=1100 #112

# @title Utility Functions

# Set random seed for reproducibility
def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Load dataset
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = df.drop(columns=['Folder Name'])
    return df

# Initialize data in tensor format
def initialize_tensors(df: pd.DataFrame) -> tuple:
    X_np = df[['Temperature_K', 'Strain', 'Shear_Strain']].values
    Y_np = df['Avg_twist_ang'].values
    X_i = torch.tensor(X_np, dtype=torch.float64).to(device)
    Y_org = torch.tensor(Y_np, dtype=torch.float64).unsqueeze(-1).to(device)
    return (X_i, Y_org, torch.tensor(X_np, dtype=torch.float64).to(device),
            torch.tensor(Y_np, dtype=torch.float64).unsqueeze(-1).to(device))

# Function to generate seed points at the beginning of each BO iteration
def initialize_samples(X_i: torch.Tensor, Y_org: torch.Tensor, \
                       num_samples: int) -> tuple:
    indices = torch.randperm(X_i.size(0))[:num_samples]
    X_init = X_i[indices]
    Y_init = Y_org[indices]
    return X_init, Y_init

# Initialization of GP model
def initialize_model(X: torch.Tensor, Y: torch.Tensor, \
                     kernel: gpytorch.kernels.Kernel) -> SingleTaskGP:
    gp_model = SingleTaskGP(X, Y, covar_module=kernel).to(device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    fit_gpytorch_mll(mll)
    return gp_model

# Function for optimization of acquisition function
def optimize_acquisition_function(acq_func, gp_model: SingleTaskGP, \
                                  Y_sample: torch.Tensor, bounds: torch.Tensor, \
                                  batch_size: int, best_f: torch.Tensor) -> torch.Tensor:
    acq = acq_func(gp_model, best_f=best_f)
    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=batch_size,
        num_restarts=5,
        raw_samples=20,
    )
    return candidates


def general_optoptimize_acquisition_function(acq_func, bounds: torch.Tensor, batch_size: int) -> torch.Tensor:
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=5,
        raw_samples=20,
    )
    return candidates

def thompson_sampling_acquisition(gp_model, batch_size=3):
    from botorch.sampling.get_sampler import SobolQMCNormalSampler
    sampler = SobolQMCNormalSampler(torch.Size([batch_size]))  # Batch sampling
    thompson_acq = qSimpleRegret(model=gp_model, sampler=sampler)
    return thompson_acq

# Function to rank the unique candidates based on acquisition function values
def rank_candidates(acq_func, gp_model, candidates, best_f=None):
    individual_acq_values = []
    for point in candidates:
        if acq_func in [qExpectedImprovement, qProbabilityOfImprovement]:
            acq = acq_func(gp_model, best_f=best_f)
        elif acq_func == qUpperConfidenceBound:
            acq = acq_func(gp_model, beta=0.1)
        else:
            acq = acq_func

        with torch.no_grad():
            acq_value = acq(point.unsqueeze(0).to(device))
        individual_acq_values.append(acq_value.item())

    ranked_candidates = torch.stack([x for _, x in \
                                     sorted(zip(individual_acq_values, candidates), \
                                     key=lambda pair: pair[0], reverse=True)])
    ranked_values = sorted(individual_acq_values, reverse=True)
    return ranked_candidates, ranked_values

# Function to save ranked unique candidates from each iteration to .dat files
def save_rank_file(acq_func_name: str, \
                   iteration: int, ranked_candidates: list) -> None:
    rank_file_path = f'rank_{acq_func_name}_iter{iteration}.dat'
    with open(rank_file_path, 'w') as f:
        f.write('Ranked Candidates:\n')
        for candidate in ranked_candidates:
            f.write(f"Candidate: {candidate}\n")

# Function to extract the Y value prediction from GP model post each BO iteration from the ranked unique candidates
def get_Y_next(candidates_dict: dict, target_value : float, initial_parameters_file: str) -> tuple:
    from onlineMD import onlineMD
    md_output = onlineMD(candidate_parameters=candidates_dict, initial_parameters_file=initial_parameters_file)
    Y_next = torch.tensor((np.array(md_output['output'])-target_value)**2, dtype=torch.float64).unsqueeze(-1).to(device)

    return Y_next # Y_next_mean.unsqueeze(-1), Y_next_variance.unsqueeze(-1)

# Function to augment original dataset after each BO loop
def update_samples(unique_candidates_tensor: torch.Tensor, X_sample: torch.Tensor, Y_sample: torch.Tensor,\
                   Y_next: torch.Tensor) -> tuple:
    unique_candidates_tensor = unique_candidates_tensor.to(device)
    Y_next = Y_next.to(device)
    X_sample = X_sample.to(device)
    Y_sample = Y_sample.to(device)

    X_sample = torch.cat([X_sample, unique_candidates_tensor])
    Y_sample = torch.cat([Y_sample, Y_next])
    return X_sample, Y_sample

# Function to find the closest X values in X_i for the unique candidates
def find_closest_X_values(unique_candidates: torch.Tensor, X_org: torch.Tensor) -> torch.Tensor:
    closest_X_values = []
    for candidate in unique_candidates:
        distances = torch.norm(X_i - candidate, dim=1)
        closest_idx = torch.argmin(distances)
        closest_X_values.append(X_i[closest_idx])
    return torch.stack(closest_X_values)

def get_true_Y_closest(unique_candidates_tensor: torch.Tensor, X_org: torch.Tensor, Y_true: torch.Tensor) -> torch.Tensor:
    true_Y_closest = []
    for candidate in unique_candidates_tensor:
        distances = torch.norm(X_i - candidate, dim=1)
        closest_idx = torch.argmin(distances).item()  # Use argmin to get the index of the closest X
        true_Y_closest.append(Y_true[closest_idx].item())
    return torch.tensor(true_Y_closest, dtype=torch.float64).to(device)

# Function to calculate model uncertainty
def calculate_uncertainty(gp_model: SingleTaskGP, X_sample: torch.Tensor) -> float:
    X_sample = X_sample.to(device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = gp_model.likelihood(gp_model(X_sample))
        variance = observed_pred.variance
    return torch.mean(variance).item()

# Prints iteration information post each BO iteration
def print_iteration_info(iteration: int, candidates: torch.Tensor,
                         Y_next: torch.Tensor) -> None:
    print(f"Iteration {iteration + 1}: Candidates = {candidates}, \
            Y_next = {Y_next.cpu().numpy().flatten()}\n")

# Function calculate model performance -> to pull True Avg. Twist Angle values from MD simulations.
def MD_set_get(gp_model: SingleTaskGP, candidates: torch.Tensor, \
               Y_true: torch.Tensor) -> float:
    gp_model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Get predictive distribution from the model
        pred_dist = gp_model.likelihood(gp_model(candidates))
        # Extract the mean and variance
        Y_pred = pred_dist.mean
        uncertainty = pred_dist.variance
    # Compute the difference between true values and predicted means
    prediction_error = (Y_pred - Y_true).abs()
    # Calculate weighted uncertainty by comparing prediction error with predicted variance
    weighted_uncertainty = (prediction_error / uncertainty).mean().item()

    return weighted_uncertainty

# Function to create a final dataframe of final augmented dataset
def create_results_dataframe(X_org: torch.Tensor, Y_true: torch.Tensor) -> pd.DataFrame:
    return pd.DataFrame({
        'Temperature_K': X_org[:, 0].cpu().numpy(),
        'Strain': X_org[:, 1].cpu().numpy(),
        'Shear_Strain': X_org[:, 2].cpu().numpy(),
        'Objective_Value': Y_true.flatten().cpu().numpy()
    })

# Function to create 3D sampling evolution plots -> Gives a temporal sense of evolution of BO iterations
def plot_3d_scatter(unique_candidates, X_sample: torch.Tensor, title: str, Y_sample: torch.Tensor) -> str:
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')

    X_sample_np = X_sample.cpu().numpy()
    Y_sample_np = Y_sample.cpu().numpy()

    # Plot original data points
    # scatter = ax.scatter(X_i_np[:, 1], X_i_np[:, 2], X_i_np[:, 0], c=Y_org_np, \
    #                      cmap='viridis', marker='o', s=50, edgecolor='black', \
    #                      alpha=0.6, label='Ground truth (Avg. Twist Angles)')

    # Plot initial samples
    scatter = ax.scatter(X_sample_np[:, 1], X_sample_np[:, 2], X_sample_np[:, 0], c=Y_sample_np, \
               marker='^', alpha=1.0, s=100, label='Sampled Points')
    # Plot candidates
    if len(unique_candidates) > 0:
        candidates = np.array(unique_candidates).reshape(-1, 3)
        ax.scatter(candidates[:, 1], candidates[:, 2], candidates[:, 0], c='r',\
                   marker='d', alpha=1.0, s=180, label='Candidates')

    ax.set_xlabel('Strain')
    ax.set_ylabel('Shear_Strain')
    ax.set_zlabel('Temperature_K')
    ax.set_title(title)
    ax.legend()

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Average Twist Angle')

    # Increase vertical distance between legends
    plt.legend(bbox_to_anchor=(1.25, 1), loc='upper right', frameon=False, \
               numpoints=1, markerscale=1, handletextpad=0.5, labelspacing=1.0)

    plot_filename = f'{title}.png'
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    return plot_filename

# Function to create .gif from images of BO iterations
#def create_gif(plot_filenames: list, gif_filename: str, delay: float = 100.0) -> None:
#    with imageio.get_writer(gif_filename, mode='I', duration=delay) as writer:
#        for filename in plot_filenames:
#            image = imageio.imread(filename)
#            writer.append_data(image)
#            os.remove(filename)

# Define Bayesian Optimization Function
def bayesian_optimization(n_iterations: int, X_sample: torch.Tensor, \
                          Y_sample: torch.Tensor, bounds: torch.Tensor, \
                          desired_target: float, initial_param_filepath: str, batch_size : int, tolerance=1e-2, restart=False) -> pd.DataFrame:

    # batch_size = 5

    previous_candidates = set()
    plot_filenames = []
    
    for i in range(n_iterations):
        
        # save sample status every iteration
        df = create_results_dataframe(X_sample, Y_sample)
        if restart:
            df.to_csv(f'sample_in_iteration{i+1+restart}', sep=' ')
        else:
            df.to_csv(f'sample_in_iteration{i+1}', sep=' ')


        # initialize GP
        kernel = gpytorch.kernels.MaternKernel().to(device)
        gp_model = initialize_model(X_sample, Y_sample, kernel)
        acq_func = thompson_sampling_acquisition(gp_model, batch_size=nbatch)

        candidates = general_optoptimize_acquisition_function(acq_func=acq_func, bounds=bounds, batch_size=nbatch)
        ranked_candidates, ranked_values = rank_candidates(acq_func, gp_model, \
                                                           candidates, torch.tensor(desired_target,\
                                                                                    dtype=torch.float64).to(device))

        unique_candidates = []
        for candidate in ranked_candidates:
            candidate_tuple = tuple(candidate.cpu().numpy())
            if candidate_tuple not in previous_candidates:
                unique_candidates.append(candidate.cpu().numpy())
                previous_candidates.add(candidate_tuple)

        if unique_candidates:
            unique_candidates_tensor = torch.tensor(np.array(unique_candidates), dtype=torch.float64).to(device)
            if unique_candidates_tensor.dim() > 2:
                unique_candidates_tensor = unique_candidates_tensor.squeeze()

            # Ensure unique_candidates_tensor is in the same device as other tensors
            if torch.cuda.is_available():
                unique_candidates_tensor = unique_candidates_tensor.to(device)

            
            if unique_candidates_tensor.dim() < 2:
                unique_candidates_tensor = unique_candidates_tensor.unsqueeze(0)

            # Extract unique candidates as a dictionary
            unique_candidates_dicts = []
            for row in unique_candidates_tensor.cpu().numpy():
                unique_candidates_dicts.append({
                    'Temp_K': row[0],
                    'Strain': row[1],
                    'Shear_Strain': row[2]
                })


            # FETCH NEXT ROUND OF GROUND TRUTH
            Y_next = get_Y_next(unique_candidates_dicts, target_value=desired_target, initial_parameters_file=initial_param_filepath)
            X_sample, Y_sample = update_samples(unique_candidates_tensor, \
                                                                X_sample, Y_sample, \
                                                                Y_next)
        
            if restart:
                plot_filename = plot_3d_scatter(unique_candidates, X_sample, \
                                                f'Iteration_{i+1+restart}', Y_sample)
            else:
                plot_filename = plot_3d_scatter(unique_candidates, X_sample, \
                                                f'Iteration_{i+1}', Y_sample)
                
                plot_filenames.append(plot_filename)

        else:
            print("No unique candidates found. Skipping update for this iteration.")
            ranked_candidates = []
            candidates = []
            Y_next = []

        if restart:
            save_rank_file('thompson_sapling', i+int(restart)-1, ranked_candidates)
            print_iteration_info(i+int(restart)-1, candidates, Y_next)
        else:
            save_rank_file('thompson_sampling', i, ranked_candidates)
            print_iteration_info(i, candidates, Y_next)
        
        # STOPPING CRITERIA

        if np.abs(desired_target - Y_sample[-1])<=tolerance:
            break

        #-------------------#


    results_df = create_results_dataframe(X_sample, Y_sample)
#    create_gif(plot_filenames, 'bayesian_optimization.gif', delay=1000.0)
    return results_df, unique_candidates_dicts


def restart_BO(sample_file):

    data = pd.read_csv(sample_file, sep='\s+')
    X_sample = np.array([data['Temperature_K'], data['Strain'], data['Shear_Strain']]).T
    Y_sample = np.array(data['Objective_Value'])

    return X_sample, Y_sample

if __name__ == "__main__":
    required_packages = {
        "torch": "torch",
        "botorch": "botorch",
        "pandas": "pandas",
        "matplotlib": "matplotlib.pyplot",
        "scikit-learn": "sklearn.metrics",
        "numpy": "numpy",
        "gpytorch": "gpytorch",
        "scikit-learn": "sklearn.preprocessing"
    }

    # check_and_install_packages(required_packages)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # @title Run BO_AL
    set_random_seed(42)
    desired_value = 0.9 # coverage_proability
    num_initial_samples = 5 #00 #00 #
    nbatch = 5 #100 #00 #40 # 5 


    def random_init(lo, hi, num_initial_samples):
        return np.random.random_sample(num_initial_samples)*(hi-lo) + lo

    temperature_lo = 500
    temperature_hi = 3500
    temperature_init = random_init(temperature_lo, temperature_hi, num_initial_samples)

    lattice_strain_lo = 1.0
    lattice_strain_hi = 1.05
    lattice_strain_init = random_init(lattice_strain_lo, lattice_strain_hi, num_initial_samples)

    shear_strain_lo = 1.0
    shear_strain_hi = 1.05
    shear_strain_init = random_init(shear_strain_lo, shear_strain_hi, num_initial_samples)


    bounds = torch.tensor([[temperature_lo, lattice_strain_lo, shear_strain_lo], [temperature_hi, lattice_strain_hi, shear_strain_hi]], dtype=torch.float64).to(device)

    params_dict = []
    for ic in range(num_initial_samples): 
        params_dict.append({'Temp_K': temperature_init[ic], 'Strain': lattice_strain_init[ic], 'Shear_Strain': shear_strain_init[ic]})

###############################################
# # FOR INITIALIZING THE WORKFLOW
    # MD_PROCS = 1120
    from onlineMD import onlineMD
    init_md_data = onlineMD(candidate_parameters=params_dict, initial_parameters_file=os.path.abspath(sys.argv[1]))
    print (init_md_data['output'])
    print ('size of output : ',len(init_md_data['output']))
    Y_init = torch.tensor((np.array(init_md_data['output'])-desired_value)**2, dtype=torch.float64).unsqueeze(-1).to(device)
    print (Y_init)
    print ('size of Y_init : ', Y_init.shape)
    X_init = torch.tensor(np.array([temperature_init, lattice_strain_init, shear_strain_init]).T, dtype=torch.float64).to(device)
    print(X_init)
    print ('size of X_init : ', X_init.shape)

################################################
# FOR RESTARTING THE WORKFLOW

#    X_sample, Y_sample = restart_BO('sample_in_iteration2')
#    X_init = torch.tensor(X_sample, dtype=torch.float64).to(device)
#    Y_init = torch.tensor(Y_sample, dtype=torch.float64).unsqueeze(-1).to(device)
    
    # Run Bayesian Optimization with the original format of data
    results_df, cand_dict = bayesian_optimization(n_iterations=20, X_sample=X_init,\
                                          Y_sample=Y_init, bounds=bounds, desired_target=desired_value, initial_param_filepath=os.path.abspath('input_paramters.json'),batch_size=nbatch, restart=1)
    results_df.to_csv('optimization_results.csv', index=False)

    # Convert the Objective_Value column to a PyTorch tensor
    objective_values = torch.tensor(results_df.Objective_Value.values, dtype=torch.float64)

    # Find the index of the minimum difference to the desired value
    best_idx = torch.argmin(torch.abs(objective_values - torch.tensor(desired_value, dtype=torch.float64)))

    print("Optimization finished.")

    # Access the best location and value from the DataFrame
    best_location = results_df.iloc[best_idx.item(), 0:3].values
    best_value = results_df.iloc[best_idx.item()].Objective_Value

    print(f"Best location: {best_location}")
    print(f"Best value: {best_value}")



    
