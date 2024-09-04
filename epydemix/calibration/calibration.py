import numpy as np 
import pandas as pd 
from .calibration_results import CalibrationResults
from ..utils.utils import compute_quantiles, combine_simulation_outputs
from .metrics import *
from datetime import datetime, timedelta
from scipy import stats
from ..utils.abc_smc_utils import initialize_particles, default_perturbation_kernel, compute_covariance_matrix, compute_weights, compute_effective_sample_size, weighted_quantile
from multiprocess import Pool
from typing import Callable, Dict, Any, Optional, Tuple


def calibration(strategy: str, 
                simulation_function: Callable[[Dict[str, Any]], Any], 
                priors: Dict[str, Any], 
                parameters: Dict[str, Any], 
                data: Dict[str, Any], 
                Nsim: int = 100, 
                top_perc: float = 0.05, 
                tolerance: float = 0.1, 
                error_metric: Callable[[Dict[str, Any], Any], float] = rmse, 
                include_quantiles: bool = True, 
                dates_column: str = "simulation_dates", 
                perturbation_kernel: Callable = default_perturbation_kernel, 
                distance_function: Callable[[Dict[str, Any], Any], float] = rmse, 
                p_discrete_transition: float = 0.3,
                num_particles: int = 1000, 
                max_generations: int = 10, 
                tolerance_quantile: float = 0.50,
                minimum_tolerance: float = 0.15, 
                max_time: Optional[timedelta] = None,
                scaling_factor: float = 1.0,
                apply_bandwidth: bool = True,
                n_jobs: int = 4) -> CalibrationResults:
    """
    Unified calibration function to handle top percentage, ABC rejection, and ABC SMC methods.
    
    Args:
        strategy (str): The calibration strategy to use ("top_perc", "abc_rejection", or "abc_smc").
        simulation_function (Callable[[Dict[str, Any]], Any]): The function that runs the simulation, which takes the parameters dictionary as input.
        priors (Dict[str, Any]): A dictionary of prior distributions for the parameters.
        parameters (Dict[str, Any]): A dictionary of parameters used in the simulation.
        data (Dict[str, Any]): A dictionary containing the observed data.
        Nsim (int, optional): The number of simulation runs to perform (default is 100).
        top_perc (float, optional): Top percentage of simulations to select based on error metric (used in "top_perc").
        tolerance (float, optional): Error tolerance for rejection sampling (used in "abc_rejection").
        error_metric (Callable[[Dict[str, Any], Any], float], optional): The error metric function used to evaluate the simulations (default is rmse).
        include_quantiles (bool, optional): Whether to compute and include quantiles in the results (default is True).
        dates_column (str, optional): The key used for simulation dates in the parameters (default is "simulation_dates").
        
        Arguments for "abc_smc":
            perturbation_kernel (Callable, optional): Perturbation kernel for ABC SMC (default is default_perturbation_kernel).
            distance_function (Callable[[Dict[str, Any], Any], float], optional): The distance function used to measure the distance between observed and simulated data (default is rmse).
            p_discrete_transition (float, optional): The transition probability for discrete parameters (default is 0.3).
            num_particles (int, optional): Number of particles for ABC SMC (default is 1000).
            max_generations (int, optional): Maximum number of generations for ABC SMC (default is 10).
            tolerance_quantile (float, optional): Quantile to reduce the tolerance by in ABC SMC (default is 0.50).
            minimum_tolerance (float, optional): Minimum tolerance for ABC SMC (default is 0.15).
            max_time (Optional[timedelta], optional): Maximum time allowed for ABC SMC calibration (default is None).
            scaling_factor (float, optional): Scaling factor for ABC SMC covariance matrix (default is 1.0).
            apply_bandwidth (bool, optional): Whether to apply bandwidth scaling to covariance matrix (default is True).
            n_jobs (int, optional): Number of parallel jobs to run during ABC SMC calibration (default is 4).
    
    Returns:
        CalibrationResults: The results of the calibration, including posterior distributions and selected simulations.
    """
    if strategy == "top_perc":
        return calibration_top_perc(simulation_function, 
                                    priors, 
                                    parameters, 
                                    data, 
                                    top_perc=top_perc, 
                                    error_metric=error_metric, 
                                    Nsim=Nsim, 
                                    include_quantiles=include_quantiles, 
                                    dates_column=dates_column)
    
    elif strategy == "abc_rejection":
        return calibration_abc_rejection(simulation_function, 
                                         priors, 
                                         parameters, 
                                         data, 
                                         tolerance=tolerance, 
                                         error_metric=error_metric, 
                                         N_particles=Nsim, 
                                         include_quantiles=include_quantiles, 
                                         dates_column=dates_column)
    
    elif strategy == "abc_smc":
        return calibration_abc_smc(simulation_function, 
                                   priors, 
                                   parameters, 
                                   data, 
                                   perturbation_kernel=perturbation_kernel, 
                                   distance_function=distance_function, 
                                   p_discrete_transition=p_discrete_transition, 
                                   num_particles=num_particles, 
                                   max_generations=max_generations, 
                                   tolerance_quantile=tolerance_quantile, 
                                   minimum_tolerance=minimum_tolerance, 
                                   max_time=max_time, 
                                   include_quantiles=include_quantiles, 
                                   scaling_factor=scaling_factor, 
                                   apply_bandwidth=apply_bandwidth, 
                                   dates_column=dates_column, 
                                   n_jobs=n_jobs)
    
    else:
        raise ValueError(f"Unsupported calibration strategy: {strategy}")


def calibration_top_perc(simulation_function: Callable, 
                         priors: Dict[str, Any], 
                         parameters: Dict[str, Any], 
                         data: Dict[str, Any],
                         top_perc: float = 0.05,
                         error_metric: Callable[[Dict[str, Any], Any], float] = rmse,
                         Nsim: int = 100, 
                         include_quantiles: bool = True, 
                         dates_column: str = "simulation_dates") -> CalibrationResults:
    """
    Calibrates the model by selecting the top percentage of simulations based on the chosen error metric.

    Args:
        simulation_function (Callable[[Dict[str, Any]], Any]): The function that runs the simulation, taking a dictionary of parameters as input.
        priors (Dict[str, Any]): A dictionary of prior distributions for the parameters.
        parameters (Dict[str, Any]): A dictionary of parameters used in the simulation.
        data (Dict[str, Any]): A dictionary containing the observed data with a key "data" pointing to observations.
        top_perc (float, optional): The top percentage of simulations to select based on the error metric (default is 0.05).
        error_metric (Callable[[Dict[str, Any], Any], float], optional): The error metric function used to evaluate the simulations (default is rmse).
        Nsim (int, optional): The number of simulation runs to perform (default is 100).
        include_quantiles (bool, optional): Whether to compute and include quantiles in the results (default is True).
        dates_column (str, optional): The key used for simulation dates in the parameters dictionary (default is "simulation_dates").

    Returns:
        CalibrationResults: An object containing the results of the calibration, including the posterior distribution, selected trajectories, and quantiles.
    """
    
    simulations, errors = [], []
    sampled_params = {p: [] for p in priors.keys()}
    simulation_params = parameters.copy()
    for n in range(Nsim):

        # sample
        for param, distr in priors.items():
            rv = distr.rvs()
            sampled_params[param].append(rv)
            simulation_params[param] = rv

        results = simulation_function(simulation_params)
        simulations.append(results)
        errors.append(error_metric(data=data, simulation=results))

    # compute error threshold
    err_threshold = np.quantile(errors, q=top_perc)
    idxs = np.argwhere(np.array(errors) <= err_threshold).ravel()

    # select runs 
    selected_simulations = np.array(simulations)[idxs]

    # format simulation results
    selected_simulations_formatted = {}
    for res in selected_simulations:
        selected_simulations_formatted = combine_simulation_outputs(selected_simulations_formatted, res)

    # select parameters
    selected_params = {p: np.array(arr)[idxs] for p, arr in sampled_params.items()}

    # format results 
    results = CalibrationResults()
    results.set_calibration_strategy("top_perc")
    results.set_posterior_distribution(pd.DataFrame(data=selected_params))
    results.set_selected_trajectories(np.array([el["data"] for el in selected_simulations]))
    results.set_data(data)
    results.set_priors(priors)
    results.set_calibration_params({"top_perc": top_perc, 
                                    "error_metric": error_metric, 
                                    "Nsim": Nsim})
    results.set_error_distribution(np.array(errors)[idxs])

    if include_quantiles: 
        # compute quantiles 
        selected_simulations_quantiles = compute_quantiles(selected_simulations_formatted, simulation_dates=simulation_params[dates_column])
        results.set_selected_quantiles(selected_simulations_quantiles)

    return results


def perturb_and_simulate(args):
    """
    Perturb a particle, simulate data, and compute the distance.

    Args:
        args (Tuple): A tuple containing the following elements:
            - particle_index (int): Index of the particle to perturb.
            - particles (np.ndarray): Array of particles.
            - perturbation_kernel (Callable): A function that perturbs a particle.
            - continuous_params (list): List of continuous parameters.
            - discrete_params (list): List of discrete parameters.
            - cov_matrix (np.ndarray): Covariance matrix for perturbing particles.
            - priors (Dict[str, Any]): Dictionary of priors for the parameters.
            - p_discrete_transition (float): Probability of transitioning discrete parameters.
            - parameters (Dict[str, Any]): Model parameters to be passed to the simulation.
            - model (Callable): A model function that simulates data.
            - observed_data (Dict[str, Any]): The observed data to compare against.
            - distance_function (Callable): A function that calculates the distance between observed and simulated data.

    Returns:
        Tuple[Dict[str, Any], float, Dict[str, Any]]: A tuple containing:
            - perturbed particle (Dict[str, Any]): The perturbed particle.
            - distance (float): The distance between the observed and simulated data.
            - simulated_data (Dict[str, Any]): The data simulated from the perturbed particle.
    """
    particle_index, particles, perturbation_kernel, continuous_params, discrete_params, cov_matrix, priors, p_discrete_transition, parameters, model, observed_data, distance_function = args
    
    # Sample and Perturb particle
    i = particle_index
    particle = perturbation_kernel(particles[i], continuous_params, discrete_params, cov_matrix, priors, p_discrete_transition)

    # Simulate data from perturbed particle
    full_params = {}
    full_params.update(particle)
    full_params.update(parameters)
    simulated_data = model(full_params)

    # Calculate distance
    distance = distance_function(observed_data, simulated_data)

    return particle, distance, simulated_data


def calibration_abc_smc(model: Callable, 
                        priors: Dict[str, Any], 
                        parameters: Dict[str, Any], 
                        observed_data: Dict[str, Any], 
                        perturbation_kernel: Callable = default_perturbation_kernel, 
                        distance_function: Callable[[Dict[str, Any], Dict[str, Any]], float] = rmse, 
                        p_discrete_transition: float = 0.3,
                        num_particles: int = 1000, 
                        max_generations: int = 10, 
                        tolerance_quantile: float = 0.50,
                        minimum_tolerance: float = 0.15, 
                        max_time: Optional[timedelta] = None,
                        include_quantiles: bool = True, 
                        scaling_factor: float = 1.0,
                        apply_bandwidth: bool = True,
                        dates_column: str = "simulation_dates", 
                        n_jobs: int = 4) -> CalibrationResults:
    """
    Calibrates the model using Approximate Bayesian Computation Sequential Monte Carlo (ABC-SMC).

    Args:
        model (Callable): The model function that takes in parameters and returns simulated data.
        priors (Dict[str, Any]): A dictionary of prior distributions for the parameters.
        parameters (Dict[str, Any]): A dictionary of model parameters.
        observed_data (Dict[str, Any]): The observed data to compare against.
        perturbation_kernel (Callable, optional): A kernel function for perturbing particles (default is default_perturbation_kernel).
        distance_function (Callable[[Dict[str, Any], Dict[str, Any]], float], optional): Function to compute the distance between observed and simulated data (default is rmse).
        p_discrete_transition (float, optional): Probability of transitioning discrete parameters (default is 0.3).
        num_particles (int, optional): Number of particles in the population (default is 1000).
        max_generations (int, optional): Maximum number of generations for ABC SMC (default is 10).
        tolerance_quantile (float, optional): Quantile used to reduce tolerance at each generation (default is 0.50).
        minimum_tolerance (float, optional): Minimum tolerance to stop the ABC SMC process (default is 0.15).
        max_time (Optional[timedelta], optional): Maximum allowed time for the calibration (default is None).
        include_quantiles (bool, optional): Whether to compute quantiles from the results (default is True).
        scaling_factor (float, optional): Scaling factor for the covariance matrix in perturbation (default is 1.0).
        apply_bandwidth (bool, optional): Whether to apply bandwidth scaling to the covariance matrix (default is True).
        dates_column (str, optional): The key for simulation dates in the parameters (default is "simulation_dates").
        n_jobs (int, optional): Number of parallel jobs for the ABC SMC process (default is 4).

    Returns:
        CalibrationResults: Object containing the results of the ABC SMC calibration, including posterior distributions, trajectories, and quantiles.
    """
    
    start_time = datetime.now()

    # Get continuous and discrete parameters
    continuous_params, discrete_params = [], []
    for param in priors:
        if isinstance(priors[param].dist, stats.rv_continuous):
            continuous_params.append(param)
        elif isinstance(priors[param].dist, stats.rv_discrete):
            discrete_params.append(param)

    particles, weights, tolerance = initialize_particles(priors, num_particles)

    for generation in range(max_generations):
        start_generation_time = datetime.now()

        # Compute covariance matrix
        cov_matrix = compute_covariance_matrix(particles, continuous_params, weights, scaling_factor, apply_bandwidth)
        
        new_particles = []
        distances = []
        simulated_points = []

        total_accepted, total_simulations = 0, 0

        # Parallel processing setup
        pool = Pool(n_jobs)

        while total_accepted < num_particles:
            args_list = [
                (
                    np.random.choice(range(num_particles), p=weights), 
                    particles, 
                    perturbation_kernel, 
                    continuous_params, 
                    discrete_params, 
                    cov_matrix, 
                    priors, 
                    p_discrete_transition, 
                    parameters, 
                    model, 
                    observed_data, 
                    distance_function
                )
                for _ in range(n_jobs)
            ]
            
            results = pool.map(perturb_and_simulate, args_list)
            
            for particle, distance, simulated_data in results:
                total_simulations += 1
                if distance < tolerance:
                    new_particles.append(particle)
                    distances.append(distance)
                    total_accepted += 1
                    simulated_points.append(simulated_data["data"])
                    if total_accepted >= num_particles:
                        break

        pool.close()
        pool.join()

        # Update particles and weights
        weights = compute_weights(new_particles, particles, weights, priors, cov_matrix, continuous_params, discrete_params, p_discrete_transition)
        particles = new_particles

        # Update tolerance for the next generation
        #tolerance = np.quantile(distances, tolerance_quantile)
        tolerance = weighted_quantile(distances, weights, tolerance_quantile)

        # Print generation information
        ESS = compute_effective_sample_size(weights)
        end_generation_time = datetime.now()
        elapsed_time = end_generation_time - start_generation_time
        formatted_time = f"{elapsed_time.seconds // 3600:02}:{(elapsed_time.seconds % 3600) // 60:02}:{elapsed_time.seconds % 60:02}"
        print(f"Generation {generation + 1}: Tolerance = {tolerance}, Accepted Particles = {len(new_particles)}/{total_simulations}, Time = {formatted_time}, ESS = {ESS}")

        # Check if the tolerance is below the minimum
        if tolerance < minimum_tolerance:
            print(f"Minimum tolerance reached at generation {generation + 1}")
            break

        # Check if the walltime has been reached
        if max_time is not None and datetime.now() - start_time > max_time:
            print(f"Maximum time reached at generation {generation + 1}")
            break

    # Format output
    results = CalibrationResults()
    results.set_calibration_strategy("abc_smc")

    posterior_dict = {param: [sample[param] for sample in particles] for param in priors.keys()}
    posterior_dict["weights"] = weights
    results.set_posterior_distribution(pd.DataFrame(posterior_dict))
    results.set_selected_trajectories(np.array(simulated_points))
    results.set_data(observed_data)
    results.set_priors(priors)
    results.set_calibration_params({
        "num_particles": num_particles,
        "minimum_tolerance": minimum_tolerance,
        "distance_function": distance_function,
        "max_generations": max_generations,
        "max_time": max_time
    })
    results.set_error_distribution(distances)
    
    if include_quantiles: 
        # Compute quantiles 
        selected_simulations_quantiles = compute_quantiles({"data": simulated_points}, simulation_dates=parameters[dates_column])
        results.set_selected_quantiles(selected_simulations_quantiles)
    
    return results


def calibration_abc_rejection(simulation_function: Callable, 
                              priors: Dict[str, Any], 
                              parameters: Dict[str, Any], 
                              data: Dict[str, Any],
                              tolerance: float,
                              error_metric: Callable[[Dict[str, Any], Any], float] = rmse,
                              N_particles: int = 100,
                              include_quantiles: bool = True, 
                              dates_column: str = "simulation_dates") -> CalibrationResults:
    """
    Calibrates the model using Approximate Bayesian Computation (ABC) rejection sampling.

    Args:
        simulation_function (Callable): The function that runs the simulation, taking a dictionary of parameters as input.
        priors (Dict[str, Any]): A dictionary of prior distributions for the parameters.
        parameters (Dict[str, Any]): A dictionary of model parameters.
        data (Dict[str, Any]): The observed data.
        tolerance (float): The error tolerance for accepting a simulation.
        error_metric (Callable[[Dict[str, Any], Any], float], optional): The error metric function used to evaluate the simulations (default is rmse).
        N_particles (int, optional): The number of particles to accept based on the tolerance (default is 100).
        include_quantiles (bool, optional): Whether to compute and include quantiles in the results (default is True).
        dates_column (str, optional): The key used for simulation dates in the parameters dictionary (default is "simulation_dates").

    Returns:
        CalibrationResults: An object containing the results of the calibration, including the posterior distribution, selected trajectories, and quantiles.
    """
            
    simulation_params = parameters.copy()
    accepted_params = {p: [] for p in priors.keys()}
    accepted_errors, accepted_simulations = [], [], []

    while len(accepted_errors) < N_particles:

        # sample
        sampled_params = {}
        for param, distr in priors.items():
            rv = distr.rvs()
            sampled_params[param] = rv
            simulation_params[param] = rv

        # simulate
        results = simulation_function(simulation_params)
        error = error_metric(data=data, simulation=results)
        
        if error < tolerance: 
            for p in accepted_params.keys(): 
                accepted_params[p].append(sampled_params[p])
            accepted_errors.append(error)
            accepted_simulations.append(results)

    # format simulation results
    accepted_simulations_formatted = {}
    for res in accepted_simulations:
        accepted_simulations_formatted = combine_simulation_outputs(accepted_simulations_formatted, res)

    # format results 
    results = CalibrationResults()
    results.set_calibration_strategy("abc_rejection")
    results.set_posterior_distribution(pd.DataFrame(data=accepted_params))
    results.set_selected_trajectories(np.array([el["data"] for el in accepted_simulations]))
    results.set_data(data)
    results.set_priors(priors)
    results.set_calibration_params({"tolerance": tolerance, 
                                    "error_metric": error_metric, 
                                    "N_particles": N_particles})
    results.set_error_distribution(np.array(accepted_errors))

    if include_quantiles: 
        # compute quantiles 
        selected_simulations_quantiles = compute_quantiles(accepted_simulations_formatted, simulation_dates=simulation_params[dates_column])
        results.set_selected_quantiles(selected_simulations_quantiles)

    return results


def run_projections(simulation_function: Callable, 
                    calibration_results: CalibrationResults, 
                    parameters: Dict[str, Any], 
                    iterations: int = 100, 
                    include_quantiles: bool = True, 
                    dates_column: str = "simulation_dates") -> Dict[str, Any]:
    """
    Runs projections based on the posterior distribution obtained from calibration.

    Args:
        simulation_function (Callable[[Dict[str, Any]], Any]): The function that runs the simulation, taking a dictionary of parameters as input.
        calibration_results (CalibrationResults): The results from a calibration process, containing the posterior distribution.
        parameters (Dict[str, Any]): A dictionary of parameters used for the projections.
        iterations (int, optional): The number of projection iterations to run (default is 100).
        include_quantiles (bool, optional): Whether to compute and return quantiles from the projections (default is True).
        dates_column (str, optional): The key used for simulation dates in the parameters dictionary (default is "simulation_dates").

    Returns:
        Dict[str, Any]: A dictionary containing the formatted projections, including quantiles if requested.
    """


    # get posterior distribution
    posterior_distribution = calibration_results.get_posterior_distribution()

    # iterate
    simulations = []
    for n in range(iterations): 
        # sample from posterior
        posterior_sample = posterior_distribution.iloc[np.random.randint(0, len(posterior_distribution))]
        
        # prepare and run simulation
        parameters.update(posterior_sample)
        result_projections = simulation_function(parameters)
        simulations.append(result_projections)

    # format simulation results
    selected_simulations_formatted = {}
    for res in simulations:
        selected_simulations_formatted = combine_simulation_outputs(selected_simulations_formatted, res)

    if include_quantiles:
        selected_simulations_quantiles = compute_quantiles(selected_simulations_formatted, simulation_dates=parameters[dates_column])
        return selected_simulations_quantiles
    
    else: 
        return selected_simulations_formatted