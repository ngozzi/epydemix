import numpy as np 
import os
import pandas as pd 
from .calibration_results import CalibrationResults
from .metrics import *
from datetime import datetime, timedelta
from ..utils.abc_smc_utils import DefaultPerturbationContinuous, DefaultPerturbationDiscrete, sample_prior
from ..utils.utils import combine_simulation_outputs
from typing import Callable, Dict, Any, Optional, Tuple

def wrapper_function(func, *args, **kwargs):
    """
    Wrapper function to ensure the passed function returns a dictionary.
    
    Parameters:
    - func: The function to be called.
    - *args, **kwargs: Arguments and keyword arguments for the function.

    Returns:
    - The output of the passed function if it's a dictionary.

    Raises:
    - ValueError: If the function does not return a dictionary.
    """
    result = func(*args, **kwargs)
    if not isinstance(result, dict):
        raise ValueError(f"The function {func.__name__} must return a dictionary, but it returned {type(result)}.")
    return result


def calibrate(strategy: str, 
              simulation_function: Callable[[Dict[str, Any]], Any], 
              priors: Dict[str, Any], 
              parameters: Dict[str, Any], 
              observed_data: Dict[str, Any], 
              top_fraction: float = 0.05, 
              Nsim : int = 100, 
              epsilon: float = 0.1,
              num_particles: int = 1000, 
              distance_function: Callable[[Dict[str, Any], Any], float] = rmse, 
              max_generations: int = 10, 
              epsilon_quantile_level: float = 0.50,
              minimum_epsilon: float = 0.15, 
              max_time: Optional[timedelta] = None,   
              total_simulations_budget: Optional[int] = None,             
              epsilon_schedule = None, 
              perturbations = None) -> CalibrationResults:
    """
    Unified calibration function to handle top fraction, ABC rejection, and ABC SMC methods.
    
    Args:
        strategy (str): The calibration strategy to use ("abc_top_fraction", "abc_rejection", or "abc_smc").
        simulation_function (Callable[[Dict[str, Any]], Any]): The function that runs the simulation, which takes the parameters dictionary as input.
        priors (Dict[str, Any]): A dictionary of prior distributions for the parameters.
        parameters (Dict[str, Any]): A dictionary of parameters used in the simulation.
        observed_data (Dict[str, Any]): A dictionary containing the observed data. Key must be "data".
        top_fraction (float, optional): Top fraction of simulations to select based on error metric (used in "abc_top_fraction").
        Nsim (int, optional): The number of simulation runs to perform in "abc_top_fraction" approach (default is 100).
        epsilon (float, optional): Error tolerance for rejection sampling (used in "abc_rejection").
        num_particles (int, optional): Number of particles for "abc_rejection" and "abc_smc" approach (default is 1000).
        distance_function (Callable[[Dict[str, Any], Any], float], optional): The error metric function used to evaluate the simulations (default is rmse).
        max_generations (int, optional): Maximum number of generations for ABC SMC (used in "abc_smc" approach).
        epsilon_quantile_level (float, optional): Quantile level to set next generation's epsilon (used in "abc_smc" approach).
        minimum_epsilon (float, optional): Minimum tolerance for ABC SMC (used in "abc_smc" approach).
        max_time (Optional[timedelta], optional): Maximum time allowed for ABC SMC calibration (used in "abc_smc" approach).
        total_simulations_budget (Optional[int], optional): Maximum number of simulations allowed for ABC SMC calibration (used in "abc_smc" approach).
        epsilon_schedule (Optional[List[float]], optional): List of epsilon thresholds for each generation (used in "abc_smc" approach).
        perturbations (Optional[Dict[str, Any]], optional): Dictionary of parameter_name -> Perturbation objects (used in "abc_smc" approach).
        
    Returns:
        CalibrationResults: The results of the calibration, including posterior distributions and selected simulations.
    """
    
    if strategy == "abc_top_fraction":
        return calibration_abc_top_fraction(simulation_function=simulation_function, 
                                    priors=priors, 
                                    parameters=parameters, 
                                    observed_data=observed_data, 
                                    top_fraction=top_fraction, 
                                    distance_function=distance_function, 
                                    Nsim=Nsim)
    
    elif strategy == "abc_rejection":
        return calibration_abc_rejection(simulation_function, 
                                         priors, 
                                         parameters, 
                                         observed_data, 
                                         epsilon=epsilon, 
                                         distance_function=distance_function, 
                                         num_particles=num_particles, 
                                         max_time=max_time,
                                         total_simulations_budget=total_simulations_budget)
    
    elif strategy == "abc_smc":
        return calibration_abc_smc(observed_data=observed_data, 
                                    num_particles=num_particles, 
                                    num_generations=max_generations, 
                                    epsilon_schedule=epsilon_schedule, 
                                    priors=priors, 
                                    epsilon_quantile_level=epsilon_quantile_level, 
                                    perturbations=perturbations, 
                                    distance_function=distance_function,
                                    simulate_model=simulation_function, 
                                    user_params=parameters, 
                                    minimum_epsilon=minimum_epsilon, 
                                    max_time=max_time, 
                                    total_simulations_budget=total_simulations_budget)
                            
    else:
        raise ValueError(f"Unsupported calibration strategy: {strategy}")


def calibration_abc_top_fraction(simulation_function: Callable, 
                                 priors: Dict[str, Any], 
                                 parameters: Dict[str, Any], 
                                 observed_data: Dict[str, Any],
                                 top_fraction: float = 0.05,
                                 distance_function: Callable[[Dict[str, Any], Any], float] = rmse,
                                 Nsim: int = 100) -> CalibrationResults:
    """
    Calibrates the model by selecting the top percentage of simulations based on the chosen error metric.

    Args:
        simulation_function (Callable[[Dict[str, Any]], Any]): The function that runs the simulation, taking a dictionary of parameters as input.
        priors (Dict[str, Any]): A dictionary of prior distributions for the parameters.
        parameters (Dict[str, Any]): A dictionary of parameters used in the simulation.
        observed_data (Dict[str, Any]): A dictionary containing the observed data with a key "data" pointing to observations.
        top_fraction (float, optional): The top fraction of simulations to select based on the error metric (default is 0.05).
        distance_function (Callable[[Dict[str, Any], Any], float], optional): The error metric function used to evaluate the simulations (default is rmse).
        Nsim (int, optional): The number of simulation runs to perform (default is 100).

    Returns:
        CalibrationResults: An object containing the results of the calibration, including the posterior distribution, selected trajectories, and quantiles.
    """
    
    simulations, distances = [], []
    sampled_params = {p: [] for p in priors.keys()}

    # Parameter names for consistent indexing
    param_names = list(priors.keys())

    # Run simulations
    for _ in range(Nsim):
        params = sample_prior(priors, param_names)
        full_params = {**parameters, **dict(zip(param_names, params))}
        results = wrapper_function(simulation_function, full_params)
        distance = distance_function(data=observed_data, simulation=results)
        simulations.append(results)
        distances.append(distance)
        for i, p in enumerate(param_names):
            sampled_params[p].append(params[i])

    # Compute distance threshold
    distance_threshold = np.quantile(distances, q=top_fraction)
    idxs = np.argwhere(np.array(distances) <= distance_threshold).ravel()

    # select runs and parameters
    selected_simulations = np.array(simulations)[idxs]
    selected_params = {p: np.array(arr)[idxs] for p, arr in sampled_params.items()}

    # format results 
    results = CalibrationResults(
        calibration_strategy="abc_top_fraction",
        posterior_distributions={0: pd.DataFrame(data=selected_params)},
        selected_trajectories={0: selected_simulations},
        distances={0: np.array(distances)[idxs]},
        observed_data=observed_data,
        priors=priors,
        calibration_params={
            "top_fraction": top_fraction, 
            "distance_function": distance_function, 
            "Nsim": Nsim
        }
    )
    return results


def calibration_abc_smc(observed_data, 
            num_particles, 
            num_generations, 
            epsilon_schedule=None, 
            priors=None, 
            epsilon_quantile_level=0.5, 
            perturbations=None, 
            distance_function=rmse,
            simulate_model=None, 
            user_params=None, 
            minimum_epsilon=None, 
            max_time=None, 
            total_simulations_budget=None):
    """
    Implements the ABC-SMC algorithm with a customizable simulation model.

    Args:
        observed_data: the observed dataset
        num_particles: number of particles in each generation
        num_generations: number of generations to run
        epsilon_schedule: list of epsilon thresholds for each generation (optional)
        priors: dictionary of parameter_name -> scipy.stats prior distribution
        epsilon_quantile_level: quantile level to set next generation's epsilon (default 0.5)
        perturbations: dictionary of parameter_name -> Perturbation objects (optional)
        distance_function: function that computes the distance between observed and simulated data
        simulate_model: function that simulates data given a dictionary of parameters
        user_params: dictionary of fixed parameters provided by the user (optional)
        minimum_epsilon: minimum epsilon threshold (optional)
        max_time: maximum time allowed for calibration (optional)
        total_simulations_budget: maximum number of simulations allowed for calibration (optional)
    
    Returns: 
        CalibrationResults: An object containing the results of the calibration, including the posterior distribution, selected trajectories, and quantiles.
    """
    if simulate_model is None:
        raise ValueError("A simulation model function must be provided.")
    
    if user_params is None:
        user_params = {}

    # Define the results object 
    results = CalibrationResults(calibration_strategy="abc_smc") 

    particles, weights, distances, simulations = [], [], [], []

    # Parameter names for consistent indexing
    param_names = list(priors.keys())

    # Separate parameters into continuous and discrete
    continuous_params = [name for name in param_names if hasattr(priors[name], 'pdf')]
    discrete_params = [name for name in param_names if name not in continuous_params]

    # Choose perturbation functions
    if perturbations is None:
        perturbations = {
            param: (DefaultPerturbationContinuous(param) if param in continuous_params else DefaultPerturbationDiscrete(param, priors[param]))
            for param in param_names
        }
    else: 
        for param in param_names:
            if param not in perturbations.keys():
                if param in continuous_params:
                    perturbations[param] = DefaultPerturbationContinuous(param)
                else:
                    perturbations[param] = DefaultPerturbationDiscrete(param, priors[param])

    # Initialize the first generation 
    for _ in range(num_particles):
        params = sample_prior(priors, param_names)
        full_params = {**user_params, **dict(zip(param_names, params))}
        simulated_data = wrapper_function(simulate_model, full_params)
        dist = distance_function(data=observed_data, simulation=simulated_data)
        particles.append(params)
        weights.append(1.0 / num_particles)
        distances.append(dist)
        simulations.append(simulated_data)

    particles = np.array(particles)
    weights = np.array(weights)
    distances = np.array(distances)
    simulations = np.array(simulations)

    # Set the results for the first generation
    results.posterior_distributions[0] = pd.DataFrame(data={param_names[i]: particles[:, i] for i in range(len(param_names))})
    results.distances[0] = distances
    results.weights[0] = weights
    results.selected_trajectories[0] = simulations

    # Sequential generations
    start_time = datetime.now()
    n_simulations = 0 # Counter for total number of simulations
    for gen in range(0, num_generations):
        start_generation_time = datetime.now()

        # Compute epsilon for the generation
        if epsilon_schedule is None:
            epsilon = np.quantile(distances, epsilon_quantile_level)
        else:
            epsilon = epsilon_schedule[gen]

        print(f"Running generation {gen + 1}, epsilon: {epsilon}")

        # Check if minimum tolerance is reached
        if minimum_epsilon is not None and epsilon < minimum_epsilon:
            print(f"Minimum epsilon reached at generation {gen + 1}")
            break

        # Check if the walltime has been reached
        if max_time is not None and datetime.now() - start_time > max_time:
            print(f"Maximum time reached at generation {gen + 1}")
            break

        # Check if the total simulations budget has been reached
        if total_simulations_budget is not None and n_simulations > total_simulations_budget:
            print(f"Total simulations budget reached at generation {gen + 1}")
            break

        # Update perturbations based on particles and weights
        for perturbation in perturbations.values():
            perturbation.update(particles, weights, param_names)

        new_particles, new_weights, new_distances, new_simulations = [], [], [], []
        for _ in range(num_particles):
            while True:
                # Resample a particle based on weights
                index = np.random.choice(len(particles), p=weights / weights.sum())
                candidate_params = particles[index]

                # Propose new parameters (perturbation kernel)
                perturbed_params = [
                    perturbations[param_names[i]].propose(candidate_params[i]) for i in range(len(param_names))
                ]
                full_params = {**user_params, **dict(zip(param_names, perturbed_params))}

                # Check if perturbed parameters have prior probability > 0
                prior_probabilities = [
                  priors[param].pdf(perturbed_params[i]) if param in continuous_params
                  else priors[param].pmf(perturbed_params[i])
                  for i, param in enumerate(param_names)
                ]
                if all(prob > 0 for prob in prior_probabilities):
                    simulated_data = wrapper_function(simulate_model, full_params)
                    dist = distance_function(data=observed_data, simulation=simulated_data)
                    n_simulations += 1

                    if dist < epsilon:
                        new_particles.append(perturbed_params)
                        weight_numerator = np.prod([
                          priors[param].pdf(perturbed_params[i]) if param in continuous_params
                          else priors[param].pmf(perturbed_params[i])
                          for i, param in enumerate(param_names)
                        ])
                        weight_denominator = np.sum([
                            weights[j] * np.prod([
                                perturbations[param_names[i]].pdf(
                                    perturbed_params[i], particles[j][i]
                                ) for i in range(len(param_names))
                            ]) for j in range(len(particles))
                        ])
                        new_weights.append(weight_numerator / weight_denominator)
                        new_distances.append(dist)
                        new_simulations.append(simulated_data)
                        break

        # Normalize weights
        new_weights = np.array(new_weights)
        new_weights /= new_weights.sum()

        particles = np.array(new_particles)
        weights = new_weights
        distances = np.array(new_distances)
        simulations = np.array(new_simulations)

        # Set the results for the generation
        results.posterior_distributions[gen + 1] = pd.DataFrame(data={param_names[i]: particles[:, i] for i in range(len(param_names))})
        results.distances[gen + 1] = distances
        results.weights[gen + 1] = weights
        results.selected_trajectories[gen + 1] = simulations

        # Print generation information
        end_generation_time = datetime.now()
        elapsed_time = end_generation_time - start_generation_time
        formatted_time = f"{elapsed_time.seconds // 3600:02}:{(elapsed_time.seconds % 3600) // 60:02}:{elapsed_time.seconds % 60:02}"
        print(f"\tElapsed time: {formatted_time}") 

    # Complete the results object
    results.observed_data = observed_data
    results.priors = priors
    results.calibration_params = {
        "num_particles": num_particles,
        "minimum_epsilon": minimum_epsilon,
        "distance_function": distance_function,
        "num_generations": num_generations,
        "max_time": max_time, 
        "total_simulations_budget": total_simulations_budget, 
        "epsilon_schedule": epsilon_schedule,
        "perturbations": perturbations
    }

    return results


def calibration_abc_rejection(simulation_function: Callable, 
                              priors: Dict[str, Any], 
                              parameters: Dict[str, Any], 
                              observed_data: Dict[str, Any],
                              epsilon: float,
                              distance_function: Callable[[Dict[str, Any], Any], float] = rmse,
                              num_particles: int = 100, 
                              max_time=None, 
                              total_simulations_budget=None) -> CalibrationResults:
    """
    Calibrates the model using Approximate Bayesian Computation (ABC) rejection sampling.

    Args:
        simulation_function (Callable): The function that runs the simulation, taking a dictionary of parameters as input.
        priors (Dict[str, Any]): A dictionary of prior distributions for the parameters.
        parameters (Dict[str, Any]): A dictionary of model parameters.
        observed_data (Dict[str, Any]): The observed data.
        epsilon (float): The error tolerance for accepting a simulation.
        distance_function (Callable[[Dict[str, Any], Any], float], optional): The error metric function used to evaluate the simulations (default is rmse).
        num_particles (int, optional): The number of particles to accept based on the tolerance (default is 100).
        max_time: maximum time allowed for calibration (optional)
        total_simulations_budget: maximum number of simulations allowed for calibration (optional)

    Returns:
        CalibrationResults: An object containing the results of the calibration, including the posterior distribution, selected trajectories, and quantiles.
    """

    simulations, distances = [], []
    sampled_params = {p: [] for p in priors.keys()}

    # Parameter names for consistent indexing
    param_names = list(priors.keys())

    start_time = datetime.now()
    n_simulations = 0 # Counter for total number of simulations
    while len(distances) < num_particles: 
        params = sample_prior(priors, param_names)
        full_params = {**parameters, **dict(zip(param_names, params))}
        results = wrapper_function(simulation_function, full_params)
        distance = distance_function(data=observed_data, simulation=results)
        if distance < epsilon:
            simulations.append(results)
            distances.append(distance)
            for i, p in enumerate(param_names):
                sampled_params[p].append(params[i])

        n_simulations += 1

        # Check if the walltime has been reached
        if max_time is not None and datetime.now() - start_time > max_time:
            print(f"Maximum time reached")
            break

        # Check if the total simulations budget has been reached
        if total_simulations_budget is not None and n_simulations > total_simulations_budget:
            print(f"Total simulations budget reached")
            break

    # format results 
    results = CalibrationResults(
        calibration_strategy="abc_rejection",
        posterior_distributions={0: pd.DataFrame(data=sampled_params)},
        selected_trajectories={0: simulations},
        distances={0: np.array(distances)},
        observed_data=observed_data,
        priors=priors,
        calibration_params={
            "epsilon": epsilon,
            "distance_function": distance_function, 
            "num_particles": num_particles, 
            "max_time": max_time,
            "total_simulations_budget": total_simulations_budget
        }
    )
    
    return results


def run_projections(simulation_function: Callable, 
                    calibration_results: CalibrationResults, 
                    parameters: Dict[str, Any],
                    iterations: int = 100) -> Tuple[Dict[str, Any], pd.DataFrame]: 
    """
    Runs projections based on the posterior distribution obtained from calibration.

    Args:
        simulation_function (Callable[[Dict[str, Any]], Any]): The function that runs the simulation, taking a dictionary of parameters as input.
        calibration_results (CalibrationResults): The results from a calibration process, containing the posterior distribution.
        parameters (Dict[str, Any]): A dictionary of parameters used for the projections.
        iterations (int, optional): The number of projection iterations to run (default is 100).

    Returns:
        Tuple[Dict[str, Any], pd.DataFrame]: A dictionary containing the formatted projections and a DataFrame containing the posterior samples used for each iteration.
    """
        
    # get posterior distribution
    posterior_distribution = calibration_results.get_posterior_distribution()
    
    # run projections and store results
    projections, posterior_samples = [], {}
    for n in range(iterations): 
        # sample from posterior
        posterior_sample = posterior_distribution.iloc[np.random.randint(0, len(posterior_distribution))]

        for k in posterior_sample.keys():
            if k not in posterior_samples.keys():
                posterior_samples[k] = []
            posterior_samples[k].append(posterior_sample[k])
        
        # prepare and run simulation
        parameters.update(posterior_sample)
        result_projections = simulation_function(parameters)
        projections.append(result_projections)

    # combine results
    projections_formatted = combine_simulation_outputs(projections)

    return projections_formatted, pd.DataFrame(posterior_samples)