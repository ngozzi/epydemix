import numpy as np 
import pandas as pd 
from .calibration_results import CalibrationResults
from .utils import compute_quantiles, combine_simulation_outputs
from .metrics import *
import pyabc
from datetime import timedelta
import uuid
import os 
import tempfile


def calibration_top_perc(simulation_function, 
                         priors, 
                         parameters, 
                         data,
                         top_perc=0.05,
                         error_metric=rmse,
                         Nsim=100, 
                         include_quantiles=True, 
                         dates_column="simulation_dates"): 
            
    """
    Calibrates the model by selecting the top percentage of simulations based on the chosen error metric.

    Parameters:
    -----------
        simulation_function (callable): The function that runs the simulation, which takes the parameters dictionary as input.
        priors (dict): A dictionary of prior distributions for the parameters.
        parameters (dict): A dictionary of parameters used in the simulation.
        data (dict): A dictionary containing the observed data with a key "data" pointing to an array of observations.
        top_perc (float, optional): The top percentage of simulations to select based on the error metric (default is 0.05).
        error_metric (callable, optional): The error metric function used to evaluate the simulations (default is rmse).
        Nsim (int, optional): The number of simulation runs to perform (default is 100).

    Returns:
    --------
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


def calibration_abc_smc(simulation_function, 
                         priors, 
                         parameters, 
                         data,
                         error_metric=rmse,
                         transitions : pyabc.AggregatedTransition = None,
                         max_walltime : timedelta = None,
                         population_size : int = 1000,
                         minimum_epsilon : float = 0.15, 
                         max_nr_populations : int = 10, 
                         filename : str = '', 
                         include_quantiles : bool = True,
                         dates_column="simulation_dates",
                         run_id = None, 
                         db = None): 
    
    """
    Performs Approximate Bayesian Computation with Sequential Monte Carlo (ABC-SMC) calibration.

    Parameters:
    -----------
        simulation_function (callable): The function that runs the simulation, which takes the parameters dictionary as input.
        priors (dict): A dictionary of prior distributions for the parameters.
        parameters (dict): A dictionary of fixed parameters used in the simulation.
        data (dict): A dictionary containing the observed data with a key "data" pointing to an array of observations.
        error_metric (callable, optional): The error metric function used to evaluate the simulations (default is rmse).
        transitions (pyabc.AggregatedTransition, optional): Transition kernel for the ABC-SMC algorithm (default is None).
        max_walltime (timedelta, optional): The maximum walltime for the ABC-SMC run (default is None).
        population_size (int, optional): The size of the population for each ABC-SMC generation (default is 1000).
        minimum_epsilon (float, optional): The minimum value of epsilon for the ABC-SMC algorithm (default is 0.15).
        max_nr_populations (int, optional): The maximum number of populations for the ABC-SMC algorithm (default is 10).
        filename (str, optional): The filename for storing the results (default is '').
        run_id (optional): The run ID for continuing a previous ABC-SMC run (default is None).
        db (optional): The database for storing the ABC-SMC results (default is None).

    Returns:
    --------
        CalibrationResults: An object containing the results of the calibration, including the posterior distribution, selected trajectories, and quantiles.
    """
    

    def model(p): 
        return {'data': simulation_function({**p, **parameters})['data']}

    if filename == '':
        filename = str(uuid.uuid4())

    abc = pyabc.ABCSMC(models=model, 
                       parameter_priors=pyabc.Distribution(priors), 
                       distance_function=error_metric, 
                       transitions=transitions, 
                       population_size=population_size)
    
    db_path = os.path.join(tempfile.gettempdir(), "test.db")
    abc.new("sqlite:///" + db_path, data)
    #if db == None:
    #    db_path = os.path.join(f'./calibration_runs/{basin_name}/dbs/', f"{filename}.db")
    #    abc.new("sqlite:///" + db_path, {"data": observations})

    #else:
    #    abc.load(db, run_id)
        
    history = abc.run(minimum_epsilon=minimum_epsilon, 
                      max_nr_populations=max_nr_populations,
                      max_walltime=max_walltime)
    
    # format output
    results = CalibrationResults()
    results.set_calibration_strategy("abc_smc")
    results.set_posterior_distribution(history.get_distribution()[0])
    results.set_selected_trajectories(np.array([d["data"] for d in history.get_weighted_sum_stats()[1]]))
    results.set_data(data)
    results.set_priors(priors)
    results.set_calibration_params({"population_size": population_size,
                                    "minimum_epsilon": minimum_epsilon,
                                    "error_metric": error_metric, 
                                    "max_nr_populations": max_nr_populations,
                                    "max_walltime": max_walltime, 
                                    "history": history})
    results.set_selected_quantiles(selected_simulations_quantiles)
    results.set_error_distribution(history.get_weighted_distances()["distance"].values)
    
    if include_quantiles: 
        # compute quantiles 
        selected_simulations_quantiles = compute_quantiles({"data": [d["data"] for d in history.get_weighted_sum_stats()[1]]}, simulation_dates=parameters[dates_column])
        results.set_selected_quantiles(selected_simulations_quantiles)
    
    return results


def run_projections(simulation_function, 
                    calibration_results, 
                    parameters, 
                    iterations=100, 
                    include_quantiles=True, 
                    dates_column="simulation_dates"): 

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