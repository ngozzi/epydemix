import numpy as np 
import pandas as pd 
from .calibration_results import CalibrationResults
from .utils import compute_quantiles, combine_simulation_outputs

def rmse(data, simulation): 
    return np.sqrt(np.mean((data["data"] - simulation["data"])**2))

def calibration_top_perc(simulation_function, 
                priors, 
                parameters, 
                data,
                top_perc=0.05,
                error_metric=rmse,
                Nsim=100, 
                post_processing_function=None): 
    
    """
    Parameters
        - priors (dict): dictionary of prior distributions
    """
    
    simulations, errors = [], []
    sampled_params = {p: [] for p in priors.keys()}
    for n in range(Nsim):

        # sample
        for param, distr in priors.items():
            rv = distr.rvs()
            sampled_params[param].append(rv)
            parameters[param] = rv

        if post_processing_function is None:
            results = simulation_function(parameters)
        else: 
            results = simulation_function(parameters, post_processing_function=post_processing_function)
            
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

    # compute quantiles 
    selected_simulations_quantiles = compute_quantiles(selected_simulations_formatted, simulation_dates=parameters["simulation_dates"])

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
    results.set_selected_quantiles(selected_simulations_quantiles)

    return results