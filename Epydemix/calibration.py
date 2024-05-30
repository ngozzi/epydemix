import numpy as np 

def rmse(data, simulation): 
    return np.sqrt(np.mean((data["data"] - simulation["data"])**2))

def calibration_top_perc(simulation_function, 
                priors, 
                parameters, 
                data=None,
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

    # select parameters
    selected_params = {p: np.array(arr)[idxs] for p, arr in sampled_params.items()}

    return selected_simulations, selected_params