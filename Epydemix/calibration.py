def calibration(simulation_function, 
                priors, 
                parameters, 
                data=None,
                Nsim=100): 
    
    """
    Parameters
        - priors (dict): dictionary of prior distributions
    """
    
    simulations = []
    sampled_params = {p: [] for p in priors.keys()}
    for n in range(Nsim):

        # sample
        for param, distr in priors.items():
            rv = distr.rvs()
            sampled_params[param].append(rv)
            parameters[param] = rv

        results = simulation_function(parameters)
        simulations.append(results)

    return simulations, sampled_params