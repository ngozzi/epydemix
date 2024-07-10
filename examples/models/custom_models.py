import numpy as np

def my_stochastic_SIR(parameters):
    """
    Simulate a stochastic SIR model using chain binomial processes.

    Parameters:
    parameters (dict): Dictionary containing the following keys:
        'beta' (float): Transmission rate.
        'gamma' (float): Recovery rate.
        'S' (int): Initial number of susceptible individuals.
        'I' (int): Initial number of infected individuals.
        'R' (int): Initial number of recovered individuals.
        'N' (int): Total number of individuals.
        'simulation_dates' (list): list of simulation dates.
        'dt' (float): Time step for the simulation.

    Returns:
    dict: A dictionary with key "data" containing the simulated incidence.
    """
    
    beta = parameters['beta']
    mu = parameters['mu']
    S = parameters['S']
    I = parameters['I']
    R = parameters['R']
    dates = parameters['simulation_dates']
    dt = parameters['dt']
    N = parameters['N']

    S_arr, I_arr, R_arr = [], [], []
    incidence = []

    for date in dates:
        # Probability of infection and recovery in one time step
        p_infection = 1 - np.exp(-beta * I / N * dt)
        p_recovery = 1 - np.exp(-mu * dt)

        # Number of new infections and recoveries
        new_infections = np.random.binomial(S, p_infection)
        new_recoveries = np.random.binomial(I, p_recovery)

        # Update counts
        S -= new_infections
        I += new_infections - new_recoveries
        R += new_recoveries

        incidence.append(new_infections)
        S_arr.append(S)
        I_arr.append(I) 
        R_arr.append(R)

    return {'data': incidence,
            "S": S_arr, "I": I_arr, "R": R_arr}

