import numpy as np

def stochastic_sir(S_t0, I_t0, R_t0, beta, mu, N, timesteps):
    """
    Simulates a stochastic SIR model using the chain binomial process.

    Parameters:
    S_t0, I_t0, R_t0 : int
        Initial number of susceptible, infectious, and recovered individuals.
    beta : float
        Transmission rate per contact.
    mu : float
        Recovery rate (1/infectious period).
    N : int
        Total population.
    timesteps : int
        Number of time steps to simulate.

    Returns:
    dict
        Time series of SIR compartments and incidence.
    """
    
    S, I, R = S_t0, I_t0, R_t0
    S_arr, I_arr, R_arr = [S], [I], [R]
    incidence = [0]

    for _ in range(timesteps-1):
        # Infection process (Binomial draw of new infections)
        new_infected = np.random.binomial(S, 1 - np.exp(-beta * I / N))
        
        # Recovery process
        new_recovered = np.random.binomial(I, 1 - np.exp(-mu))

        # Update compartments
        S -= new_infected
        I += new_infected - new_recovered
        R += new_recovered

        # Store results
        S_arr.append(S)
        I_arr.append(I)
        R_arr.append(R)

        incidence.append(new_infected)

    return {"S": S_arr, "I": I_arr, "R": R_arr, "incidence": incidence}
