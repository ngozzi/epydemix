import numpy as np 
import pandas as pd

def compute_quantiles(data, simulation_dates, axis=0, quantiles=[0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]):
    """
    Computes the specified quantiles for each key in the provided data over the given dates.

    Parameters:
    -----------
        - data (dict of np.ndarray): A dictionary where keys represent different data categories (e.g., compartments, demographic groups) 
                                   and values are arrays containing the simulation results.
        - simulation_dates (list of str or pd.Timestamp): The dates corresponding to the simulation time steps.
        - axis (int, optional): The axis along which to compute the quantiles (default is 0).
        - quantiles (list of float, optional): A list of quantiles to compute for the simulation results (default is [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]).

    Returns:
    --------
        - pd.DataFrame: A DataFrame containing the quantile values for each data category and date.
                        The DataFrame has columns for the data category, quantile, and date.
    """
    dict_quantiles = {k: [] for k in data.keys()}
    dict_quantiles["quantile"] = []
    dict_quantiles["date"] = []

    for q in quantiles:
        for k, v in data.items():
            arrq = np.quantile(v, axis=axis, q=q)
            dict_quantiles[k].extend(arrq)
        dict_quantiles["quantile"].extend([q] * len(arrq))
        dict_quantiles["date"].extend(simulation_dates)

    df_quantile = pd.DataFrame(data=dict_quantiles) 
    return df_quantile


def format_simulation_output(simulation_output, parameters): 
    """
    Formats the simulation output into a dictionary with compartment and demographic information.

    Parameters:
    -----------
        - simulation_output (np.ndarray): A 3D array containing the simulation results.
                                          The dimensions are expected to be (time_steps, compartments, demographics).
        - parameters (dict): A dictionary containing the simulation parameters.
                             It must include:
                             - "epimodel": An object with a `compartments_idx` attribute, which is a dictionary mapping compartment names to their indices.
                             - "population": An object with a `Nk_names` attribute, which is a list of demographic group names.

    Returns:
    --------
        - dict: A dictionary where keys are in the format "compartment_demographic" and values are 2D arrays (time_steps, values).
                An additional key "compartment_total" is included for each compartment, representing the sum across all demographics.
    """
    formatted_output = {}
    for comp, pos in parameters["epimodel"].compartments_idx.items(): 
        for i, dem in enumerate(parameters["population"].Nk_names): 
            formatted_output[f"{comp}_{dem}"] = simulation_output[:, pos, i]
        formatted_output[f"{comp}_total"] = np.sum(simulation_output[:, pos, :], axis=1)

    return formatted_output


def combine_simulation_outputs(combined_simulation_outputs, simulation_outputs):
    """
    Combines multiple simulation outputs into a single dictionary by appending new outputs to existing keys.

    Parameters:
    -----------
        - combined_simulation_outputs (dict): A dictionary to accumulate combined simulation outputs.
                                              Keys are compartment-demographic names and values are lists of arrays.
        - simulation_outputs (dict): A dictionary containing the latest simulation output to be combined.
                                     Keys are compartment-demographic names and values are arrays of simulation results.

    Returns:
    --------
        - dict: A dictionary where keys are compartment-demographic names and values are lists of arrays.
                Each list contains simulation results accumulated from multiple runs.
    """
    if not combined_simulation_outputs:
        # If combined_dict is empty, initialize it with the new dictionary
        for key in simulation_outputs:
            combined_simulation_outputs[key] = [simulation_outputs[key]]
    else:
        # If combined_dict already has keys, append the new dictionary's values
        for key in simulation_outputs:
            if key in combined_simulation_outputs:
                combined_simulation_outputs[key].append(simulation_outputs[key])
            else:
                combined_simulation_outputs[key] = [simulation_outputs[key]]
    return combined_simulation_outputs
