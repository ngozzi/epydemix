import numpy as np 
import pandas as pd


def compute_quantiles(dem_group_names, compartments_idx, simulated_compartments, simulation_dates, quantiles=[0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]):
    """
    Computes quantiles for the simulated compartments over the simulation dates.

    Parameters:
    -----------
        - dem_group_names (list): The names of the demographic groups.
        - compartments_idx (dict): The indices of the compartments.
        - simulated_compartments (list): A list of simulated compartment values.
        - simulation_dates (list of pd.Timestamp): The dates over which the simulation runs.
        - quantiles (list of float, optional): A list of quantiles to compute (default is [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]).

    Returns:
    --------
        - pd.DataFrame: A DataFrame containing the quantile values for each compartment, demographic group, and date.
    """
    # Convert simulated compartments to a NumPy array for easier manipulation
    simulated_compartments = np.array(simulated_compartments)

    # Initialize lists to store the results
    demographic_group, quantile_levels, dates, compartment_values, compartment_names = [], [], [], [], []

    # Iterate over each compartment
    for comp in compartments_idx.keys():
        comp_idx = compartments_idx[comp]
        
        # Compute quantiles for each specified quantile level
        for q in quantiles:
            # Compute quantiles for each demographic group
            for group in dem_group_names:
                arr = np.quantile(simulated_compartments[:, :, comp_idx, group], q=q, axis=0)
                
                # Extend results to lists
                compartment_values.extend(arr)
                compartment_names.extend([comp] * len(arr))
                quantile_levels.extend([q] * len(arr))
                demographic_group.extend([group] * len(arr))
                dates.extend(simulation_dates)

            # Compute quantiles for the total over all demographic groups
            total_arr = np.quantile(np.sum(simulated_compartments[:, :, comp_idx], axis=2), q=q, axis=0)
            
            # Extend results to lists
            compartment_values.extend(total_arr)
            compartment_names.extend([comp] * len(total_arr))
            quantile_levels.extend([q] * len(total_arr))
            demographic_group.extend(["total"] * len(total_arr))
            dates.extend(simulation_dates)

    # Create a DataFrame from the results
    df_quantiles = pd.DataFrame({
        "date": dates,
        "demographic_group": demographic_group,
        "quantile": quantile_levels,
        "compartment": compartment_names,
        "value": compartment_values
    })

    return df_quantiles