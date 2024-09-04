import numpy as np 
import pandas as pd
from collections.abc import Iterable
import datetime
import random
import string
from evalidate import Expr, base_eval_model


def validate_parameter_shape(key, value, T, n_age):
    """
    Validates the shape of the input value based on its type and expected dimensions.

    Parameters:
    -----------
        - key (str): The key associated with the value in the parameters dictionary.
        - value (np.ndarray): The value to be validated.
        - T (int): The expected length of the first dimension.
        - n_age (int): The expected length of the second dimension.

    Raises:
    -------
    ValueError
        If the value does not meet the required shape criteria.
    """
    if isinstance(value, (int, float)):
        return  # Scalars don't need validation beyond their type

    elif isinstance(value, Iterable):
        value = np.array(value)

        if value.ndim == 1:  # 1D array
            if len(value) < T:
                raise ValueError(f"The length of the 1D iterable for parameter '{key}' is smaller than simulation length ({T}).")
        
        elif value.ndim == 2:  # 2D array
            if value.shape[0] != T and value.shape[0] != 1:
                raise ValueError(f"The first dimension of the 2D iterable for parameter '{key}' must be 1 or match simulation length ({T}).")
            if value.shape[1] != n_age:
                raise ValueError(f"The second dimension of the 2D iterable for parameter '{key}' must match number of age groups ({n_age}).")
        
        else:
            raise ValueError(f"Unsupported number of dimensions for parameter '{key}': {value.ndim}")
        
    else:
            raise ValueError(f"Unsupported type for parameter '{key}': {type(value)}")


def resize_parameter(value, T, n_age):
    """
    Resizes the input value to have the shape (T, n_age).

    Parameters:
    -----------
        - value (np.ndarray or scalar): The value to be resized.
        - T (int): The length of the first dimension.
        - n_age (int): The length of the second dimension.

    Returns:
    --------
        np.ndarray: A 2D array with shape (T, n_age).
    """
    if isinstance(value, (int, float)):  # Scalar value
        return np.full((T, n_age), value)

    value = np.array(value)

    if value.ndim == 1:  # 1D array
        return np.tile(value, (n_age, 1)).T

    elif value.ndim == 2:  # 2D array
        if value.shape[0] == 1:  # If the first dimension is 1, repeat it to match T
            return np.tile(value, (T, 1))
        return value
    

def create_definitions(parameters, T, n_age):
    """
    Generates a dictionary where each value is a 2D array based on the input dictionary and the parameters provided.

    Parameters:
    -----------
        - parameters (dict): A dictionary where values can be either scalars or iterables.
        - T (int): The length of the first dimension of the arrays to be created.
        - n_age (int): The length of the second dimension of the arrays to be created.

    Returns:
    --------
    A dictionary where keys are the same as in `parameters` and values are 2D arrays of shape `(T, n_age)`.
    """
    definitions = {}
    for key, value in parameters.items():
        validate_parameter_shape(key, value, T, n_age)
        definitions[key] = resize_parameter(value, T, n_age)
    
    return definitions


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


def format_simulation_output(simulation_output, compartments_idx, Nk_names): 
    """
    Formats the simulation output into a dictionary with compartment and demographic information.

    Parameters:
    -----------
        - simulation_output (np.ndarray): A 3D array containing the simulation results.
                                          The dimensions are expected to be (time_steps, compartments, demographics).
        - compartments_idx (dict): dictionary mapping compartment names to their indices.
        - Nk_names (list): a list of demographic group names.

    Returns:
    --------
        - dict: A dictionary where keys are in the format "compartment_demographic" and values are 2D arrays (time_steps, values).
                An additional key "compartment_total" is included for each compartment, representing the sum across all demographics.
    """
    formatted_output = {}
    for comp, pos in compartments_idx.items(): 
        for i, dem in enumerate(Nk_names): 
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


def str_to_date(date_str):
    return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()


def apply_overrides(definitions, overrides, dates):
    """
    Applies overrides to the definitions based on the specified dates.

    Parameters:
    -----------
        - definitions (dict): A dictionary where keys are parameter names and values are arrays 
                              representing the values of the parameters over time.
        - overrides (dict): A dictionary of parameter overrides
        - dates (list): A list of dates corresponding to the values in the definitions arrays.

    Returns:
    --------
        - A dictionary with the same keys as definitions, but with values updated according to the overrides.
    """
    for name, overrides in overrides.items():
        if name in definitions:
            values = definitions[name]
            for override in overrides:
                start_date = str_to_date(override["start_date"])
                end_date = str_to_date(override["end_date"])
                override_value = override["value"]

                # validate override value
                T, n_age = sum(start_date <= d.date() <= end_date for d in dates), definitions[name].shape[1]
                validate_parameter_shape(name, override_value, T=T, n_age=n_age)

                # resize override value
                override_value = resize_parameter(override_value, T=T, n_age=n_age)

                # override
                override_idxs = [i for i, date in enumerate(dates) if start_date <= date.date() <= end_date]
                values[override_idxs] = override_value

    return definitions


def generate_unique_string(length=12):
    """
    Generates a random unique string containing only letters.

    Parameters:
    -----------
    length : int, optional
        The length of the generated string (default is 12).

    Returns:
    --------
    str
        A random unique string containing only letters.
    """
    letters = string.ascii_letters  # Contains both lowercase and uppercase letters
    return ''.join(random.choice(letters) for _ in range(length))


def compute_days(start_date, end_date):
    return pd.date_range(start_date, end_date).shape[0]


def evaluate(expr, env):
    """
    Evaluates the expression with the given environment, allowing only whitelisted operations.

    This function extends the base evaluation model to whitelist the 'Mult' and 'Pow' operations,
    ensuring only these operations are permitted during the evaluation.

    Parameters:
    -----------
    expr : str
        The expression to evaluate.
    env : dict
        The environment containing variable values.

    Returns:
    --------
    any
        The result of evaluating the expression.

    Raises:
    -------
    EvalException
        If there is an error in evaluating the expression.
    """
    eval_model = base_eval_model
    eval_model.nodes.extend(['Mult', 'Pow'])
    return Expr(expr, model=eval_model).eval(env)


def compute_simulation_dates(start_date, end_date, steps="daily"): 
    if steps == "daily":
        simulation_dates = pd.date_range(start=start_date, end=end_date, freq="d").tolist()
    else: 
        simulation_dates = pd.date_range(start=start_date, end=end_date, periods=steps).tolist()

    return simulation_dates


def apply_initial_conditions(epimodel, **kwargs): 
    # initialize population in different compartments and demographic groups
    initial_conditions = np.zeros((len(epimodel.compartments), len(epimodel.population.Nk)), dtype='int')
    for comp in epimodel.compartments:
        if comp in kwargs: 
            if comp in epimodel.compartments:
                initial_conditions[epimodel.compartments_idx[comp]] = kwargs[comp]

    return initial_conditions


def convert_to_2Darray(lst):
    arr = np.array(lst)
    arr = arr.reshape(1, len(lst))
    return arr