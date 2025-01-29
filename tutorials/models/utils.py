import numpy as np 
from datetime import datetime
import pandas as pd
from typing import List

def compute_contact_reductions(mob_data : pd.DataFrame, 
                               columns : List[str]) -> pd.DataFrame:

    """
    This function computes the contact reductions factor from mobility data
    Parameters
    ----------
        @param mob_data (pd.DataFrame): mobility data dataframe
        @param columns (List[str]): list of columns to use to compute the contact reduction factor
    Return
    ------
        @return: returns pd.DataFrame of contact reduction factors for each date
    """

    contact_reductions = pd.DataFrame(data={'date': mob_data.date,
                                            'r': (1 + mob_data[columns].mean(axis=1) / 100)**2})
    return contact_reductions

def get_beta(R0, mu, Nk, C):
    """
    Compute the transmission rate beta for a SEIR model with age groups
    Parameters
    ----------
        @param R0: basic reproductive number
        @param mu: recovery rate
        @param Nk: number of individuals in different age groups
        @param C: contact matrix
    Return
    ------
        @return: the transmission rate beta
    """
    # get seasonality adjustment
    C_hat = np.zeros((C.shape[0], C.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C_hat[i, j] = (Nk[i] / Nk[j]) * C[i, j]
    return R0 * mu / (np.max([e.real for e in np.linalg.eig(C_hat)[0]]))


def apply_seasonality(day: datetime, 
                      seasonality_min: float, 
                      basin_hemispheres: int, 
                      seasonality_max: float = 1) -> float:
    """
    Applies seasonality adjustment to a given day based on the specified parameters.

    Parameters
    - day (datetime): The specific day for which seasonality adjustment is applied.
    - seasonality_min (float): The minimum value of the seasonality adjustment.
    - basin_hemispheres (int): The indicator of the basin hemisphere (0 for northern, 1 for tropical, 2 for southern).
    - seasonality_max (float, optional): The maximum value of the seasonality adjustment. Defaults to 1.

    Returns:
    - float: The seasonality adjustment value for the specified day and basin hemisphere.

    Raises:
    - None
    """

    s_r = seasonality_min / seasonality_max
    day_max_north = datetime(day.year, 1, 15)
    day_max_south = datetime(day.year, 7, 15)

    seasonal_adjustment = np.empty(shape=(3,), dtype=np.float64)

    # north hemisphere
    seasonal_adjustment[0] = 0.5 * ((1 - s_r) * np.sin(2 * np.pi / 365 * (day - day_max_north).days + 0.5 * np.pi) + 1 + s_r)

    # tropical hemisphere
    seasonal_adjustment[1] = 1.0

    # south hemisphere
    seasonal_adjustment[2] = 0.5 * ((1 - s_r) * np.sin(2 * np.pi / 365 * (day - day_max_south).days + 0.5 * np.pi) + 1 + s_r)

    return seasonal_adjustment[basin_hemispheres]


def get_gamma_params(mu, std):
    """
    Compute the parameters of a gamma distribution given its mean and standard deviation.
    """
    shape = (mu/std)**2
    scale = std**2 / mu
    return shape, scale

