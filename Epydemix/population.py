import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from collections import OrderedDict


class Population:
    """
    Population object class that handles contact matrices and population data.
    """

    def __init__(self, name="population"):
        """
        Initializes the Population object.
        
        Parameters:
        -----------
            - name (str): Name of the population object (default is "population").
        """
        self.name = name
        self.contact_matrices = {}  # Dictionary to hold contact matrices for different layers
        self.Nk = []  # Population data


    def add_contact_matrix(self, contact_matrix, layer_name="all"):
        """
        Adds a contact matrix for a specified layer.
        
        Parameters:
        -----------
            - contact_matrix (np.ndarray): The contact matrix to be added.
            - layer_name (str): The name of the contact layer (default is "all").
        """
        self.contact_matrices[layer_name] = contact_matrix


    def add_population(self, Nk): 
        """
        Adds population data.
        -----------
        Parameters:
            - Nk (list): Population data array.
        """
        self.Nk = Nk


    def update_contacts(self, date):
        """
        Rescales the contacts matrix for a specific date according to the reduction factor.
        
        Parameters:
        -----------
            - date (datetime): The date for which to rescale the contacts matrix.
        
        Returns:
        --------
            - np.ndarray: Rescaled contacts matrix for the given date.
        
        Raises:
        -------
            - ValueError: If the reduction factor for the date is not found.
        """
        try:
            reduction_factor = self.reductions.loc[self.reductions.year_week == date.strftime('%G-%V')]["red"].values[0]
        except IndexError:
            raise ValueError(f"Reduction factor for date {date} not found.")
        return reduction_factor * self.contact_matrices.get("all", np.zeros_like(self.Nk))


    def compute_contacts(self, start_date, end_date):
        """
        Computes contact matrices over a given time window.
        
        Parameters:
        -----------
            - start_date (str): The initial date in 'YYYY-MM-DD' format.
            - end_date (str): The final date in 'YYYY-MM-DD' format.
        """
        dates = pd.date_range(start=start_date, end=end_date)
        self.Cs = OrderedDict({date: self.update_contacts(date) for date in dates})


