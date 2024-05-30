import numpy as np 
import pandas as pd
import os


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
        self.Nk = []                # Population data
        self.Nk_names = []          # list of demographic group names


    def add_contact_matrix(self, contact_matrix, layer_name="all"):
        """
        Adds a contact matrix for a specified layer.
        
        Parameters:
        -----------
            - contact_matrix (np.ndarray): The contact matrix to be added.
            - layer_name (str): The name of the contact layer (default is "all").
        """
        self.contact_matrices[layer_name] = contact_matrix


    def add_population(self, Nk, Nk_names=None): 
        """
        Adds population data.
        -----------
        Parameters:
            - Nk (list): Population data array.
        """
        self.Nk = Nk
        if Nk_names is None:
            self.Nk_names = list(range(len(Nk)))
        else: 
            self.Nk_names = Nk_names


    def load_population(self, population_name, path_to_data): 
        self.name = population_name
        for layer_name in ["school", "work", "home", "community"]:
            self.add_contact_matrix(np.load(os.path.join(path_to_data, population_name, f"contacts-matrix/contacts_matrix_{layer_name}.npz"))["arr_0"], layer_name=layer_name)

        Nk = pd.read_csv(os.path.join(path_to_data, population_name, "demographic/Nk.csv"))
        self.add_population(Nk=Nk["value"].values, Nk_names=Nk["group"].values)