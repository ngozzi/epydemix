import numpy as np 
import pandas as pd
import os 
from collections import OrderedDict

demographic_grouping_prem = OrderedDict({
    "0-4": np.arange(0, 5).astype(str), 
    "5-9": np.arange(5, 10).astype(str),
    "10-14": np.arange(10, 15).astype(str),
    "15-19": np.arange(15, 20).astype(str),
    "20-24": np.arange(20, 25).astype(str),
    "25-29": np.arange(25, 30).astype(str),
    "30-34": np.arange(30, 35).astype(str),
    "35-39": np.arange(35, 40).astype(str),
    "40-44": np.arange(40, 45).astype(str),
    "45-49": np.arange(45, 50).astype(str),
    "50-54": np.arange(50, 55).astype(str),
    "55-59": np.arange(55, 60).astype(str),
    "60-64": np.arange(60, 65).astype(str),
    "65-69": np.arange(65, 70).astype(str),
    "70-74": np.arange(70, 75).astype(str),
    "75+": np.concatenate((np.arange(75, 84), ["84+"])).astype(str)})

contacts_age_group_mapping_prem = {
    "0-4": ["0-4"], 
    "5-19": ["5-9", "10-14", "15-19"], 
    "20-49": ["20-24", "25-29", "30-34", "35-39", "40-44", "45-49"], 
    "50-64": ["50-54", "55-59", "60-64"],
    "65+": ["65-69", "70-74", "75+"]}

contacts_age_group_mapping_mistry = {
    "0-4": np.arange(0, 5).astype(str),
    "5-19": np.arange(5, 20).astype(str), 
    "20-49": np.arange(20, 50).astype(str), 
    "50-64": np.arange(50, 65).astype(str),
    "65+": np.concatenate((np.arange(65, 84).astype(str), ["84+"]))}


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

def map_age_groups_to_idx(age_group_mapping, old_age_groups_idx, new_age_group_idx):
    # Initialize the result dictionary
    age_group_mapping_idx = {}

    # Iterate through each key-value pair in the first dictionary
    for new_group, old_groups in age_group_mapping.items():

        # Get the corresponding integer for this new group
        related_int = new_age_group_idx[new_group]

        # Map each group in the list to the related index
        for grp in old_groups:
            # Get the integer for the list item
            item_int = old_age_groups_idx[grp]
            # Set the mapping in the result dictionary
            age_group_mapping_idx[item_int] = related_int

    return age_group_mapping_idx


def aggregate_matrix(initial_matrix, 
                     old_population, 
                     new_population, 
                     age_group_mapping, 
                     old_age_groups_idx, 
                     new_age_group_idx):

    # Turn matrix of rates into contacts
    real_contacts = initial_matrix.copy()
    for i in range(real_contacts.shape[0]): 
        real_contacts[i] = real_contacts[i] * old_population[i]

    # compute age group mapping
    age_group_mapping_idxs = map_age_groups_to_idx(age_group_mapping, old_age_groups_idx, new_age_group_idx)

    # Determine the number of aggregated groups
    num_aggregated_groups = max(age_group_mapping_idxs.values()) + 1

    # Initialize the aggregated matrix
    aggregated_matrix = np.zeros((num_aggregated_groups, num_aggregated_groups))

    # Fill the aggregated matrix
    for i in range(real_contacts.shape[0]):
        for j in range(real_contacts.shape[1]):
            aggregated_i = age_group_mapping_idxs[i]
            aggregated_j = age_group_mapping_idxs[j]
            aggregated_matrix[aggregated_i, aggregated_j] += real_contacts[i, j]

    # Turn into rates
    aggregated_matrix_rate = aggregated_matrix.copy() 
    for i in range(aggregated_matrix_rate.shape[0]):
        aggregated_matrix_rate[i] = aggregated_matrix_rate[i] / new_population[i]

    return aggregated_matrix_rate

  
def aggregate_demographic(data, grouping): 
    Nk_new, Nk_names_new = [], []

    for new_group in grouping.keys(): 
        Nk_names_new.append(new_group)
        Nk_new.append(data.loc[data.group_name.isin(grouping[new_group])]["value"].sum())

    df_Nk_new = pd.DataFrame({"group_name": Nk_names_new, "value": Nk_new})
    return df_Nk_new  
 

def validate_population_name(population_name, path_to_data):
    locations_list = pd.read_csv(os.path.join(path_to_data, "locations.csv"))["location"].values
    if population_name not in locations_list:
        raise ValueError(f"Location {population_name} not found in the list of supported locations. See {path_to_data}/locations.csv")
    

def get_primary_contacts_source(population_name, path_to_data):
    contact_matrices_sources = pd.read_csv(os.path.join(path_to_data, "locations.csv"))
    source_location = contact_matrices_sources.loc[contact_matrices_sources.location == population_name, "primary_contact_source"]
    contacts_source = source_location.iloc[0]
    return contacts_source


def validate_contacts_source(contacts_source, supported_contacts_sources):
    if contacts_source not in supported_contacts_sources:
        raise ValueError(f"Source {contacts_source} not found in the list of supported sources. Supported sources are {supported_contacts_sources}")


def validate_age_group_mapping(age_group_mapping, allowed_values):
    values = np.concatenate(list(age_group_mapping.values())) 
    if not np.all(np.isin(values, allowed_values)): 
        raise ValueError(f"Age group mapping values must be in {allowed_values}")


def load_population(population_name, 
                    contacts_source=None, 
                    path_to_data=None, 
                    layers=["school", "work", "home", "community"], 
                    age_group_mapping=None, 
                    supported_contacts_sources=["prem_2017", "prem_2021", "mistry_2021"], 
                    path_to_data_github="https://raw.githubusercontent.com/ngozzi/epydemix/main/epydemix_data/"): 
    
    population = Population(name=population_name)

    # if path is None tries online import
    if path_to_data is None:
        path_to_data = path_to_data_github
    
    # check location is supported
    validate_population_name(population_name, path_to_data)

    # check if contacts_source is supported
    if contacts_source is None: 
        contacts_source = get_primary_contacts_source(population_name, path_to_data)
    validate_contacts_source(contacts_source, supported_contacts_sources)

    # load the demographic data
    Nk = pd.read_csv(os.path.join(path_to_data, population_name, "demographic/age_distribution.csv"))
    
    # if contact matrices are from Prem et al 2017 or Prem et al 2021 we need to aggregate population data
    if contacts_source in ["prem_2017", "prem_2021"]: 
        Nk = aggregate_demographic(Nk, demographic_grouping_prem)

    # get age group mapping
    if age_group_mapping is None: 
        if contacts_source in ["prem_2017", "prem_2021"]:
            age_group_mapping = contacts_age_group_mapping_prem
        else: 
            age_group_mapping = contacts_age_group_mapping_mistry

    # validate age group mapping
    validate_age_group_mapping(age_group_mapping, Nk.group_name.values)
        
    # aggregate population to new age groups
    Nk_new = aggregate_demographic(Nk, age_group_mapping)
    population.add_population(Nk=Nk_new["value"].values, Nk_names=Nk_new["group_name"].values)

    # load the contact matrices
    for layer_name in layers:
        C = pd.read_csv(os.path.join(path_to_data, population_name, "contact_matrices", contacts_source, f"contacts_matrix_{layer_name}.csv"), header=None).values

        # aggregate contact matrices to new age groups
        C_aggr = aggregate_matrix(C, 
                                  old_population=Nk["value"].values, 
                                  new_population=Nk_new["value"].values, 
                                  age_group_mapping=age_group_mapping, 
                                  old_age_groups_idx={name: idx for idx, name in enumerate(Nk.group_name.values)}, 
                                  new_age_group_idx={name: idx for idx, name in enumerate(age_group_mapping.keys())})

        population.add_contact_matrix(C_aggr, layer_name=layer_name)

    return population
