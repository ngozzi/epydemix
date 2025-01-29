from .transition import Transition
from ..utils.utils import format_simulation_output, create_definitions, apply_overrides, generate_unique_string, evaluate, compute_simulation_dates, apply_initial_conditions
from .simulation_output import Trajectory
from .simulation_results import SimulationResults
import numpy as np 
import pandas as pd
from numpy.random import multinomial
from ..population.population import Population, load_epydemix_population
from typing import List, Dict, Optional, Union, Any
import copy

class EpiModel:
    """
    EpiModel: A compartmental epidemic model simulator
    
    Example:
        >>> model = EpiModel()
        >>> model.add_compartment("S")\\
        ...      .add_compartment("I")\\
        ...      .add_compartment("R")
        >>> model.add_transition(source="S", target="I", rate="beta", agent="I")
        >>> model.add_transition(source="I", target="R", rate="gamma")
        >>> results = model.run_simulations(
        ...     start_date="2020-01-01",
        ...     end_date="2020-12-31",
        ...     Nsim=100
        ... )
    """

    def __init__(self, 
                 name : str = "EpiModel",
                 compartments: Optional[List] = None, 
                 parameters: Optional[Dict] = None,
                 population_name: str = "epydemix_population", 
                 population_data_path: Optional[str] = None, 
                 contact_layers: Optional[List] = None, 
                 contacts_source: Optional[str] = None, 
                 age_group_mapping: Optional[Dict] = None, 
                 supported_contacts_sources: Optional[List] = None, 
                 use_default_population: bool = True,
                 predefined_model: Optional[str] = None,  
                 transmission_rate: float = 0.3,  
                 recovery_rate: float = 0.1,
                 incubation_rate: float = 0.2) -> None:
            """
            Initializes the EpiModel instance.

            Args:
                name (str, optional): The name of the epidemic model. Defaults to "EpiModel".
                compartments (list, optional): A list of compartments for the model. Defaults to an empty list if None.
                population_name (str, optional): The name of the population to load or create. Defaults to 'epydemix_population'.
                population_data_path (str or None, optional): The path to the population data.
                contact_layers (list or None, optional): A list of contact layers to load for the population. Defaults to 
                    ["school", "work", "home", "community"] if None.
                contacts_source (str or None, optional): The source of contact data. If None, the function will automatically detect the source.
                age_group_mapping (dict or None, optional): A mapping of age groups for the population.
                supported_contacts_sources (list or None, optional): A list of supported contact data sources. Defaults to 
                    ["prem_2017", "prem_2021", "mistry_2021"] if None.
                use_default_population (bool, optional): If True, creates a default population; if False, tries to load the population from the provided path and population name. Defaults to True.
                predefined_model (str, optional): Load a predefined model like "SIR", "SEIR", etc. Defaults to None.
                transmission_rate (float, optional): Transmission rate for predefined models. Defaults to 0.3.
                recovery_rate (float, optional): Recovery rate for predefined models. Defaults to 0.1.

            Returns:
                None
            """
            # Initialize core attributes
            self.name = name
            self.transitions = {}
            self.transitions_list = []
            self.interventions = []
            self.compartments = []
            self.compartments_idx = {}
            self.transitions_idx = {}
            self.parameters = {}
            self.definitions = {}
            self.overrides = {}
            self.Cs = {}

            # Handle default empty lists for compartments and contact layers
            if compartments is None:
                compartments = []
            if contact_layers is None:
                contact_layers = ["school", "work", "home", "community"]

            # Add parameters if provided
            if parameters:
                self.add_parameter(parameters_dict=parameters)

            # Handle default contact sources if not provided
            if supported_contacts_sources is None:
                supported_contacts_sources = ["prem_2017", "prem_2021", "mistry_2021"]

            # Load or create population based on use_default_population flag
            self.population = self._load_or_create_population(
                population_name,
                population_data_path,
                contact_layers,
                contacts_source,
                age_group_mapping,
                supported_contacts_sources,
                use_default_population
            )

            # Load predefined model if specified
            if predefined_model is not None:
                self.load_predefined_model(predefined_model, transmission_rate, recovery_rate, incubation_rate)
            else:
                self.add_compartments(compartments)  # Add custom compartments if no predefined model


    def __repr__(self) -> str:
        """
        Returns a string representation of the EpiModel object,
        summarizing its name, compartments, transitions, parameters, and population.

        Returns:
            str: String representation of the EpiModel object.
        """
        repr_str = f"EpiModel(name='{self.name}')\n"

        # Compartments summary
        repr_str += f"Compartments: {len(self.compartments)}\n"
        if len(self.compartments) > 0:
            repr_str += "  " + ", ".join(self.compartments) + "\n"
        else:
            repr_str += "  No compartments defined\n"

        # Transitions summary
        repr_str += f"Transitions: {len(self.transitions_list)}\n"
        if len(self.transitions_list) > 0:
            repr_str += "  Transitions between compartments:\n"
            for tr in self.transitions_list:
                repr_str += f"    {tr.source} -> {tr.target}, rate: {tr.rate}\n"
        else:
            repr_str += "  No transitions defined\n"

        # Parameters summary
        repr_str += f"Parameters: {len(self.parameters)}\n"
        if len(self.parameters) > 0:
            repr_str += "  Model parameters:\n"
            for param_name, param_value in self.parameters.items():
                repr_str += f"    {param_name}: {param_value}\n"
        else:
            repr_str += "  No parameters defined\n"

        # Population summary
        repr_str += f"Population: {self.population.name}\n"
        repr_str += f"  Population size: {sum(self.population.Nk)} individuals\n"
        repr_str += f"  Demographic groups: {len(self.population.Nk_names)}\n"
        if len(self.population.Nk_names) > 0:
            repr_str += "    " + ", ".join([str(name) for name in self.population.Nk_names]) + "\n"

        return repr_str


    def load_predefined_model(self, model_name: str, transmission_rate: float = 0.3, recovery_rate: float = 0.1, incubation_rate : float = 0.2) -> None:
        """
        Loads a predefined epidemic model (e.g., SIR, SEIR, SIS) with predefined compartments and transitions.

        Args:
            model_name (str): The name of the predefined model to load (e.g., "SIR").
            transmission_rate (float, optional): The transmission rate for the SIR model. Default is 0.3.
            recovery_rate (float, optional): The recovery rate for the SIR model. Default is 0.1.
            incubation_rate (float, optional): The incubation rate for the SEIR model. Default is 0.2.
        
        Raises:
            ValueError: If the model_name is not recognized.
        """
        if model_name.upper() == "SIR":
            # SIR model setup
            self.clear_compartments()
            self.clear_transitions()

            # Add SIR compartments
            self.add_compartments(["Susceptible", "Infected", "Recovered"])

            # Add parameters for SIR model
            self.add_parameter(name="transmission_rate", value=transmission_rate)
            self.add_parameter(name="recovery_rate", value=recovery_rate)

            # Add transitions for SIR model
            self.add_transition(source="Susceptible", target="Infected", rate="transmission_rate", agent="Infected")
            self.add_transition(source="Infected", target="Recovered", rate="recovery_rate")

        elif model_name.upper() == "SEIR":
            # SEIR model setup (Susceptible, Exposed, Infected, Recovered)
            self.clear_compartments()
            self.clear_transitions()

            # Add SEIR compartments
            self.add_compartments(["Susceptible", "Exposed", "Infected", "Recovered"])

            # Add parameters for SEIR model
            self.add_parameter(name="transmission_rate", value=transmission_rate)
            self.add_parameter(name="incubation_rate", value=incubation_rate)
            self.add_parameter(name="recovery_rate", value=recovery_rate)

            # Add transitions for SEIR model
            self.add_transition(source="Susceptible", target="Exposed", rate="transmission_rate", agent="Infected")
            self.add_transition(source="Exposed", target="Infected", rate="incubation_rate") 
            self.add_transition(source="Infected", target="Recovered", rate="recovery_rate")

        elif model_name.upper() == "SIS":
            # SIS model setup (Susceptible, Infected)
            self.clear_compartments()
            self.clear_transitions()

            # Add SIS compartments
            self.add_compartments(["Susceptible", "Infected"])

            # Add parameters for SIS model
            self.add_parameter(name="transmission_rate", value=transmission_rate)
            self.add_parameter(name="recovery_rate", value=recovery_rate)

            # Add transitions for SIS model
            self.add_transition(source="Susceptible", target="Infected", rate="transmission_rate", agent="Infected")
            self.add_transition(source="Infected", target="Susceptible", rate="recovery_rate")

        else:
            raise ValueError(f"Unknown predefined model: {model_name}")


    def _load_or_create_population(self, 
                                population_name: str, 
                                population_data_path: Optional[str], 
                                contact_layers: List, 
                                contacts_source: Optional[str], 
                                age_group_mapping: Optional[Dict], 
                                supported_contacts_sources: List, 
                                use_default_population: bool) -> Population:
        """
        Loads a population from a file or creates a default population if specified.

        Args:
            population_name (str): The name of the population.
            population_data_path (str or None): The path to the population data file. If None, data may be fetched online.
            contact_layers (list): A list of contact layers to load for the population (e.g., "home", "work", "school").
            contacts_source (str or None): The source of contact data. If None, load_population will auto-determine the source.
            age_group_mapping (dict or None): A mapping for age groups. If None, a default mapping is used.
            supported_contacts_sources (list): A list of supported contact data sources (e.g., "prem_2017", "prem_2021").
            use_default_population (bool): If True, creates a default population with a single contact matrix and population size of 100000. 
                If False, attempts to load population data from a local or online source.

        Returns:
            Population: A population object, either created manually or loaded from data.
        """
        if use_default_population:
            # Create a default population manually
            population = Population(name=population_name)
            population.add_contact_matrix(np.array([[1.]]))  # Default contact matrix
            population.add_population(np.array([100000]))    # Default population size
            return population
        else:
            # Load population using load_population, which handles both online and local sources
            return load_epydemix_population(
                population_name, 
                contacts_source=contacts_source, 
                path_to_data=population_data_path, 
                layers=contact_layers, 
                age_group_mapping=age_group_mapping, 
                supported_contacts_sources=supported_contacts_sources
            )


    def import_epydemix_population(self, 
                            population_name: str, 
                            population_data_path: Optional[str] = None, 
                            contact_layers: Optional[List] = None, 
                            contacts_source: Optional[str] = None, 
                            age_group_mapping: Optional[Dict] = None, 
                            supported_contacts_sources: Optional[List] = None) -> None:
        """
        Sets or updates the population for the model.

        Args:
            population_name (str): The name of the population to set.
            population_data_path (str or None, optional): The path to the population data file. If None, data may be fetched online.
            contact_layers (list or None, optional): A list of contact layers to load for the population. Defaults to 
                ["school", "work", "home", "community"] if None.
            contacts_source (str or None, optional): The source of contact data. If None, the function will attempt to determine the source automatically.
            age_group_mapping (dict or None, optional): A mapping for age groups. If None, a default mapping is used.
            supported_contacts_sources (list or None, optional): A list of supported contact data sources (e.g., "prem_2017", "prem_2021"). 
                Defaults to ["prem_2017", "prem_2021", "mistry_2021"] if None.

        Returns:
            None
        """
        # Handle default contact layers if not provided
        if contact_layers is None:
            contact_layers = ["school", "work", "home", "community"]

        # Handle default contact sources if not provided
        if supported_contacts_sources is None:
            supported_contacts_sources = ["prem_2017", "prem_2021", "mistry_2021"]

        # Load a new population using the same logic as in the constructor
        self.population = load_epydemix_population(
            population_name, 
            contacts_source=contacts_source, 
            path_to_data=population_data_path, 
            layers=contact_layers, 
            age_group_mapping=age_group_mapping, 
            supported_contacts_sources=supported_contacts_sources
        )


    def add_compartments(self, compartments: Union[List, object]) -> None:
        """
        Adds compartments to the epidemic model.

        Args:
            compartments (list or object): A list of compartments or a single compartment to be added.

        Returns:
            None
        """
        # Ensure we are always working with a list, even if a single compartment is passed
        if not isinstance(compartments, (list, tuple)):
            compartments = [compartments]

        # Add compartments to the model
        self.compartments.extend(compartments)

        # Determine the current maximum index in compartments_idx or set to -1 if empty
        max_idx = max(self.compartments_idx.values(), default=-1)

        # Add each compartment with corresponding transitions and index
        for i, compartment in enumerate(compartments, start=1):
            self.transitions[compartment] = []  # Initialize empty transitions for each compartment
            self.compartments_idx[compartment] = max_idx + i  # Assign a unique index to each compartment

    @property
    def n_compartments(self) -> int:
        """Get total number of compartments."""
        return len(self.compartments)

    
    def clear_compartments(self) -> None: 
        """
        This method resets the `compartments` list and clears the `compartments_idx` dictionary.

        Returns:
            None
        """
        self.compartments = []
        self.compartments_idx = {}


    def add_parameter(self, 
                    name: Optional[str] = None, 
                    value: Any = None, 
                    parameters_dict: Optional[Dict] = None) -> None:
        """ 
        Adds new parameters to the model.

        Args:
            name (str, optional): The name of the parameter to add. Ignored if `parameters_dict` is provided.
            value (any, optional): The value of the parameter to add. Ignored if `parameters_dict` is provided.
            parameters_dict (dict, optional): A dictionary of parameter names and values. If provided, 
                `name` and `value` are ignored and the parameters are updated using this dictionary.

        Raises:
            ValueError: If neither `name`/`value` nor `parameters_dict` is provided.
        
        Returns:
            None
        """
        if parameters_dict:
            self.parameters.update(parameters_dict)
        elif name is not None and value is not None:
            self.parameters.update({name: value})
        else:
            raise ValueError("Either name and value or parameters_dict must be provided.")


    def get_parameter(self, name: str) -> Any:
        """
        Retrieves the value of a specified parameter.

        Args:
            name (str): The name of the parameter to retrieve.

        Returns:
            any: The value of the specified parameter.

        Raises:
            KeyError: If the parameter with the given name is not found.
        """
        return self.parameters[name]


    def delete_parameter(self, name: str) -> Any:
        """
        Deletes a parameter from the dictionary of the model's parameters.

        Args:
            name (str): The name of the parameter to delete.

        Returns:
            any: The value of the deleted parameter.

        Raises:
            KeyError: If the parameter with the given name is not found.
        """
        return self.parameters.pop(name)


    def clear_parameters(self) -> None:
        """
        Clears all parameters from the model's parameters dictionary.

        Returns:
            None
        """
        self.parameters = {}


    def override_parameter(self, start_date: str, end_date: str, name: str, value: Any) -> None:
        """
        Adds an override for a parameter with the specified name.

        Args:
            start_date (str): The start date for the override period.
            end_date (str): The end date for the override period.
            name (str): The name of the parameter to override.
            value (any): The value to override the parameter with.

        Returns:
            None
        """
        override_dict = {"start_date": start_date, "end_date": end_date, "value": value}

        if name in self.overrides:
            self.overrides[name].append(override_dict)
        else:
            self.overrides[name] = [override_dict]


    def delete_override(self, name: str) -> None:
        """
        Deletes overrides for a specific parameter.

        Args:
            name (str): The name of the parameter whose overrides are to be deleted.

        Returns:
            None
        """
        self.overrides.pop(name, None)


    def clear_overrides(self) -> None:
        """
        Clears all parameter overrides from the model.

        Returns:
            None
        """
        self.overrides = {}


    def add_transition(self, source: str, target: str, rate: Union[str, float], agent: Optional[str] = None) -> None:
        """
        Adds a transition to the epidemic model.

        Args:
            source (str): The source compartment of the transition.
            target (str): The target compartment of the transition.
            rate (str | float): The expression or the value of the rate for the transition.
            agent (str, optional): The interacting agent for the transition (default is None).

        Raises:
            ValueError: If the source, target, or agent (if provided) is not in the compartments list.

        Returns:
            None
        """
        # Check that the source, target, and agent (if provided) are in the compartments list
        missing_compartments = [comp for comp in [source, target, agent] if comp and comp not in self.compartments]
        if missing_compartments:
            raise ValueError(f"These compartments are not in the compartments list: {', '.join(missing_compartments)}")

        # Handle rate based on type (string or float)
        if isinstance(rate, str):
            transition = Transition(source=source, target=target, rate=rate, agent=agent)
        else:
            # Generate a unique name for the rate if it's a float
            rate_name = generate_unique_string()
            transition = Transition(source=source, target=target, rate=rate_name, agent=agent)
            # Add rate to parameters
            self.add_parameter(name=rate_name, value=rate)

        # Append the transition to the transitions list and dictionary
        self.transitions_list.append(transition)
        self.transitions[source].append(transition)

        # Update transitions index
        transition_name = f"{source}_to_{target}"
        if transition_name not in self.transitions_idx:
            self.transitions_idx[transition_name] = len(self.transitions_idx)

    @property
    def n_transitions(self) -> int:
        """Get total number of transitions."""
        return len(self.transitions_idx)
        
    
    def clear_transitions(self) -> None:
        """
        Clears all transitions from the model.

        This method resets the `transitions_list` and reinitializes the `transitions` dictionary, 
        clearing all existing transitions while keeping the structure intact for each compartment.

        Returns:
            None
        """
        self.transitions_list = []
        self.transitions = {comp: [] for comp in self.compartments}


    def add_intervention(self, 
                        layer_name: str, 
                        start_date: Union[str, pd.Timestamp], 
                        end_date: Union[str, pd.Timestamp], 
                        reduction_factor: Optional[float] = None, 
                        new_matrix: Optional[np.ndarray] = None, 
                        name: str = "") -> None:
        """
        Adds an intervention to the epidemic model.

        Args:
            layer_name (str): The name of the layer to which the intervention applies.
            start_date (str or datetime): The start date of the intervention.
            end_date (str or datetime): The end date of the intervention.
            reduction_factor (float, optional): The factor by which to reduce the contact matrix. Default is None.
            new_matrix (np.ndarray, optional): A new contact matrix to use during the intervention. Default is None.
            name (str, optional): The name of the intervention. Default is an empty string.

        Raises:
            ValueError: If neither reduction_factor nor new_matrix is provided.

        Returns:
            None
        """
        # Convert start_date and end_date to pandas datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Check if both reduction_factor and new_matrix are None
        if reduction_factor is None and new_matrix is None:
            raise ValueError("Either reduction_factor or new_matrix must be provided")

        # Append the intervention to the interventions list
        self.interventions.append({
            "layer": layer_name, 
            "start_date": start_date, 
            "end_date": end_date, 
            "reduction_factor": reduction_factor, 
            "new_matrix": new_matrix,
            "name": name
        })

 
    def clear_interventions(self) -> None:
        """
        Clears all interventions from the model.

        This method resets the `interventions` list, removing all existing interventions.

        Returns:
            None
        """
        self.interventions = []


    def apply_intervention(self, intervention: Dict, simulation_dates: List[pd.Timestamp]) -> None:
        """
        Applies an intervention to the contact matrices for specified simulation dates.

        Args:
            intervention (dict): A dictionary containing intervention details with the following keys:
                - "layer" (str): The name of the layer to which the intervention applies.
                - "start_date" (pd.Timestamp): The start date of the intervention.
                - "end_date" (pd.Timestamp): The end date of the intervention.
                - "reduction_factor" (float, optional): The factor by which to reduce the contact matrix.
                - "new_matrix" (np.ndarray, optional): A new contact matrix to use during the intervention.
            simulation_dates (list of pd.Timestamp): A list of dates for which the simulation is run.

        Raises:
            ValueError: If neither reduction_factor nor new_matrix is provided in the intervention.

        Returns:
            None
        """
        # Early validation of the intervention inputs
        reduction_factor = intervention.get("reduction_factor")
        new_matrix = intervention.get("new_matrix")
        
        if reduction_factor is None and new_matrix is None:
            raise ValueError("Intervention must have either a reduction_factor or a new_matrix")

        # Extract commonly used values
        layer = intervention["layer"]
        start_date = intervention["start_date"]
        end_date = intervention["end_date"]

        # Apply the intervention to the relevant dates
        for date in filter(lambda d: start_date <= d <= end_date, simulation_dates):
            if reduction_factor is not None:
                self.Cs[date][layer] *= reduction_factor
            else:  # If reduction_factor is None, we assume new_matrix is provided
                self.Cs[date][layer] = new_matrix


    def compute_contact_reductions(self, simulation_dates: List[pd.Timestamp]) -> None:
        """
        Computes the contact reductions for a population over the given simulation dates.

        This function applies interventions to the contact matrices and computes the overall contact matrix 
        for each date in the simulation period.

        Args:
            simulation_dates (list of pd.Timestamp): A list of dates over which the simulation is run.

        Returns:
            None: The function updates the instance variable `self.Cs` with the contact matrices for each date, 
                including the overall contact matrix after applying interventions.
        """
        # Initialize contact matrices for each simulation date, copying the population's initial contact matrices
        self.Cs = {date: {layer: np.copy(matrix) for layer, matrix in self.population.contact_matrices.items()} 
                for date in simulation_dates}

        # Apply interventions to the contact matrices
        for intervention in self.interventions:
            self.apply_intervention(intervention, simulation_dates)

        # Compute the overall contact matrix for each date by summing up the contact matrices across layers
        for date in self.Cs:
            self.Cs[date]["overall"] = np.sum(np.array(list(self.Cs[date].values())), axis=0)


    def create_default_initial_conditions(self, percentage_in_agents: float = 0.0005) -> Dict[str, np.ndarray]:
        """
        Creates default initial conditions for the epidemic model. If initial conditions are not provided, 
        assigns a percentage of the population to agent compartments and the rest to source compartments, 
        considering different age groups.

        Args:
            percentage_in_agents (float): Percentage of population to place in agent compartments. Defaults to 0.05%.
            initial_conditions (dict, optional): A dictionary specifying initial conditions. If None, this function creates default conditions.

        Returns:
            dict: A dictionary with initial conditions for each compartment, with values as arrays representing different age groups.
        """

        population = self.population.Nk

        # Initialize initial conditions dictionary
        initial_conditions_dict = {}

        # Total population for each age group
        total_population_per_age_group = np.array(population)

        # Get compartments that are agents in transitions
        agent_compartments = {tr.agent for tr in self.transitions_list if tr.agent}
        
        # Get compartments that are sources in transitions with agents
        source_compartments = {tr.source for tr in self.transitions_list if tr.agent}

        # Total number of agent compartments
        num_agent_compartments = len(agent_compartments)

        # Distribute population into agent compartments
        if num_agent_compartments > 0:
            population_in_agents_per_age_group = (total_population_per_age_group * percentage_in_agents).astype(int)
            population_per_agent_per_age_group = population_in_agents_per_age_group // num_agent_compartments

            for comp in agent_compartments:
                initial_conditions_dict[comp] = population_per_agent_per_age_group.copy()

        # Assign the rest of the population to source compartments
        remaining_population_per_age_group = total_population_per_age_group - np.sum(list(initial_conditions_dict.values()), axis=0)
        num_source_compartments = len(source_compartments)

        if num_source_compartments > 0:
            population_per_source_per_age_group = remaining_population_per_age_group // num_source_compartments

            for comp in source_compartments:
                initial_conditions_dict[comp] = population_per_source_per_age_group.copy()

        # If some compartments are missing, ensure they get zero population
        for comp in self.compartments:
            if comp not in initial_conditions_dict:
                initial_conditions_dict[comp] = np.zeros_like(total_population_per_age_group)

        return initial_conditions_dict


    def set_population(self, population: Population) -> None:
        """
        Sets the population for the epidemic model.

        Args:
            population (Population): The population object to set.

        Returns:
            None
        """
        self.population = population
    

    def run_simulations(self, 
                       start_date: Union[str, pd.Timestamp] = "2020-01-01", 
                       end_date: Union[str, pd.Timestamp] = "2020-12-31", 
                       initial_conditions_dict: Optional[Dict[str, np.ndarray]] = None, 
                       Nsim: int = 100, 
                       percentage_in_agents: float = 0.0005,
                       dt: Optional[float] = 1.,
                       resample_frequency: Optional[str] = "D",
                       resample_aggregation: Optional[Union[str, dict]] = "last",
                       fill_method: Optional[str] = "ffill") -> SimulationResults:
        """
        Simulates the epidemic model multiple times over the given time period.

        Args:
            start_date (str or pd.Timestamp): The start date of the simulation. Default is "2020-01-01".
            end_date (str or pd.Timestamp): The end date of the simulation. Default is "2020-12-31".
            initial_conditions_dict (dict, optional): A dictionary of initial conditions for the simulation.
            Nsim (int, optional): The number of simulation runs to perform (default is 100).
            percentage_in_agents (float, optional): The percentage of the population to initialize in the agents compartment.
            dt (float, optional): The time step for the simulation, expressed in days. Default is 1 (day).
            resample_frequency (str, optional): The frequency at which to resample the simulation results. Default is "D" (daily).
            resample_aggregation (str, optional): The aggregation method to use when resampling the simulation results. Default is "sum".
            fill_method (str, optional): Method to fill NaN values after resampling. Default is "ffill".

        Returns:
            SimulationResults: An object containing all simulation trajectories.

        Raises:
            RuntimeError: If the simulation fails.
        """
        
        # Run multiple simulations and collect trajectories
        try:
            trajectories = []
            for _ in range(Nsim):
                trajectory = simulate(
                    self, 
                    start_date=start_date,
                    end_date=end_date,
                    dt=dt,
                    initial_conditions_dict=initial_conditions_dict,
                    percentage_in_agents=percentage_in_agents,
                    resample_frequency=resample_frequency,
                    resample_aggregation=resample_aggregation,
                    fill_method=fill_method
                )
                trajectories.append(trajectory)
        except Exception as e:
            raise RuntimeError(f"Simulation failed: {str(e)}") from e

        # Return SimulationResults with all trajectories
        return SimulationResults(
            trajectories=trajectories,
            parameters=self.parameters
        )


def simulate(epimodel, 
             start_date: Union[str, pd.Timestamp] = "2020-01-01", 
             end_date: Union[str, pd.Timestamp] = "2020-12-31", 
             initial_conditions_dict: Optional[Dict[str, np.ndarray]] = None, 
             percentage_in_agents: float = 0.0005,
             dt: Optional[float] = 1.,
             resample_frequency: Optional[str] = "D",
             resample_aggregation_compartments: Optional[Union[str, dict]] = "last",
             resample_aggregation_transitions: Optional[Union[str, dict]] = "sum",
             fill_method: Optional[str] = "ffill",
             **kwargs) -> Trajectory:
    """
    Runs a simulation of the epidemic model over the specified simulation dates.

    Args:
        epimodel (EpiModel): The epidemic model instance to simulate.
        start_date (str or pd.Timestamp): The start date of the simulation. Default is "2020-01-01".
        end_date (str or pd.Timestamp): The end date of the simulation. Default is "2020-12-31".
        initial_conditions_dict (dict, optional): A dictionary of initial conditions for the simulation.
        percentage_in_agents (float, optional): The percentage of the population to initialize in the agents compartment.
        dt (float, optional): The time step for the simulation, expressed in days. Default is 1 (day).
        resample_frequency (str, optional): The frequency at which to resample the simulation results. Default is "D" (daily).
        resample_aggregation_compartments (str, optional): The aggregation method to use when resampling the compartments. Default is "last".
        resample_aggregation_transitions (str, optional): The aggregation method to use when resampling the transitions. Default is "sum".
        fill_method (str, optional): The method to use when filling NaN values after resampling. Default is "ffill".
        **kwargs: Additional parameters to overwrite model parameters during the simulation.

    Returns:
        Trajectory: The trajectory of the simulation

    Raises:
        ValueError: If the model has no transitions defined.
    """

    # check that the model has transitions
    if len(epimodel.transitions_list) == 0:
        raise ValueError("The model has no transitions defined. Please add transitions before running simulations.")
    
    # Compute the simulation dates
    simulation_dates = compute_simulation_dates(start_date, end_date, dt=dt)

    # Compute initial conditions if needed
    if initial_conditions_dict is None:
        initial_conditions_dict = epimodel.create_default_initial_conditions(percentage_in_agents=percentage_in_agents)
        
    # Compute the contact reductions based on the interventions
    epimodel.compute_contact_reductions(simulation_dates)

    # Update parameters if any are provided via kwargs (needed for calibration purposes)
    parameters = epimodel.parameters.copy()
    parameters.update(kwargs)

    # Compute the definitions and apply overrides
    epimodel.definitions = create_definitions(parameters, len(simulation_dates), epimodel.population.Nk.shape[0])
    epimodel.definitions = apply_overrides(epimodel.definitions, epimodel.overrides, simulation_dates)

    # Initialize population in different compartments and demographic groups
    initial_conditions = apply_initial_conditions(epimodel, initial_conditions_dict)

    # Pre-compute contact matrices list
    contact_matrices = [epimodel.Cs[date]["overall"] for date in simulation_dates]
    
    # Run simulation with pre-computed contacts
    compartments_evolution, transitions_evolution = stochastic_simulation(
        T=len(simulation_dates),
        contact_matrices=contact_matrices,  
        epimodel=epimodel,
        parameters=epimodel.definitions,
        initial_conditions=initial_conditions,
        dt=dt
    )

    # Format the simulation output
    results = format_simulation_output(compartments_evolution, transitions_evolution, 
                                       epimodel.compartments_idx, epimodel.transitions_idx, 
                                       epimodel.population.Nk_names)
    trajectory = Trajectory(compartments=results["compartments"], transitions=results["transitions"], 
                            dates=simulation_dates, compartment_idx=epimodel.compartments_idx, 
                            transitions_idx=epimodel.transitions_idx, parameters=epimodel.definitions)

    # Only resample if necessary
    if resample_frequency is not None:
        # Check if resampling is needed (simulation dates frequency != requested frequency)
        sim_freq = pd.infer_freq(simulation_dates)
        if sim_freq != resample_frequency:
            trajectory = trajectory.resample(resample_frequency, 
                                             resample_aggregation_compartments, 
                                             resample_aggregation_transitions, 
                                             fill_method)
    return trajectory


def stochastic_simulation(T: int,
                         contact_matrices: List[np.ndarray],
                         epimodel,
                         parameters: Dict,
                         initial_conditions: np.ndarray,
                         dt: float) -> np.ndarray:
    """
    Run a stochastic simulation of the epidemic model.
    
    Args:
        T: Number of time steps
        contact_matrices: Pre-computed list of contact matrices
        epimodel: The epidemic model
        parameters: Model parameters
        initial_conditions: Initial population distribution
        dt: Time step size
    """
    # Pre-allocate arrays
    N = len(epimodel.population.Nk)
    C = len(epimodel.compartments)
    
    compartments_evolution = np.zeros((T + 1, C, N), dtype=np.float64)
    transitions_evolution = np.zeros((T, epimodel.n_transitions, N), dtype=np.float64)
    compartments_evolution[0] = initial_conditions
    
    # Pre-compute population sizes and create views for better performance
    pop_sizes = epimodel.population.Nk
    
    # Pre-allocate arrays for probabilities and transitions
    prob = np.zeros((C, N), dtype=np.float64)
    new_pop = np.zeros((C, N), dtype=np.float64)
    
    # Pre-compute indices for better performance
    comp_indices = epimodel.compartments_idx
    
    # Simulate each time step
    for t in range(T):
        contact_matrix = contact_matrices[t]
        pop = compartments_evolution[t]
        new_pop[:] = pop  
        
        for comp in epimodel.compartments:
            transitions = epimodel.transitions[comp]
            if not transitions: 
                continue
                
            prob.fill(0)
            source_idx = comp_indices[comp]

            for tr in transitions:
                target_idx = comp_indices[tr.target]
                env_copy = copy.deepcopy(parameters)
                rate = evaluate(expr=tr.rate, env=env_copy)[t]
                
                if tr.agent is not None:
                    agent_idx = comp_indices[tr.agent]
                    interaction = np.sum(
                        contact_matrix * pop[agent_idx] / pop_sizes, 
                        axis=1
                    )
                    rate *= interaction
                
                prob[target_idx] += rate * dt
            
            # Compute transition probabilities 
            np.negative(prob, out=prob)
            np.exp(prob, out=prob)
            np.subtract(1, prob, out=prob)
            prob[source_idx] = 1 - np.sum(prob, axis=0)
            
            current_pop = pop[source_idx]
            if not np.any(current_pop):
                continue
            
            # Compute transitions using multinomial (vectorized)
            delta = np.array([
                multinomial(n, p) if n > 0 else np.zeros(C)
                for n, p in zip(current_pop, prob.T)
            ])

            # Store transition counts
            for tr in transitions:
                tr_name = f"{tr.source}_to_{tr.target}"
                tr_idx = epimodel.transitions_idx[tr_name]
                transitions_evolution[t, tr_idx] = delta[:, epimodel.compartments_idx[tr.target]]
            
            # Update populations
            np.subtract(new_pop[source_idx], np.sum(delta, axis=1), 
                       out=new_pop[source_idx])
            new_pop += delta.T
        
        compartments_evolution[t + 1] = new_pop
    
    return compartments_evolution[1:], transitions_evolution