# libraries
from .transition import Transition
from .utils import compute_quantiles, format_simulation_output, combine_simulation_outputs, create_definitions, apply_overrides, generate_unique_string, evaluate, load_population
from .simulation_results import SimulationResults
import numpy as np 
import pandas as pd
from numpy.random import multinomial
from datetime import timedelta
from epydemix.population import Population
import uuid


class EpiModel:
    """
    EpiModel Class
    """

    def __init__(self, compartments=[], population_name="epydemix_population", population_data_path=None):
        """
        EpiModel constructor
        """
        self.transitions = {}
        self.transitions_list = []
        self.interventions = []
        self.compartments = []
        self.compartments_idx = {}
        self.parameters = {}
        self.definitions = {}
        self.overrides = {}
        self.add_compartments(compartments)
        self.Cs = {}

        if population_data_path is not None: 
            self.population = load_population(population_name, path_to_data=population_data_path)
        else: 
            self.population = Population(name=population_name)
            self.population.add_contact_matrix(np.array([[0.5, 0.5], [0.5, 0.5]]))
            self.population.add_population(np.array([50000, 50000]))


    def add_compartments(self, compartments): 
        """
        Adds compartments to the epidemic model.

        Parameters:
        -----------
            - compartments (list or object): A list of compartments or a single compartment to be added.
        """
        if isinstance(compartments, list):
            self.compartments.extend(compartments)
            if len(self.compartments_idx.values()) == 0: 
                max_idx = -1
            else:
                max_idx = max(self.compartments_idx.values())
            for i, compartment in enumerate(compartments):
                self.transitions[compartment] = []
                self.compartments_idx[compartment] = max_idx + (i + 1)

        else:
            self.compartments.append(compartments)
            self.transitions[compartments] = []
            if len(self.compartments_idx.values()) == 0: 
                max_idx = -1
            else:
                max_idx = max(self.compartments_idx.values())
            self.compartments_idx[compartments] = max_idx + 1


    def clear_compartments(self): 
        """
        Clears the compartments list.
        """
        self.compartments = []
        self.compartments_idx = {}


    def add_parameter(self, name=None, value=None, parameters_dict=None):
        """
        Add new parameters to the model.

        Parameters:
        -----------
        name : str, optional
            The name of the parameter.
        value : any, optional
            The value of the parameter.
        parameters_dict : dict, optional
            A dictionary of parameter names and values.

        If `parameters_dict` is provided, `name` and `value` are ignored and
        the parameters are updated using the dictionary.
        """
        if parameters_dict:
            self.parameters.update(parameters_dict)
        elif name is not None and value is not None:
            self.parameters.update({name: value})
        else:
            raise ValueError("Either name and value or parameters_dict must be provided.")


    def get_parameter(self, name): 
        """
        Return the value of a parameter

        Parameters: 
            - name (str): The nema of the parameter
        """
        return self.parameters[name]


    def delete_parameter(self, name): 
        """
        Delete a parameter from the dictionary of model's parameters

        Parameters: 
            - name (str): The name of the parameter
        """
        return self.parameters.pop(name)
    

    def clear_parameters(self):
        """
        Clears the parameters dictionary.
        """
        self.parameters = {}


    def override_parameter(self, start_date, end_date, name, value):
        """
        Adds an override for a parameter with the specified name.

        Parameters:
        -----------
            - start_date (str): The start date for the override period.
            - end_date (str): The end date for the override period
            - name (str): The name of the parameter to override.
            - value (any): The value to override the parameter with.
        """
        override_dict = {"start_date": start_date, "end_date": end_date, "value": value}
            
        if name in self.overrides:
            self.overrides[name].append(override_dict)
        else:
            self.overrides[name] = [override_dict]


    def delete_override(self, name): 
        """
        Delete overrides for a parameter 

        Parameters: 
            - name (str): The name of the parameter
        """
        self.overrides.pop(name)


    def clear_overrides(self): 
        """
        Clears the overrides dictionary.
        """
        self.overrides = {}


    def add_transition(self, source, target, rate, agent=None):
        """
        Adds a transition to the epidemic model.

        Parameters:
        -----------
            - source (str): The source compartment of the transition.
            - target (str): The target compartment of the transition.
            - rate (str, float): The expression or the value of the rate for the transition.
            - agent (str, optional): The interacting agent for the transition (default is None).

        Raises:
        -------
            - ValueError: If the source, target, or agent (if provided) is not in the compartments list.
        """

        # Check that the source and target are in the compartment list
        missing_compartments = [comp for comp in [source, target, agent] if comp and comp not in self.compartments]
        if missing_compartments:
            raise ValueError(f"These compartments are not in the compartments list: {', '.join(missing_compartments)}")

        if isinstance(rate, str):
            transition = Transition(source=source, target=target, rate=rate, agent=agent)
        else: 
            # create unique name for rate 
            rate_name = generate_unique_string()
            transition = Transition(source=source, target=target, rate=rate_name, agent=agent)
            # add rate to parameters
            self.add_parameter(name=rate_name, value=rate)

        # Append to transitions list
        self.transitions_list.append(transition)

        # Add transition to the dictionary
        self.transitions[source].append(transition)

    
    def clear_transitions(self): 
        """
        Clears the transitions list.
        """
        self.transitions_list = []
        self.transitions = {comp: [] for comp in self.compartments}


    def add_intervention(self, layer_name, start_date, end_date, reduction_factor=None, new_matrix=None, name=""): 
        """
        Adds an intervention to the epidemic model.

        Parameters:
        -----------
            - layer_name (str): The name of the layer to which the intervention applies.
            - start_date (str or datetime): The start date of the intervention.
            - end_date (str or datetime): The end date of the intervention.
            - reduction_factor (float, optional): The factor by which to reduce the contact matrix. Default is None.
            - new_matrix (np.ndarray, optional): A new contact matrix to use during the intervention. Default is None.
            - name (str, optional): The name of the intervention. Default is an empty string.

        Raises:
        -------
            - ValueError: If neither reduction_factor nor new_matrix is provided.
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

            
    def clear_interventions(self):
        """
        Clears the interventions list.
        """
        self.interventions = [] 


    def apply_intervention(self, intervention, simulation_dates):
        """
        Applies an intervention to the contact matrices for specified simulation dates.

        Parameters:
        -----------
            - intervention (dict): A dictionary containing intervention details with keys:
                - "layer" (str): The name of the layer to which the intervention applies.
                - "start_date" (pd.Timestamp): The start date of the intervention.
                - "end_date" (pd.Timestamp): The end date of the intervention.
                - "reduction_factor" (float, optional): The factor by which to reduce the contact matrix.
                - "new_matrix" (np.ndarray, optional): A new contact matrix to use during the intervention.
            - simulation_dates (list of pd.Timestamp): A list of dates for which the simulation is run.

        Raises:
        -------
            - ValueError: If neither reduction_factor nor new_matrix is provided in the intervention.
        """
        # Check if intervention has either reduction_factor or new_matrix
        if intervention["reduction_factor"] is None and intervention["new_matrix"] is None:
            raise ValueError("Intervention must have either a reduction_factor or a new_matrix")

        # Apply the intervention to the relevant dates
        for date in simulation_dates:
            if intervention["start_date"] <= date <= intervention["end_date"]:
                if intervention["reduction_factor"] is not None:
                    self.Cs[date][intervention["layer"]] = intervention["reduction_factor"] * self.Cs[date][intervention["layer"]]

                elif intervention["new_matrix"] is not None:
                    self.Cs[date][intervention["layer"]] = intervention["new_matrix"]


    def compute_contact_reductions(self, simulation_dates): 
        """
        Computes the contact reductions for a population over the given simulation dates.

        This function applies interventions to the contact matrices and computes the overall contact matrix for each date
        in the simulation period.

        Parameters:
        -----------
            - simulation_dates (list of pd.Timestamp): A list of dates over which the simulation is run.

        Returns:
        --------
            - None: The function updates the instance variable `self.Cs` with the contact matrices for each date, 
                    including the overall contact matrix after applying interventions.
        """
        # apply interventions
        self.Cs = {date: {l: c for l, c in self.population.contact_matrices.items()} for date in simulation_dates}
        for intervention in self.interventions:
            self.apply_intervention(intervention, simulation_dates) 
        
        # compute overall contacts
        for date in self.Cs.keys(): 
            self.Cs[date]["overall"] = np.sum(np.array(list(self.Cs[date].values())), axis=0)
 

    def run_simulations(self, start_date, end_date, steps="daily", dt=None, Nsim=100, quantiles=[0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975], post_processing_function=lambda x, **kwargs: x, **kwargs): 
        """
        Simulates the epidemic model over the given time period.

        Parameters:
        -----------
            - start_date (str or pd.Timestamp): The start date of the simulation.
            - end_date (str or pd.Timestamp): The end date of the simulation.
            - steps (int or str): The number of time steps in the simulation (default is daily, implying that the simulation step will be automatically 1 day)
            - dt (float): The length of the simulation step referred to 1 day (default is None).
            - Nsim (int, optional): The number of simulation runs to perform (default is 100).
            - quantiles (list of float, optional): A list of quantiles to compute for the simulation results (default is [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]).
            - **kwargs: Additional arguments to pass to the stochastic simulation function.

        Returns:
        --------
            - simulated_compartments (list of np.ndarray): The results of the simulated compartments for each run.
            - df_quantiles (pd.DataFrame): A DataFrame containing the quantile values for each compartment, demographic group, and date.
        """
        start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)

        if steps == "daily":
            simulation_dates = pd.date_range(start=start_date, end=end_date, freq="d").tolist()
        else: 
            simulation_dates = pd.date_range(start=start_date, end=end_date, periods=steps).tolist()

        # compute contact reductions
        self.compute_contact_reductions(simulation_dates)

        # simulation parameters 
        parameters = {"simulation_dates": simulation_dates}

        # add initial conditions to parameters
        for comp in kwargs: 
            parameters[comp] = kwargs[comp]
        
        parameters["epimodel"] = self

        # simulation dt 
        if dt is None:
            parameters["dt"] = np.diff(simulation_dates)[0] / timedelta(days=1)
        else: 
            parameters["dt"] = dt

        # simulate
        simulated_compartments = {}
        for i in range(Nsim): 
            results = stochastic_simulation(parameters=parameters, post_processing_function=post_processing_function)
            simulated_compartments = combine_simulation_outputs(simulated_compartments, results)

        simulated_compartments = {k: np.array(v) for k, v in simulated_compartments.items()}
        df_quantiles = compute_quantiles(data=simulated_compartments, simulation_dates=simulation_dates, quantiles=quantiles)
        
        # format simulation results 
        simulation_results = SimulationResults() 
        simulation_results.set_Nsim(Nsim)
        simulation_results.set_df_quantiles(df_quantiles)
        simulation_results.set_full_trajectories(simulated_compartments)
        simulation_results.set_compartment_idx(self.compartments_idx)
        simulation_results.set_parameters(parameters)

        return simulation_results
    
def simulate(): 
    return 0


def stochastic_simulation(parameters, post_processing_function=lambda x, **kwargs: x): 
    """
    Run a stochastic simulation of the epidemic model over the specified time period.

    Parameters:
    -----------
        - Nk (np.ndarray): An array representing the population in different demographic groups.
        - Cs (dict): A dictionary of contact matrices, where keys are dates and values are dictionaries 
                    containing layer matrices, including an "overall" matrix.
        - post_processing_function=lambda x: x (function): A function that manipulates the output of the simulation (default is an identity function).
        - **kwargs: Additional named arguments representing initial populations in different compartments.
                    Each key should be the name of a compartment and the value should be the initial population array
                    for that compartment.

    Returns:
    --------
        - np.ndarray: A 3D array representing the evolution of compartment populations over time. The shape of the 
                    array is (time_steps, num_compartments, num_demographic_groups).
    """

    epimodel = parameters["epimodel"]

    # check if some parameter need to be overwritten (prior)
    for k in parameters.keys(): 
        if k in epimodel.parameters: 
            epimodel.parameters[k] = parameters[k]
 
    # compute parameter definitions and apply overrides
    epimodel.definitions = create_definitions(epimodel.parameters, len(parameters["simulation_dates"]))
    epimodel.definitions = apply_overrides(epimodel.definitions, epimodel.overrides, parameters["simulation_dates"])
    parameters.update(epimodel.definitions)

    # initialize population in different compartments and demographic groups
    compartments_population = np.zeros((len(epimodel.compartments), len(epimodel.population.Nk)), dtype='int')
    for comp in epimodel.compartments:
        if comp in parameters.keys(): 
            compartments_population[epimodel.compartments_idx[comp]] = parameters[comp]
    compartments_evolution = [compartments_population]     

    # simulate
    for i, date in enumerate(epimodel.Cs.keys()):

        # get today contacts matrix and seasonality adjustment
        C = epimodel.Cs[date]["overall"]

        # last time step population in different compartments (n_comp, n_age)
        pop = compartments_evolution[-1]
        # next step solution
        new_pop = pop.copy()
        for comp in epimodel.compartments:
            # transition (edge) attribute dict in 3-tuple (u, v, ddict) and initialize probabilities
            trans = epimodel.transitions[comp]
            prob = np.zeros((len(epimodel.compartments), len(epimodel.population.Nk)), dtype='float')

            for tr in trans:
                # get source, target, and rate for this transition
                source = epimodel.compartments_idx[tr.source]
                target = epimodel.compartments_idx[tr.target]
                rate = evaluate(expr=tr.rate, env=parameters)[i]

                # check if this transition has an interaction
                if tr.agent is not None:
                    agent = epimodel.compartments_idx[tr.agent]  # get agent position
                    interaction = np.array([np.sum(C[age, :] * pop[agent, :] / epimodel.population.Nk) for age in range(len(epimodel.population.Nk))])
                    rate *= interaction        # interaction term
                prob[target, :] += rate * parameters["dt"]

            # exponential of 
            prob = 1 - np.exp(-prob)

            # prob of not transitioning
            prob[source, :] = 1 - np.sum(prob, axis=0)

            # compute transitions
            delta = np.array([multinomial(pop[source][i], prob[:, i]) for i in range(len(epimodel.population.Nk))])
            delta[:, source] = 0
            changes = np.sum(delta)

            # if no changes continue to next iteration
            if changes == 0:
                continue

            # update population of source of transition
            new_pop[source, :] -= np.sum(delta, axis=1)

            # update target compartments population
            new_pop += delta.T

        compartments_evolution.append(new_pop)

    # format simulation output
    results = format_simulation_output(np.array(compartments_evolution)[1:], parameters)

    # apply post_processing 
    results = post_processing_function(results, **parameters)
    return results
