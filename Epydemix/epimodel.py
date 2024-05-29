# libraries
from .transition import Transition
from .utils import compute_quantiles
import numpy as np 
import pandas as pd
from numpy.random import multinomial
from datetime import datetime

class EpiModel:
    """
    EpiModel Class
    """

    def __init__(self, compartments=[]):
        """
        EpiModel constructor
        """
        self.transitions = {}
        self.transitions_list = []
        self.interventions = []
        self.compartments = []
        self.compartments_idx = {}
        self.parameters = {}
        self.add_compartments(compartments)


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


    def add_parameters(self, parameters):
        """
        Add new parameters to the epidemic model.

        Parameters: 
            - parameters (dict): A dictionary of parameters
        """
        self.parameters.update(parameters)


    def add_transition(self, source, target, rate_name, agent=None):
        """
        Adds a transition to the epidemic model.

        Parameters:
        -----------
            - source (str): The source compartment of the transition.
            - target (str): The target compartment of the transition.
            - rate_name (str): The name of the rate for the transition.
            - rate (float, optional): The rate at which the transition occurs (default is None).
            - agent (str, optional): The interacting agent for the transition (default is None).

        Raises:
        -------
            - ValueError: If the source, target, or agent (if provided) is not in the compartments list.
        """

        # Check that the source and target are in the compartment list
        missing_compartments = [comp for comp in [source, target, agent] if comp and comp not in self.compartments]
        if missing_compartments:
            raise ValueError(f"These compartments are not in the compartments list: {', '.join(missing_compartments)}")
        
        # Check that the rate of the transition has been added to the parameters
        if rate_name not in self.parameters.keys(): 
            raise ValueError(f"{rate_name} of the transition has been added to the parameters")

        transition = Transition(source=source, target=target, rate_name=rate_name, agent=agent)

        # Append to transitions list
        self.transitions_list.append(transition)

        # Add transition to the dictionary
        self.transitions[source].append(transition)


    def add_intervention(self, layer_name, start_date, end_date, reduction_factor=None, new_matrix=None): 
        """
        Adds an intervention to the epidemic model.

        Parameters:
        -----------
            - layer_name (str): The name of the layer to which the intervention applies.
            - start_date (str or datetime): The start date of the intervention.
            - end_date (str or datetime): The end date of the intervention.
            - reduction_factor (float, optional): The factor by which to reduce the contact matrix. Default is None.
            - new_matrix (np.ndarray, optional): A new contact matrix to use during the intervention. Default is None.

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
            "new_matrix": new_matrix
        })


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


    def simulate(self, population, start_date, end_date, steps, Nsim=100, quantiles=[0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975], **kwargs): 
        """
        Simulates the epidemic model over the given time period.

        Parameters:
        -----------
            - population (object): An object containing population data, particularly the contact matrices and demographic groups.
            - start_date (str or pd.Timestamp): The start date of the simulation.
            - end_date (str or pd.Timestamp): The end date of the simulation.
            - steps (int): The number of time steps in the simulation.
            - Nsim (int, optional): The number of simulation runs to perform (default is 100).
            - quantiles (list of float, optional): A list of quantiles to compute for the simulation results (default is [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]).
            - **kwargs: Additional arguments to pass to the stochastic simulation function.

        Returns:
        --------
            - simulated_compartments (list of np.ndarray): The results of the simulated compartments for each run.
            - df_quantiles (pd.DataFrame): A DataFrame containing the quantile values for each compartment, demographic group, and date.
        """
        start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
        simulation_dates = pd.date_range(start=start_date, end=end_date, periods=steps).tolist()

        # apply interventions
        self.Cs = {date: {l: c for l, c in population.contact_matrices.items()} for date in simulation_dates}
        for intervention in self.interventions:
            self.apply_intervention(intervention, simulation_dates) 
        
        # compute overall contacts
        for date in self.Cs.keys(): 
            self.Cs[date]["overall"] = np.sum(np.array(list(self.Cs[date].values())), axis=0)

        # simulate
        simulated_compartments = []
        for i in range(Nsim): 
            simulated_compartments.append(self.stochastic_simulation(population.Nk, self.Cs, **kwargs))
        simulated_compartments = np.array(simulated_compartments)

        df_quantiles = compute_quantiles(population.Nk_names, self.compartments_idx, simulated_compartments, simulation_dates, quantiles=quantiles)
        return simulated_compartments, df_quantiles


    def stochastic_simulation(self, Nk, Cs, **kwargs): 
        """
        Run a stochastic simulation of the epidemic model over the specified time period.

        Parameters:
        -----------
            - Nk (np.ndarray): An array representing the population in different demographic groups.
            - Cs (dict): A dictionary of contact matrices, where keys are dates and values are dictionaries 
                        containing layer matrices, including an "overall" matrix.
            - **kwargs: Additional named arguments representing initial populations in different compartments.
                        Each key should be the name of a compartment and the value should be the initial population array
                        for that compartment.

        Returns:
        --------
            - np.ndarray: A 3D array representing the evolution of compartment populations over time. The shape of the 
                        array is (time_steps, num_compartments, num_demographic_groups).
        """
        
        # population in different compartments and demographic groups
        compartments_population = np.zeros((len(self.compartments), len(Nk)), dtype='int')
        for comp in kwargs:
            compartments_population[self.compartments_idx[comp]] = kwargs[comp]
        compartments_evolution = [compartments_population]     

        # simulate
        for date in Cs.keys():

            # get today contacts matrix and seasonality adjustment
            C = Cs[date]["overall"]

            # last time step population in different compartments (n_comp, n_age)
            pop = compartments_evolution[-1]
            # next step solution
            new_pop = pop.copy()
            for comp in self.compartments:
                # transition (edge) attribute dict in 3-tuple (u, v, ddict) and initialize probabilities
                trans = self.transitions[comp]
                prob = np.zeros((len(self.compartments), len(Nk)), dtype='float')

                for tr in trans:
                    # get source, target, and rate for this transition
                    source = self.compartments_idx[tr.source]
                    target = self.compartments_idx[tr.target]
                    rate = self.parameters[tr.rate_name]

                    # check if this transition has an interaction
                    if tr.agent is not None:
                        agent = self.compartments_idx[tr.agent]  # get agent position
                        interaction = np.array([np.sum(C[age, :] * pop[agent, :] / Nk) for age in range(len(Nk))])
                        rate *= interaction        # interaction term
                    prob[target, :] += rate

                # prob of not transitioning
                prob[source, :] = 1 - np.sum(prob, axis=0)

                # compute transitions
                delta = np.array([multinomial(pop[source][i], prob[:, i]) for i in range(len(Nk))])
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

        return np.array(compartments_evolution)[1:]