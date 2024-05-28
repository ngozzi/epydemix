# libraries
from .transition import Transition
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
            for compartment in compartments:
                self.transitions[compartment] = []
        else:
            self.compartments.append(compartments)
            self.transitions[compartments] = []


    def add_transition(self, source, target, rate, rate_name, agent=None):
        """
        Adds a transition to the epidemic model.

        Parameters:
        -----------
            - source (str): The source compartment of the transition.
            - target (str): The target compartment of the transition.
            - rate (float): The rate at which the transition occurs.
            - rate_name (str): The name of the rate for the transition.
            - agent (str, optional): The interacting agent for the transition (default is None).

        Raises:
        -------
            - ValueError: If the source, target, or agent (if provided) is not in the compartments list.
        """

        # Check that the source and target are in the compartment list
        missing_compartments = [comp for comp in [source, target, agent] if comp and comp not in self.compartments]
        if missing_compartments:
            raise ValueError(f"These compartments are not in the compartments list: {', '.join(missing_compartments)}")

        transition = Transition(source=source, target=target, rate=rate, rate_name=rate_name, agent=agent)

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
                    self.Cs[date][intervention["layer"]] *= intervention["reduction_factor"]
                elif intervention["new_matrix"] is not None:
                    self.Cs[date][intervention["layer"]] = intervention["new_matrix"]

        
    def simulate(self, population, start_date, end_date, steps, Nsim=100): 

        start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
        simulation_dates = pd.date_range(start=start_date, end=end_date, periods=steps).tolist()

        # apply interventions
        self.Cs = {date: {l: c for l, c in population.contact_matrices.items()} for date in simulation_dates}
        for intervention in self.interventions:
            self.apply_intervention(intervention, simulation_dates) 
        
        # compute overall contacts
        for date in self.Cs.keys(): 
            self.Cs[date]["overall"] = np.sum(np.array(list(self.Cs[date].values())), axis=0)





"""

    def stochastic_simulation(self, population, Cs, **kwargs): 
        


            # population in different compartments and demographic groups
            population = np.zeros((len(self.compartments), self.Nk), dtype='int')
            for comp in kwargs:
                population[self.pos[comp]] = kwargs[comp]

            compartments = [population]            # evolution of individuals in different compartments (t, n_comp, n_age)
            comps = list(self.transitions.nodes)   # compartments names
            Nk = population.sum(axis=0)            # total population per age groups

            # simulate
            for date in Cs.keys():

                # get today contacts matrix and seasonality adjustment
                C = Cs[date]["overall"]

                # last time step population in different compartments (n_comp, n_age)
                pop = compartments[-1]
                # next step solution
                new_pop = pop.copy()
                for comp in comps:
                    # transition (edge) attribute dict in 3-tuple (u, v, ddict) and initialize probabilities
                    trans = list(self.transitions.edges(comp, data=True))
                    prob = np.zeros((len(comps), self.n_age), dtype='float')

                    for _, node_j, data in trans:
                        # get source, target, and rate for this transition
                        source = self.pos[comp]
                        target = self.pos[node_j]
                        rate = data['rate']

                        # check if this transition has an interaction
                        if 'agent' in data.keys():
                            agent = self.pos[data['agent']]  # get agent position
                            interaction = np.array([np.sum(C[age, :] * pop[agent, :] / Nk) for age in range(self.n_age)])
                            rate *= interaction        # interaction term
                        prob[target, :] += rate

                    # prob of not transitioning
                    prob[source, :] = 1 - np.sum(prob, axis=0)

                    # compute transitions
                    delta = np.array([multinomial(pop[source][i], prob[:, i]) for i in range(self.n_age)])
                    delta[:, source] = 0
                    changes = np.sum(delta)

                    # if no changes continue to next iteration
                    if changes == 0:
                        continue

                    # update population of source of transition
                    new_pop[source, :] -= np.sum(delta, axis=1)

                    # update target compartments population
                    new_pop += delta.T

                compartments.append(new_pop)

            return np.array(compartments)

"""