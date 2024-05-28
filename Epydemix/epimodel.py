# libraries
from .transition import Transition

class EpiModel:
    """
    EpiModel Class
    """

    def __init__(self):
        """
        EpiModel constructor
        """
        self.transitions = {}
        self.transitions_list = []
        self.interventions = []


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
        """
        transition = Transition(source=source, target=target, rate=rate, rate_name=rate_name, agent=agent)

        # Append to transitions list
        self.transitions_list.append(transition)

        # Add transition to the dictionary, initializing list if necessary
        if source in self.transitions:
            self.transitions[source].append(transition)
        else:
            self.transitions[source] = [transition]
