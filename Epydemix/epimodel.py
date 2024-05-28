# libraries
from .transition import Transition

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
