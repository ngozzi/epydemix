class Transition:
    """
    Epidemic Transition Class
    """

    def __init__(self, source, target, rate_name, agent=None):
        """
        Initializes the Transition object.

        Parameters:
        -----------
            - source (str): The source compartment of the transition.
            - target (str): The target compartment of the transition.
            - rate_name (str): The name of the rate for the transition.
            - agent (str, optional): The interacting agent for the transition (default is None).
        """
        self.source = source
        self.target = target
        self.rate_name = rate_name
        self.agent = agent
