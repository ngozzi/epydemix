from typing import Optional

class Transition:
    """
    Represents a transition in an epidemic model.

    A transition defines a change from one compartment to another with a specified rate. Optionally, it can involve a specific agent.

    Attributes:
        source (str): The source compartment of the transition.
        target (str): The target compartment of the transition.
        rate (str): The expression representing the rate of the transition.
        agent (Optional[str]): The agent involved in the transition, if any. Defaults to None.
    """

    def __init__(self, source: str, target: str, rate: str, agent: Optional[str] = None) -> None:
        """
        Initializes the Transition object.

        Args:
            source (str): The source compartment of the transition.
            target (str): The target compartment of the transition.
            rate (str): The expression representing the rate of the transition.
            agent (Optional[str]): The interacting agent for the transition. Defaults to None.

        Returns:
            None
        """
        self.source = source
        self.target = target
        self.rate = rate
        self.agent = agent
