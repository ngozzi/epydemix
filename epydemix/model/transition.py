from typing import Union, List, Optional, Dict

class Transition:
    """
    Represents a transition in an epidemic model.

    A transition defines a change from one compartment to another with a specified rate. Optionally, it can involve a specific agent.

    Attributes:
        source (str): The source compartment of the transition.
        target (str): The target compartment of the transition.
        kind (str): The kind of transition (e.g., spontaneous or mediated).
        rate (Union[str, List[str]]): The expression representing the rate(s) of the transition.
        params (Dict): The dictionary of parameters involved in the transition.
    """

    def __init__(self, source: str, target: str, kind: str, rate: Union[str, List[str]], params: Optional[Dict]=None) -> None:
        """
        Initializes the Transition object.

        Args:
            source (str): The source compartment of the transition.
            target (str): The target compartment of the transition.
            kind (str): The kind of transition (e.g., spontaneous or mediated).
            rate (Union[str, List[str]]): The expression representing the rate(s) of the transition.
            params (Dict): The dictionary of parameters involved in the transition.

        Returns:
            None
        """
        self.source = source
        self.target = target
        self.kind = kind
        if not isinstance(rate, list):
            self.rate = [rate]
        else:
            self.rate = rate
        if params is None:
            self.params = {}
        else:
            self.params = params