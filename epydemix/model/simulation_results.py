import pandas as pd
from typing import List, Dict, Any, Optional

class SimulationResults:
    """
    A class to store and manage results from a simulation.

    Attributes:
        Nsim (int): The number of simulations run.
        full_trajectories (List[Any]): List of full trajectory data from simulations.
        df_quantiles (pd.DataFrame): DataFrame containing quantiles of simulation results.
        compartment_idx (Dict[str, int]): Dictionary mapping compartment names to indices.
        parameters (Dict[str, Any]): Dictionary of parameters used in the simulation.
    """

    def __init__(self) -> None:
        """
        Initializes the SimulationResults object with default values.

        Attributes:
            Nsim (int): Initialized to 0.
            full_trajectories (List[Any]): Initialized to an empty list.
            df_quantiles (pd.DataFrame): Initialized to an empty DataFrame.
            compartment_idx (Dict[str, int]): Initialized to an empty dictionary.
            parameters (Dict[str, Any]): Initialized to an empty dictionary.
        """
        self.Nsim = 0 
        self.full_trajectories = []
        self.df_quantiles = pd.DataFrame()
        self.compartment_idx = {}
        self.parameters = {}

    def set_Nsim(self, Nsim: int) -> None:
        """
        Sets the number of simulations.

        Args:
            Nsim (int): The number of simulations to set.

        Returns:
            None
        """
        self.Nsim = Nsim

    def set_df_quantiles(self, df_quantiles: pd.DataFrame) -> None:
        """
        Sets the DataFrame containing quantiles of simulation results.

        Args:
            df_quantiles (pd.DataFrame): DataFrame with quantiles to set.

        Returns:
            None
        """
        self.df_quantiles = df_quantiles

    def set_full_trajectories(self, full_trajectories: List[Any]) -> None:
        """
        Sets the list of full trajectory data from simulations.

        Args:
            full_trajectories (List[Any]): List of full trajectory data to set.

        Returns:
            None
        """
        self.full_trajectories = full_trajectories 

    def set_compartment_idx(self, compartment_idx: Dict[str, int]) -> None:
        """
        Sets the dictionary mapping compartment names to indices.

        Args:
            compartment_idx (Dict[str, int]): Dictionary of compartment names and indices to set.

        Returns:
            None
        """
        self.compartment_idx = compartment_idx

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Sets the dictionary of parameters used in the simulation.

        Args:
            parameters (Dict[str, Any]): Dictionary of parameters to set.

        Returns:
            None
        """
        self.parameters = parameters

    def get_Nsim(self) -> int:
        """
        Retrieves the number of simulations.

        Returns:
            int: The number of simulations.
        """
        return self.Nsim 

    def get_df_quantiles(self) -> pd.DataFrame:
        """
        Retrieves the DataFrame containing quantiles of simulation results.

        Returns:
            pd.DataFrame: The DataFrame with quantiles.
        """
        return self.df_quantiles 

    def get_full_trajectories(self) -> List[Any]:
        """
        Retrieves the list of full trajectory data from simulations.

        Returns:
            List[Any]: The list of full trajectory data.
        """
        return self.full_trajectories 

    def get_compartment_idx(self) -> Dict[str, int]:
        """
        Retrieves the dictionary mapping compartment names to indices.

        Returns:
            Dict[str, int]: The dictionary of compartment names and indices.
        """
        return self.compartment_idx 

    def get_parameters(self) -> Dict[str, Any]:
        """
        Retrieves the dictionary of parameters used in the simulation.

        Returns:
            Dict[str, Any]: The dictionary of parameters.
        """
        return self.parameters
