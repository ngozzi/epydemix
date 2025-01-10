import pandas as pd
import numpy as np
from typing import List, Dict, Any

class CalibrationResults:
    """
    Class to store and manage the results of a calibration process.

    Attributes:
    -----------
    calibration_strategy : str or None
        The strategy used for calibration.
    posterior_distribution : pd.DataFrame
        DataFrame containing the posterior distribution of the parameters.
    selected_trajectories : List[Any]
        List of selected trajectories from the calibration.
    selected_quantiles : pd.DataFrame
        DataFrame containing selected quantiles from the calibration.
    data : List[Any]
        List of data used for calibration.
    priors : Dict[str, Any]
        Dictionary of prior distributions for the parameters.
    calibration_params : Dict[str, Any]
        Dictionary of parameters used in the calibration process.
    errors : List[Any]
        List of error distributions or error metrics.
    """

    def __init__(self):
        """
        Initializes the CalibrationResults object with default values.
        """
        self.calibration_strategy = None
        self.posterior_distributions = {}
        self.selected_trajectories = {}
        #self.selected_quantiles = pd.DataFrame()
        self.observed_data = None
        self.priors = {}
        self.calibration_params = {}
        self.distances = {}
        self.weights = {}   

    def set_calibration_strategy(self, strategy: str) -> None:
        """
        Sets the calibration strategy.

        Args:
        -----
        strategy : str
            The strategy used for calibration.
        """
        self.calibration_strategy = strategy

    def set_posterior_distribution(self, posterior_distribution, generation) -> None:
        """
        Sets the posterior distribution DataFrame.

        Args:
        -----
        posterior_distribution : pd.DataFrame
            DataFrame containing the posterior distribution of the parameters.
        """ 
        self.posterior_distributions[generation] = posterior_distribution

    def set_selected_trajectories(self, selected_trajectories, generation) -> None:
        """
        Sets the selected trajectories.

        Args:
        -----
        selected_trajectories : List[Any]
            List of selected trajectories from the calibration.
        """
        self.selected_trajectories[generation] = selected_trajectories

    def set_weights(self, weights, generation) -> None:
        """
        Sets the weights.

        Args:
        -----
        weights : List[Any]
            List of selected trajectories from the calibration.
        """
        self.weights[generation] = weights

    def set_selected_quantiles(self, selected_quantiles: pd.DataFrame) -> None:
        """
        Sets the selected quantiles DataFrame.

        Args:
        -----
        selected_quantiles : pd.DataFrame
            DataFrame containing selected quantiles from the calibration.
        """
        self.selected_quantiles = selected_quantiles

    def set_observed_data(self, observed_data) -> None:
        """
        Sets the data used for calibration.

        Args:
        -----
        data : List[Any]
            List of data used for calibration.
        """
        self.observed_data = observed_data

    def set_priors(self, priors: Dict[str, Any]) -> None:
        """
        Sets the prior distributions for the parameters.

        Args:
        -----
        priors : Dict[str, Any]
            Dictionary of prior distributions for the parameters.
        """
        self.priors = priors

    def set_calibration_params(self, calibration_params: Dict[str, Any]) -> None:
        """
        Sets the calibration parameters.

        Args:
        -----
        calibration_params : Dict[str, Any]
            Dictionary of parameters used in the calibration process.
        """
        self.calibration_params = calibration_params

    def set_distances(self, distances, generation) -> None:
        """
        Sets the error distributions or error metrics.

        Args:
        -----
        errors : List[Any]
            List of error distributions or error metrics.
        """
        self.distances[generation] = distances

    def get_calibration_strategy(self) -> str:
        """
        Gets the calibration strategy.

        Returns:
        --------
        str
            The strategy used for calibration.
        """
        return self.calibration_strategy

    def get_posterior_distribution(self, generation=None) -> pd.DataFrame:
        """
        Gets the posterior distribution DataFrame.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the posterior distribution of the parameters.
        """
        generations = list(self.posterior_distributions.keys())

        if generation is not None:
            if generation not in generations:
                raise ValueError(f"Generation {generation} not found, possible geneartions are {generations}.")
            return self.posterior_distributions[generation]
        else: 
            # get max generation
            max_generation = max(generations)
            return self.posterior_distributions[max_generation]
        
    def get_weights(self, generation=None) -> pd.DataFrame:
        """
        Gets the posterior distribution DataFrame.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the posterior distribution of the parameters.
        """
        generations = list(self.weights.keys())

        if generation is not None:
            if generation not in generations:
                raise ValueError(f"Generation {generation} not found, possible geneartions are {generations}.")
            return self.weights[generation]
        else: 
            # get max generation
            max_generation = max(generations)
            return self.weights[max_generation]


    def get_selected_trajectories(self, generation=None) -> List[Any]:
        """
        Gets the selected trajectories.

        Returns:
        --------
        List[Any]
            List of selected trajectories from the calibration.
        """
        generations = list(self.selected_trajectories.keys())

        if generation is not None:
            if generation not in generations:
                raise ValueError(f"Generation {generation} not found, possible geneartions are {generations}.")
            return self.selected_trajectories[generation]
        else: 
            # get max generation
            max_generation = max(generations)
            return self.selected_trajectories[max_generation]

    def get_selected_quantiles(self) -> pd.DataFrame:
        """
        Gets the selected quantiles DataFrame.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing selected quantiles from the calibration.
        """
        return self.selected_quantiles 

    def get_observed_data(self) -> List[Any]:
        """
        Gets the data used for calibration.

        Returns:
        --------
        List[Any]
            List of data used for calibration.
        """
        return self.observed_data

    def get_priors(self) -> Dict[str, Any]:
        """
        Gets the prior distributions.

        Returns:
        --------
        Dict[str, Any]
            Dictionary of prior distributions for the parameters.
        """
        return self.priors 

    def get_calibration_params(self) -> Dict[str, Any]:
        """
        Gets the calibration parameters.

        Returns:
        --------
        Dict[str, Any]
            Dictionary of parameters used in the calibration process.
        """
        return self.calibration_params
    
    def get_distances(self, generation=None) -> List[Any]:
        """
        Gets the error distributions or error metrics.

        Returns:
        --------
        List[Any]
            List of error distributions or error metrics.
        """
        generations = list(self.distances.keys())   

        if generation is not None:
            if generation not in generations:
                raise ValueError(f"Generation {generation} not found, possible geneartions are {generations}.")
            return self.distances[generation]
        else: 
            # get max generation
            max_generation = max(generations)
            return self.distances[max_generation]
