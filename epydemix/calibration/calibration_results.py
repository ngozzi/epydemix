from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import datetime

@dataclass
class CalibrationResults:
    """
    Class to store and manage the results of a calibration process.

    Attributes:
        calibration_strategy: The strategy used for calibration
        posterior_distributions: Dictionary of posterior distributions per generation
        selected_trajectories: Dictionary of selected trajectories per generation
        observed_data: Observed data used for calibration
        priors: Dictionary of prior distributions for parameters
        calibration_params: Dictionary of parameters used in calibration
        distances: Dictionary of distances per generation
        weights: Dictionary of weights per generation
    """
    calibration_strategy: Optional[str] = None
    posterior_distributions: Dict[int, pd.DataFrame] = field(default_factory=dict)
    selected_trajectories: Dict[int, List[Any]] = field(default_factory=dict)
    observed_data: Optional[Any] = None
    priors: Dict[str, Any] = field(default_factory=dict)
    calibration_params: Dict[str, Any] = field(default_factory=dict)
    distances: Dict[int, List[Any]] = field(default_factory=dict)
    weights: Dict[int, List[Any]] = field(default_factory=dict)

    def _get_generation(self, generation: Optional[int], data_dict: Dict[int, Any]) -> Any:
        """Helper method to get data for a specific generation."""
        generations = list(data_dict.keys())
        if not generations:
            return None
            
        if generation is not None:
            if generation not in generations:
                raise ValueError(f"Generation {generation} not found, possible generations are {generations}.")
            return data_dict[generation]
        return data_dict[max(generations)]

    def get_posterior_distribution(self, generation: Optional[int] = None) -> pd.DataFrame:
        """Gets the posterior distribution DataFrame for a specific generation."""
        return self._get_generation(generation, self.posterior_distributions)

    def get_selected_trajectories(self, generation: Optional[int] = None) -> List[Any]:
        """Gets the selected trajectories for a specific generation."""
        return self._get_generation(generation, self.selected_trajectories)

    def get_weights(self, generation: Optional[int] = None) -> List[Any]:
        """Gets the weights for a specific generation."""
        return self._get_generation(generation, self.weights)

    def get_distances(self, generation: Optional[int] = None) -> List[Any]:
        """Gets the distances for a specific generation."""
        return self._get_generation(generation, self.distances)

    def get_stacked_trajectories(self, generation: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Stack individual simulation outputs into arrays.
        
        Args:
            generation: Specific generation to analyze. If None, uses latest
        
        Returns:
            Dict mapping variable names to arrays of shape (n_simulations, n_timesteps)
        """
        simulations = self.get_selected_trajectories(generation)
        if simulations is None or len(simulations) == 0:
            return {}
        
        return {
            key: np.stack([sim[key] for sim in simulations], axis=0)
            for key in simulations[0].keys()
        }

    def get_quantiles(self, 
                     dates: Optional[List[datetime.date]] = None,
                     quantiles: List[float] = [0.05, 0.5, 0.95],
                     generation: Optional[int] = None,
                     variables: Optional[List[str]] = None, 
                     quantile_axis: int = 0, 
                     length_axis: int = 1) -> pd.DataFrame:
        """
        Compute quantiles from stacked trajectories.

        Args:
            dates: List of dates. If None, no dates will be added to the output
            quantiles: List of quantiles
            generation: Specific generation to get trajectories from. If None, uses latest
            variables: List of variables to return. If None, returns all
            quantile_axis: Axis to compute quantiles on. Default is 0
            length_axis: Axis representing time dimension. Default is 1

        Returns:
            pd.DataFrame: DataFrame with dates, quantiles, and variable values
        """
        trajectories = self.get_stacked_trajectories(generation)
        if not trajectories:
            return pd.DataFrame()
            
        if variables:
            trajectories = {k: v for k, v in trajectories.items() if k in variables}
        
        if dates is None:
            dates = np.arange(trajectories[list(trajectories.keys())[0]].shape[length_axis])

        simulation_dates = []
        quantile_values = []
        for q in quantiles:
            simulation_dates.extend(dates)
            quantile_values.extend([q] * len(dates))

        data = {
            "date": simulation_dates,
            "quantile": quantile_values
        }
        
        for key, vals in trajectories.items():
            data[key] = [
                val for q in quantiles 
                for val in np.quantile(vals, q, axis=quantile_axis)
            ]

        return pd.DataFrame(data)

    def get_trajectories(self,
                        variables: Optional[List[str]] = None,
                        generation: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get raw trajectories for specified variables.

        Args:
            variables: List of variables to return. If None, returns all
            generation: Specific generation to get trajectories from. If None, uses latest

        Returns:
            Dictionary mapping variable names to trajectory arrays
        """
        trajectories = self.get_selected_trajectories(generation)
        if variables:
            return {k: v for k, v in trajectories.items() if k in variables}
        return trajectories
