from dataclasses import dataclass
from typing import Dict, List, Any
import pandas as pd
import numpy as np

@dataclass
class Trajectory:
    """
    Class to store a single trajectory data.
    
    Attributes:
        data (Dict[str, np.ndarray]): Dictionary mapping compartment names to arrays of shape (timesteps, demographics)
        dates (List[pd.Timestamp]): List of simulation dates
        compartment_idx (Dict[str, int]): Dictionary mapping compartment names to indices
        parameters (Dict[str, Any]): Dictionary of parameters used in the simulation
    """
    data: Dict[str, np.ndarray]
    dates: List[pd.Timestamp]
    compartment_idx: Dict[str, int]
    parameters: Dict[str, Any]

    def resample(self, freq: str, method: str = 'last', fill_method: str = 'ffill') -> 'Trajectory':
        """
        Resample trajectory to new frequency.
        
        Args:
            freq (str): Frequency for resampling (e.g., 'D' for daily, 'W' for weekly)
            method (str): Aggregation method for resampling. Default is 'last'
            fill_method (str): Method to fill NaN values after resampling. Options are:
                - 'ffill': Forward fill (use last valid observation)
                - 'bfill': Backward fill (use next valid observation)
                - 'interpolate': Linear interpolation between points
                Default is 'ffill'.

        Returns:
            Trajectory: New trajectory with resampled data
            
        Raises:
            ValueError: If fill_method is not one of ['ffill', 'bfill', 'interpolate']
        """
        if fill_method not in ['ffill', 'bfill', 'interpolate']:
            raise ValueError("fill_method must be one of ['ffill', 'bfill', 'interpolate']")

        # Convert to DataFrame for resampling
        df = pd.DataFrame(self.data, index=self.dates)
        df_resampled = df.resample(freq).agg(method)
        
        # Handle NaN values based on fill_method
        if fill_method == 'interpolate':
            df_resampled = df_resampled.interpolate(method='linear')
            # Handle edge cases where interpolation can't fill (beginning/end)
            df_resampled = df_resampled.fillna(method='ffill').fillna(method='bfill')
        else:
            df_resampled = df_resampled.fillna(method=fill_method)
        
        # Convert back to original format
        resampled_data = {
            comp_name: np.array(df_resampled[comp_name].tolist()) 
            for comp_name in self.data.keys()
        }

        # Return the trajectory
        return Trajectory(
            data=resampled_data,
            dates=df_resampled.index.tolist(),
            compartment_idx=self.compartment_idx,
            parameters=self.parameters
        ) 