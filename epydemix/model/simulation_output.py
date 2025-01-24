from dataclasses import dataclass
from typing import Dict, List, Any
import pandas as pd
import numpy as np

@dataclass
class Trajectory:
    """
    Class to store a single trajectory data.
    
    Attributes:
        compartments (Dict[str, np.ndarray]): Dictionary mapping compartment names to arrays of shape (timesteps,)
        transitions (Dict[str, np.ndarray]): Dictionary mapping transition names to arrays of shape (timesteps,)
        dates (List[pd.Timestamp]): List of simulation dates
        compartment_idx (Dict[str, int]): Dictionary mapping compartment names to indices
        transitions_idx (Dict[str, int]): Dictionary mapping transition names to indices
        parameters (Dict[str, Any]): Dictionary of parameters used in the simulation
    """
    compartments: Dict[str, np.ndarray]
    transitions: Dict[str, np.ndarray]
    dates: List[pd.Timestamp]
    compartment_idx: Dict[str, int]
    transitions_idx: Dict[str, int]
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

        # Resample compartments
        df_comp = pd.DataFrame(self.compartments, index=self.dates)
        df_comp_resampled = df_comp.resample(freq).agg(method)
        
        # Resample transitions
        df_trans = pd.DataFrame(self.transitions, index=self.dates)
        df_trans_resampled = df_trans.resample(freq).agg(method)
        
        # Handle NaN values
        if fill_method == 'interpolate':
            df_comp_resampled = df_comp_resampled.interpolate(method='linear')
            df_trans_resampled = df_trans_resampled.interpolate(method='linear')
            # Handle edge cases
            df_comp_resampled = df_comp_resampled.fillna(method='ffill').fillna(method='bfill')
            df_trans_resampled = df_trans_resampled.fillna(method='ffill').fillna(method='bfill')
        else:
            df_comp_resampled = df_comp_resampled.fillna(method=fill_method)
            df_trans_resampled = df_trans_resampled.fillna(method=fill_method)

        return Trajectory(
            compartments={k: np.array(v) for k, v in df_comp_resampled.items()},
            transitions={k: np.array(v) for k, v in df_trans_resampled.items()},
            dates=df_comp_resampled.index.tolist(),
            compartment_idx=self.compartment_idx,
            transitions_idx=self.transitions_idx,
            parameters=self.parameters
        ) 