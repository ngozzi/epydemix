import pandas as pd
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
        self.posterior_distribution = pd.DataFrame()
        self.selected_trajectories: List[Any] = []
        self.selected_quantiles = pd.DataFrame()
        self.data: List[Any] = []
        self.priors: Dict[str, Any] = {}
        self.calibration_params: Dict[str, Any] = {}
        self.errors: List[Any] = []

    def set_calibration_strategy(self, strategy: str) -> None:
        """
        Sets the calibration strategy.

        Args:
        -----
        strategy : str
            The strategy used for calibration.
        """
        self.calibration_strategy = strategy

    def set_posterior_distribution(self, posterior_distribution: pd.DataFrame) -> None:
        """
        Sets the posterior distribution DataFrame.

        Args:
        -----
        posterior_distribution : pd.DataFrame
            DataFrame containing the posterior distribution of the parameters.
        """
        self.posterior_distribution = posterior_distribution

    def set_selected_trajectories(self, selected_trajectories: List[Any]) -> None:
        """
        Sets the selected trajectories.

        Args:
        -----
        selected_trajectories : List[Any]
            List of selected trajectories from the calibration.
        """
        self.selected_trajectories = selected_trajectories

    def set_selected_quantiles(self, selected_quantiles: pd.DataFrame) -> None:
        """
        Sets the selected quantiles DataFrame.

        Args:
        -----
        selected_quantiles : pd.DataFrame
            DataFrame containing selected quantiles from the calibration.
        """
        self.selected_quantiles = selected_quantiles

    def set_data(self, data: List[Any]) -> None:
        """
        Sets the data used for calibration.

        Args:
        -----
        data : List[Any]
            List of data used for calibration.
        """
        self.data = data

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

    def set_error_distribution(self, errors: List[Any]) -> None:
        """
        Sets the error distributions or error metrics.

        Args:
        -----
        errors : List[Any]
            List of error distributions or error metrics.
        """
        self.errors = errors

    def get_calibration_strategy(self) -> str:
        """
        Gets the calibration strategy.

        Returns:
        --------
        str
            The strategy used for calibration.
        """
        return self.calibration_strategy

    def get_posterior_distribution(self) -> pd.DataFrame:
        """
        Gets the posterior distribution DataFrame.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the posterior distribution of the parameters.
        """
        return self.posterior_distribution

    def get_selected_trajectories(self) -> List[Any]:
        """
        Gets the selected trajectories.

        Returns:
        --------
        List[Any]
            List of selected trajectories from the calibration.
        """
        return self.selected_trajectories 

    def get_selected_quantiles(self) -> pd.DataFrame:
        """
        Gets the selected quantiles DataFrame.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing selected quantiles from the calibration.
        """
        return self.selected_quantiles 

    def get_data(self) -> List[Any]:
        """
        Gets the data used for calibration.

        Returns:
        --------
        List[Any]
            List of data used for calibration.
        """
        return self.data

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
    
    def get_error_distribution(self) -> List[Any]:
        """
        Gets the error distributions or error metrics.

        Returns:
        --------
        List[Any]
            List of error distributions or error metrics.
        """
        return self.errors
