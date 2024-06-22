import pandas as pd

class CalibrationResults(): 

    def __init__(self):
        self.calibration_strategy = None
        self.posterior_distribution = pd.DataFrame()
        self.selected_trajectories = []
        self.selected_quantiles = pd.DataFrame()
        self.data = []
        self.priors = {}
        self.calibration_params = {}

    def set_calibration_strategy(self, strategy):
        self.calibration_strategy = strategy

    def set_posterior_distribution(self, posterior_distribution):
        self.posterior_distribution = posterior_distribution

    def set_selected_trajectories(self, selected_trajectories):
        self.selected_trajectories = selected_trajectories

    def set_selected_quantiles(self, selected_quantiles):
        self.selected_quantiles = selected_quantiles

    def set_data(self, data):
        self.data = data

    def set_priors(self, priors):
        self.priors = priors

    def set_calibration_params(self, calibration_params):
        self.calibration_params = calibration_params

    def get_calibration_strategy(self):
        return self.calibration_strategy

    def get_posterior_distribution(self):
        return self.posterior_distribution

    def get_selected_trajectories(self):
        return self.selected_trajectories 

    def get_selected_quantiles(self):
        return self.selected_quantiles 

    def get_data(self):
        return self.data

    def get_priors(self):
        return self.priors 

    def get_calibration_params(self):
        return self.calibration_params