import pandas as pd

class SimulationResults(): 

    def __init__(self):
        self.Nsim = 0 
        self.full_trajectories = []
        self.df_quantiles = pd.DataFrame()
        self.compartment_idx = {}
        self.parameters = {}

    def set_Nsim(self, Nsim):
        self.Nsim = Nsim

    def set_df_quantiles(self, df_quantiles):
        self.df_quantiles = df_quantiles

    def set_full_trajectories(self, full_trajectories):
        self.full_trajectories = full_trajectories 

    def set_compartment_idx(self, compartment_idx): 
        self.compartment_idx = compartment_idx

    def set_parameters(self, parameters):
        self.parameters = parameters

    def get_Nsim(self):
        return self.Nsim 

    def get_df_quantiles(self):
        return self.df_quantiles 

    def get_full_trajectories(self):
        return self.full_trajectories 

    def get_compartment_idx(self): 
        return self.compartment_idx 

    def get_parameters(self): 
        return self.parameters