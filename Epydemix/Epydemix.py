# libraries
import numpy as np
from datetime import datetime, timedelta
from numpy.random import multinomial
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from .population import Population
import os 
from collections import OrderedDict
from datetime import date, timedelta

class Epydemix:
    """
    Epydemix Class
    """

    def __init__(self, 
                 basin_name = "Indonesia", 
                 path_to_data = "./basins"):
        """
        Epydemix constructor
        """
        self.transitions = nx.MultiDiGraph()
        self.basin = Basin(name=basin_name, path_to_data=path_to_data)
        self.n_age = self.basin.contacts_matrix.shape[0]
        self.initialize_contacts(date.today() - timedelta(days=365), date.today())
        self.interventions = []
           
    def __repr__(self):
        text = 'Epidemic Model with %u compartments and %u transitions:\n\n' % \
               (self.transitions.number_of_nodes(),
                self.transitions.number_of_edges())

        for edge in self.transitions.edges(data=True):
            source, target, trans = edge[0], edge[1], edge[2]
            rate = trans['rate']

            if 'agent' in trans:
                agent = trans['agent']
                text += "%s + %s = %s %f\n" % (source, agent, target, rate)
            else:
                text += "%s -> %s %f\n" % (source, target, rate)

        return text

    def add_interaction(self, source, target, agent, rate, rate_name):
        """
        Add transition with interaction
        :param source: source compartment
        :param target: target compartment
        :param agent: interacting compartment (catalyzer)
        :param rate: rate of the transition
        :param rate_name: name of the rate
        """
        self.transitions.add_edge(source, target, agent=agent, rate=rate, rate_name=rate_name)

    def add_spontaneous(self, source, target, rate, rate_name):
        """
        Add a spontaneous transition
        :param source: source compartment
        :param target: target compartment
        :param rate: rate of the transition
        :param rate_name: name of the rate
        """
        self.transitions.add_edge(source, target, rate=rate, rate_name=rate_name)

    def apply_seasonality(self, day, seasonality_min, basin_hemispheres, seasonality_max=1):
        """
        This function computes the seasonality adjustment for transmissibility
            :param day: current day
            :param seasonality_min: seasonality parameter
            :param basin_hemispheres: hemisphere of basin (0: north, 1: tropical, 2; south)
            :param seasonality_max: seasonality parameter (default=1)
            :return: returns seasonality adjustment
        """
        s_r = seasonality_min / seasonality_max
        day_max_north, day_max_south = datetime(day.year, 1, 15), datetime(day.year, 7, 15)
        seasonal_adjustment = np.empty(shape=(3,), dtype=np.float64)
        # north hemisphere
        seasonal_adjustment[0] = 0.5 * ((1 - s_r) * np.sin(2 * np.pi / 365 * (day - day_max_north).days + 0.5 * np.pi) + 1 + s_r)
        # tropical hemisphere
        seasonal_adjustment[1] = 1.0
        # south hemisphere
        seasonal_adjustment[2] = 0.5 * ((1 - s_r) * np.sin(2 * np.pi / 365 * (day - day_max_south).days + 0.5 * np.pi) + 1 + s_r)
        return seasonal_adjustment[basin_hemispheres]

    def add_intervention(self, start_date, end_date, layer, factor): 
        self.interventions.append({"start_date": start_date, 
                                   "end_date": end_date, 
                                   "layer": layer, 
                                   "factor": factor})
        self.apply_interventions_to_contacts()
        
    def initialize_contacts(self, start_date, end_date):
        self.Cs = OrderedDict({date: {"home": self.basin.contacts_matrix_home, 
                                      "work": self.basin.contacts_matrix_work, 
                                      "school": self.basin.contacts_matrix_school, 
                                      "community": self.basin.contacts_matrix_community}  
                               for date in pd.date_range(start_date, end_date)})
        self.compute_overall_contacts()
        
    def compute_overall_contacts(self):
        for date in self.Cs.keys(): 
            self.Cs[date]["overall"] = self.Cs[date]["home"] + \
                                        self.Cs[date]["work"] + \
                                        self.Cs[date]["school"] + \
                                        self.Cs[date]["community"]

    def apply_interventions_to_contacts(self):
        for intervention in self.interventions:
            for date in pd.date_range(intervention["start_date"], intervention["end_date"]): 
                self.Cs[date][intervention["layer"]] = self.Cs[date][intervention["layer"]] * intervention["factor"]
        self.compute_overall_contacts()

    def simulate(self, start_date, end_date, Nsim=100, **kwargs):
        """
        Run stochastic simulations of the epidemic model
        :param kwargs:
        """
        self.initialize_contacts(start_date, end_date)
        self.apply_interventions_to_contacts()

        # number of age groups and compartments names and positions
        self.pos = {comp: i for i, comp in enumerate(self.transitions.nodes())}

        simulations = []
        for i in range(Nsim):
            compartments = self.stochastic_simulation(self.Cs, **kwargs)
            simulations.append(compartments)

        return np.array(simulations)

    def stochastic_simulation(self, Cs, **kwargs): 
        """
        Run stochastic simulations of the epidemic model
        :param kwargs:
        """

        # population in different compartments and age groups (n_comp, n_age)
        population = np.zeros((len(self.pos), self.n_age), dtype='int')
        for comp in kwargs:
            population[self.pos[comp]] = kwargs[comp]

        compartments = [population]            # evolution of individuals in different compartments (t, n_comp, n_age)
        comps = list(self.transitions.nodes)   # compartments names
        Nk = population.sum(axis=0)            # total population per age groups

        # simulate
        for date in Cs.keys():

            # get today contacts matrix and seasonality adjustment
            C = Cs[date]["overall"]

            # last time step population in different compartments (n_comp, n_age)
            pop = compartments[-1]
            # next step solution
            new_pop = pop.copy()
            for comp in comps:
                # transition (edge) attribute dict in 3-tuple (u, v, ddict) and initialize probabilities
                trans = list(self.transitions.edges(comp, data=True))
                prob = np.zeros((len(comps), self.n_age), dtype='float')

                for _, node_j, data in trans:
                    # get source, target, and rate for this transition
                    source = self.pos[comp]
                    target = self.pos[node_j]
                    rate = data['rate']

                    # check if this transition has an interaction
                    if 'agent' in data.keys():
                        agent = self.pos[data['agent']]  # get agent position
                        interaction = np.array([np.sum(C[age, :] * pop[agent, :] / Nk) for age in range(self.n_age)])
                        rate *= interaction        # interaction term
                    prob[target, :] += rate

                # prob of not transitioning
                prob[source, :] = 1 - np.sum(prob, axis=0)

                # compute transitions
                delta = np.array([multinomial(pop[source][i], prob[:, i]) for i in range(self.n_age)])
                delta[:, source] = 0
                changes = np.sum(delta)

                # if no changes continue to next iteration
                if changes == 0:
                    continue

                # update population of source of transition
                new_pop[source, :] -= np.sum(delta, axis=1)

                # update target compartments population
                new_pop += delta.T

            compartments.append(new_pop)

        return np.array(compartments)
        
    def get_spectral_radius(self, layer): 
        return [np.linalg.eigvals(self.Cs[date][layer]).real.max() for date in self.Cs.keys()]

    def plot_spectral_radius(self, layers=["overall"], ax=None, normalize=True):
        if ax == None: 
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        for layer in layers: 
            eigvals = self.get_spectral_radius(layer)
            if normalize: 
                eigvals /= eigvals[0]
            ax.plot(self.Cs.keys(), eigvals, label=layer)

        if normalize: 
            ylabel = "$\\rho(C_t) / \\rho(C_t0)$"
        else: 
            ylabel = "$\\rho(C_t)$"
        ax.set_ylabel(ylabel)
        ax.legend()