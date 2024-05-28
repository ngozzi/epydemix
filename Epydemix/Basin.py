# libraries
import numpy as np
import pandas as pd
import os
from datetime import timedelta, datetime
from collections import OrderedDict


class Basin:
    """
    Basin object class
    """

    def __init__(self, name, path_to_data):
        """
        Basin object constructor
        :param name: name of the basin
        :param path_to_data: path to basin data
        """

        # name, hemisphere
        self.name = name
        self.hemisphere = pd.read_csv(os.path.join(path_to_data, name, "hemisphere.csv")).hemisphere_code.values[0]

        # contacts matrix
        self.contacts_matrix = np.load(os.path.join(path_to_data, name, "contacts-matrix/contacts_matrix_all.npz"))["arr_0"]
        self.contacts_matrix_home = np.load(os.path.join(path_to_data, name, "contacts-matrix/contacts_matrix_home.npz"))["arr_0"]
        self.contacts_matrix_work = np.load(os.path.join(path_to_data, name, "contacts-matrix/contacts_matrix_work.npz"))["arr_0"]
        self.contacts_matrix_community = np.load(os.path.join(path_to_data, name, "contacts-matrix/contacts_matrix_community.npz"))["arr_0"]
        self.contacts_matrix_school = np.load(os.path.join(path_to_data, name, "contacts-matrix/contacts_matrix_school.npz"))["arr_0"]

        # demographic
        self.Nk = pd.read_csv(os.path.join(path_to_data, name, "demographic/Nk.csv")).value.values


    def update_contacts(self, date):
        """
        This functions rescale the contacts matrix for a specific date according to the reduction factor
        :param date: date
        :return: contacts matrix for the given date
        """
        reduction_factor = self.reductions.loc[self.reductions.year_week == date.strftime('%G-%V')]["red"].values[0]
        return reduction_factor * self.contacts_matrix

    def compute_contacts(self, start_date, end_date):
        """
        This function computes contacts matrices over a given time window
        :param start_date: initial date
        :param end_date: last date
        """
        dates = np.arange(start_date, end_date, timedelta(days=1)).astype(datetime)
        self.Cs = OrderedDict({date: self.update_contacts(date) for date in dates})
