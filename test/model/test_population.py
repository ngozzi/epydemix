import unittest
from epydemix.model.population import Population, load_epydemix_population

class TestPopulation(unittest.TestCase):
    def test_add_contact_matrix(self):
        self.population = Population()
        contact_matrix = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        layer_name = "work"
        self.population.add_contact_matrix(contact_matrix, layer_name)
        self.assertEqual(self.population.contact_matrices[layer_name], contact_matrix)

    def test_add_population(self):
        self.population = Population()
        Nk = [100, 200, 300]
        Nk_names = ["0-4", "5-9", "10-14"]
        self.population.add_population(Nk, Nk_names)
        self.assertEqual(self.population.Nk, Nk)
        self.assertEqual(self.population.Nk_names, Nk_names)

    def test_load_population(self):
        self.population = load_epydemix_population(
            population_name="United_States",
            contacts_source="prem_2021",
            path_to_data="../../epydemix_data/",
            layers=["school", "work", "home", "community"],
            age_group_mapping={
                "0-4": ["0-4"],
                "5-19": ["5-9", "10-14", "15-19"],
                "20-49": ["20-24", "25-29", "30-34", "35-39", "40-44", "45-49"],
                "50-64": ["50-54", "55-59", "60-64"],
                "65+": ["65-69", "70-74", "75+"]
            },
            supported_contacts_sources=["prem_2017", "prem_2021", "mistry_2021"],
            path_to_data_github="https://raw.githubusercontent.com/ngozzi/epydemix/main/epydemix_data/"
        )
        self.assertEqual(self.population.name, "United_States")
        self.assertEqual(list(self.population.contact_matrices.keys()), ["school", "work", "home", "community"])
        self.assertEqual(self.population.Nk.shape[0], 5)
        self.assertEqual(list(self.population.Nk_names), ["0-4", "5-19", "20-49", "50-64", "65+"])

if __name__ == '__main__':
    unittest.main()
