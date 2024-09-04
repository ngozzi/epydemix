import unittest
from epydemix.model.epimodel import EpiModel, SimulationResults

class TestEpiModel(unittest.TestCase):

    def create_model(self):
        # Create an instance of EpiModel for testing
        model = EpiModel(compartments=["Susceptible", "Infected", "Recovered"])
        return model

    def test_add_transition(self):
        # Create the model
        model = self.create_model()

        # Add a transition between compartments
        model.add_transition(source="Susceptible", target="Infected", rate=0.3, agent="Infected")
        
        # Check if the transition was added successfully
        transitions = model.transitions
        self.assertIn("Susceptible", transitions)
        self.assertEqual("Infected", transitions["Susceptible"][0].target)
        self.assertEqual(transitions["Susceptible"][0].agent, "Infected")

    def test_run_simulations(self):
        # Create the model
        model = self.create_model()

        # Add compartments and transitions to the model
        model.add_transition(source="Susceptible", target="Infected", rate=0.3, agent="Infected")
        model.add_transition(source="Infected", target="Recovered", rate=0.1)
        
        # Run simulations with specified time frame and initial conditions
        results = model.run_simulations(
            start_date="2019-12-01", 
            end_date="2020-04-01", 
            Susceptible=99990, 
            Infected=10
        )
        
        # Check if the simulation results are valid
        self.assertIsNotNone(results)
        self.assertIsInstance(results, SimulationResults)

    def test_override_parameter(self):
        # Create the model
        model = self.create_model()

        # Add a parameter to the model
        model.add_parameter(name="transmission_rate", value=0.5)
        
        # Override the parameter for a specific time period
        model.override_parameter(start_date="2020-01-01", end_date="2020-02-01", name="transmission_rate", value=0.2)
        
        # Check if the parameter override was applied correctly
        overrides = model.overrides
        self.assertIn("transmission_rate", overrides)
        self.assertEqual(overrides["transmission_rate"][0]["start_date"], "2020-01-01")
        self.assertEqual(overrides["transmission_rate"][0]["end_date"], "2020-02-01")
        self.assertEqual(overrides["transmission_rate"][0]["value"], 0.2)

    def test_clear_compartments(self):
        # Create the model
        model = self.create_model()

        # Add compartments to the model
        model.add_compartments(["Susceptible", "Infected", "Recovered"])
        
        # Clear the compartments
        model.clear_compartments()
        
        # Check if the compartments were cleared successfully
        compartments = model.compartments
        self.assertEqual(len(compartments), 0)

    def test_clear_transitions(self):
        # Create the model
        model = self.create_model()

        # Add a transition between compartments
        model.add_transition(source="Susceptible", target="Infected", rate=0.3, agent="Infected")
        
        # Clear the transitions
        model.clear_transitions()
        
        # Check if the transitions were cleared successfully
        transitions_list = model.transitions_list
        self.assertEqual(len(transitions_list), 0)

    def test_clear_parameters(self):
        # Create the model
        model = self.create_model()

        # Add a parameter to the model
        model.add_parameter(name="transmission_rate", value=0.5)
        
        # Clear the parameters
        model.clear_parameters()
        
        # Check if the parameters were cleared successfully
        parameters = model.parameters
        self.assertEqual(len(parameters), 0)

    def test_clear_overrides(self):
        # Create the model
        model = self.create_model()

        # Add a parameter and override it for a specific time period
        model.add_parameter(name="transmission_rate", value=0.5)
        model.override_parameter(start_date="2020-01-01", end_date="2020-02-01", name="transmission_rate", value=0.2)
        
        # Clear the overrides
        model.clear_overrides()
        
        # Check if the overrides were cleared successfully
        overrides = model.overrides
        self.assertEqual(len(overrides), 0)

if __name__ == '__main__':
    unittest.main()