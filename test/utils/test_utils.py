import unittest
import numpy as np
import pandas as pd
from datetime import date
from epydemix.utils.utils import (
    validate_parameter_shape, resize_parameter, create_definitions, compute_quantiles, 
    format_simulation_output, combine_simulation_outputs, str_to_date, apply_overrides,
    generate_unique_string, compute_days, evaluate, compute_simulation_dates, apply_initial_conditions,
    convert_to_2Darray
)

class TestUtils(unittest.TestCase):
    
    def test_validate_parameter_shape(self):
        # Test valid scalar input
        self.assertIsNone(validate_parameter_shape('param1', 10, 5, 3))

        # Test valid 1D array input
        arr_1d = np.array([1, 2, 3, 4, 5])
        self.assertIsNone(validate_parameter_shape('param1', arr_1d, 5, 3))

        # Test valid 2D array input
        arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertIsNone(validate_parameter_shape('param2', arr_2d, 3, 3))

        # Test invalid 2D array shape
        arr_2d_wrong = np.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            validate_parameter_shape('param2', arr_2d_wrong, 3, 3)

    def test_resize_parameter(self):
        # Test scalar input resizing
        result = resize_parameter(10, 5, 3)
        self.assertEqual(result.shape, (5, 3))

        # Test 1D array resizing
        arr_1d = np.array([1, 2, 3, 4, 5])
        result = resize_parameter(arr_1d, 5, 3)
        self.assertEqual(result.shape, (5, 3))

    def test_create_definitions(self):
        params = {
            'param1': np.array([1, 2, 3]),
            'param2': 5
        }
        result = create_definitions(params, 3, 2)
        self.assertEqual(result['param1'].shape, (3, 2))
        self.assertEqual(result['param2'].shape, (3, 2))

    def test_compute_quantiles(self):
        data = {'comp1': [np.random.rand(100, 10)], 'comp2': [np.random.rand(100, 10)]}
        dates = pd.date_range(start='2020-01-01', periods=100).tolist()
        result = compute_quantiles(data, dates)
        self.assertIsInstance(result, pd.DataFrame)

    def test_format_simulation_output(self):
        simulation_output = np.random.rand(10, 3, 5)  # 10 time steps, 3 compartments, 5 demographic groups
        compartments_idx = {'S': 0, 'I': 1, 'R': 2}
        Nk_names = ['Group1', 'Group2', 'Group3', 'Group4', 'Group5']
        result = format_simulation_output(simulation_output, compartments_idx, Nk_names)
        self.assertEqual(result['S_Group1'].shape, (10,))

    def test_combine_simulation_outputs(self):
        combined = {}
        output = {'S_Group1': np.random.rand(10)}
        result = combine_simulation_outputs(combined, output)
        self.assertIn('S_Group1', result)

    def test_str_to_date(self):
        result = str_to_date("2020-01-01")
        self.assertEqual(result, date(2020, 1, 1))

    def test_apply_overrides(self):
        definitions = {'param1': np.ones((10, 3))}
        overrides = {'param1': [{'start_date': '2020-01-01', 'end_date': '2020-01-05', 'value': 5}]}
        dates = pd.date_range(start='2020-01-01', periods=10).to_list()
        result = apply_overrides(definitions, overrides, dates)
        self.assertTrue((result['param1'][:5] == 5).all())

    def test_generate_unique_string(self):
        result = generate_unique_string(12)
        self.assertEqual(len(result), 12)
        self.assertTrue(result.isalpha())

    def test_compute_days(self):
        result = compute_days('2020-01-01', '2020-01-10')
        self.assertEqual(result, 10)

    def test_evaluate(self):
        result = evaluate("x * y", {'x': 2, 'y': 3})
        self.assertEqual(result, 6)

    def test_compute_simulation_dates(self):
        result = compute_simulation_dates("2020-01-01", "2020-01-10")
        self.assertEqual(len(result), 10)

    def test_apply_initial_conditions(self):
        class MockEpiModel:
            def __init__(self):
                self.compartments = ['S', 'I', 'R']
                self.compartments_idx = {'S': 0, 'I': 1, 'R': 2}
                self.population = type('', (), {})()
                self.population.Nk = [1000, 2000, 3000]
        
        epimodel = MockEpiModel()
        kwargs = {'S': [500, 1000, 1500], 'I': [200, 300, 500], 'R': [100, 200, 400]}
        result = apply_initial_conditions(epimodel, **kwargs)
        self.assertEqual(result.shape, (3, 3))

    def test_convert_to_2Darray(self):
        lst = [1, 2, 3, 4]
        result = convert_to_2Darray(lst)
        self.assertEqual(result.shape, (1, 4))
        self.assertTrue((result == np.array([[1, 2, 3, 4]])).all())

if __name__ == '__main__':
    unittest.main()
