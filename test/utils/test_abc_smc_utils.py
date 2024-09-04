import unittest
import numpy as np
from scipy import stats
from epydemix.utils.abc_smc_utils import (
    initialize_particles, default_perturbation_kernel, compute_covariance_matrix,
    compute_weights, compute_effective_sample_size, silverman_rule_of_thumb,
    weighted_quantile
)

class TestABCUtils(unittest.TestCase):

    def test_initialize_particles(self):
        priors = {
            'param1': stats.uniform(0, 1),
            'param2': stats.norm(0, 1)
        }
        num_particles = 100
        particles, weights, tolerance = initialize_particles(priors, num_particles)
        
        self.assertEqual(len(particles), num_particles)
        self.assertTrue(np.allclose(weights, np.ones(num_particles) / num_particles))
        self.assertTrue(np.isinf(tolerance))

    def test_default_perturbation_kernel(self):
        particle = {'param1': 0.5, 'param2': 1.0}
        continuous_params = ['param1']
        discrete_params = ['param2']
        cov_matrix = np.array([[0.1]])
        priors = {
            'param1': stats.uniform(0, 1),
            'param2': stats.randint(0, 3)
        }
        p_discrete_transition = 0.5
        
        perturbed_particle = default_perturbation_kernel(
            particle, continuous_params, discrete_params, cov_matrix, priors, p_discrete_transition
        )
        self.assertIn('param1', perturbed_particle)
        self.assertIn('param2', perturbed_particle)

    def test_compute_covariance_matrix(self):
        particles = [{'param1': 0.1, 'param2': 1.2}, {'param1': 0.2, 'param2': 1.3}, {'param1': 0.3, 'param2': 1.4}]
        continuous_params = ['param1', 'param2']
        weights = np.array([0.2, 0.5, 0.3])
        
        cov_matrix = compute_covariance_matrix(particles, continuous_params, weights)
        self.assertEqual(cov_matrix.shape, (2, 2))

    def test_compute_weights(self):
        new_particles = [{'param1': 0.1, 'param2': 1}, {'param1': 0.2, 'param2': 2}]
        old_particles = [{'param1': 0.05, 'param2': 0}, {'param1': 0.15, 'param2': 1}]
        old_weights = np.array([0.5, 0.5])
        
        priors = {
            'param1': stats.norm(0, 1),  # continuous parameter
            'param2': stats.randint(0, 3)  # discrete parameter
        }
        
        cov_matrix = np.array([[0.01]])  # Adjusted covariance matrix for 1 continuous parameter
        continuous_params = ['param1']
        discrete_params = ['param2']
        discrete_transition_prob = 0.5
        
        weights = compute_weights(new_particles, old_particles, old_weights, priors, cov_matrix, continuous_params, discrete_params, discrete_transition_prob)
        self.assertTrue(np.isclose(np.sum(weights), 1.0))



    def test_compute_effective_sample_size(self):
        weights = np.array([0.2, 0.5, 0.3])
        ess = compute_effective_sample_size(weights)
        self.assertGreater(ess, 0)

    def test_silverman_rule_of_thumb(self):
        neff = 50
        d = 2
        bandwidth = silverman_rule_of_thumb(neff, d)
        self.assertGreater(bandwidth, 0)

    def test_weighted_quantile(self):
        values = np.array([1, 2, 3, 4, 5])
        weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        quantile_value = weighted_quantile(values, weights, 0.5)
        self.assertEqual(quantile_value, 3)

if __name__ == '__main__':
    unittest.main()
