import numpy as np 
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union, Any


def initialize_particles(
        priors: Dict[str, Any],
        num_particles: int
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, float]:
    """
    Initializes particles from the given priors and computes their initial weights.

    Args:
        priors (Dict[str, rv_frozen]): A dictionary where keys are parameter names and values are
            `scipy.stats.rv_frozen` objects representing the priors for each parameter.
        num_particles (int): The number of particles to initialize.

    Returns:
        Tuple[List[Dict[str, float]], np.ndarray, float]:
            - A list of dictionaries where each dictionary represents a particle with sampled values
              for each parameter.
            - An array of weights for each particle, initialized to equal values.
            - A float representing the initial tolerance, set to infinity.
    """
    # Initialize particles from the prior
    particles = [{p: priors[p].rvs() for p in priors} for _ in range(num_particles)]
    weights = np.ones(num_particles) / num_particles
    tolerance = np.inf
    return particles, weights, tolerance


def default_perturbation_kernel(
        particle: Dict[str, Any],
        continuous_params: List[str],
        discrete_params: List[str],
        cov_matrix: np.ndarray,
        priors: Dict[str, Any],
        p_discrete_transition: float
    ) -> Dict[str, Any]:
    """
    Perturbs a particle's parameters.

    Args:
        particle (Dict[str, float]): Dictionary where keys are parameter names and values are the current values of the parameters.
        particles (List[Dict[str, float]]): List of dictionaries where each dictionary contains the parameters of a particle.
        continuous_params (List[str]): List of parameter names that are continuous.
        discrete_params (List[str]): List of parameter names that are discrete.
        cov_matrix (np.ndarray): Covariance matrix for the continuous parameters.
        priors (Dict[str, rv_frozen]): Dictionary where keys are parameter names and values are `scipy.stats.rv_frozen` objects representing the prior distributions.
        p_discrete_transition (float): Probability of transitioning to a new state for discrete parameters.

    Returns:
        Dict[str, float]: The perturbed particle, with updated parameter values.
    """
    perturbed_particle = particle.copy()

    # Perturb continuous parameters
    if continuous_params:
        # Extract current values for continuous parameters
        current_values = np.array([particle[param] for param in continuous_params])

        while True:
            # Apply multivariate normal transition
            perturbed_values = current_values + np.random.multivariate_normal(mean=np.zeros_like(current_values), cov=cov_matrix)

            # Check if all perturbed values are within the prior support
            within_support = True
            for i, param in enumerate(continuous_params):
                if not (priors[param].support()[0] <= perturbed_values[i] <= priors[param].support()[1]):
                    within_support = False
                    break

            if within_support:
                break

        # Update the perturbed particle with new continuous values
        for i, param in enumerate(continuous_params):
            perturbed_particle[param] = perturbed_values[i]

    # Perturb discrete parameters
    for param in discrete_params:

        current_value = particle[param]
        if np.random.rand() < p_discrete_transition:

            # sample an alternative value for this parameter 
            new_value = priors[param].rvs()
            while new_value == current_value:
                new_value = priors[param].rvs()
            perturbed_particle[param] = new_value

    return perturbed_particle


def compute_covariance_matrix(
        particles: List[Dict[str, Any]],
        continuous_params: List[str],
        weights: Optional[np.ndarray] = None,
        scaling_factor: float = 1.0,
        apply_bandwidth: bool = True
    ) -> np.ndarray:
    """
    Computes the covariance matrix for the continuous parameters in a list of particles.

    Args:
        particles (List[Dict[str, float]]): A list of dictionaries where each dictionary contains parameter values for a particle.
        continuous_params (List[str]): A list of parameter names that are continuous.
        weights (Optional[np.ndarray]): An optional array of weights for the particles. Defaults to None, which implies equal weights.
        scaling_factor (float): A factor by which to scale the covariance matrix. Defaults to 1.0.
        apply_bandwidth (bool): Whether to apply a bandwidth adjustment to the covariance matrix. Defaults to True.

    Returns:
        np.ndarray: The covariance matrix of the continuous parameters.
    """
    # Extract the values of the continuous parameters from each particle
    data = np.array([[particle[param] for param in continuous_params] for particle in particles])

    # Compute the covariance matrix
    if len(continuous_params) == 1:
        covariance_matrix = np.array([[np.var(data)]])
    else:
        covariance_matrix = np.cov(data, aweights=weights, rowvar=False)

    # Compute bandwidth factor
    if apply_bandwidth:
        if weights is None:
            weights = np.ones(len(particles)) / len(particles)
        bandwidth_factor = silverman_rule_of_thumb(compute_effective_sample_size(weights), len(continuous_params))
        covariance_matrix *= bandwidth_factor**2

    # Apply scaling factor
    covariance_matrix *= scaling_factor

    return covariance_matrix


def compute_weights(
        new_particles: List[Dict[str, Any]],
        old_particles: List[Dict[str, Any]],
        old_weights: List[float],
        priors: Dict[str, Any],
        cov_matrix: np.ndarray,
        continuous_params: List[str],
        discrete_params: List[str],
        discrete_transition_prob: float
    ) -> np.ndarray:
    """
    Computes the weights for new particles in generation t.

    Args:
        new_particles (List[Dict[str, float]]): A list of dictionaries where each dictionary contains the parameters of new particles.
        old_particles (List[Dict[str, float]]): A list of dictionaries where each dictionary contains the parameters of old particles from the previous generation.
        old_weights (List[float]): A list of weights for the old particles.
        priors (Dict[str, rv_frozen]): A dictionary where keys are parameter names and values are `scipy.stats.rv_frozen` objects representing the prior distributions.
        cov_matrix (np.ndarray): Covariance matrix used in the perturbation kernel for continuous parameters.
        continuous_params (List[str]): List of parameter names that are continuous.
        discrete_params (List[str]): List of parameter names that are discrete.
        discrete_transition_prob (float): Probability of transitioning to a new state for discrete parameters.

    Returns:
        np.ndarray: A NumPy array of computed weights for the new particles, normalized to sum to 1.
    """

    new_weights = np.zeros(len(new_particles))
    
    for i, new_particle in enumerate(new_particles):
        # Compute the prior probability of the new particle
        prior_prob = np.prod([priors[param].pdf(new_particle[param]) for param in continuous_params])
        prior_prob *= np.prod([priors[param].pmf(new_particle[param]) for param in discrete_params])

        # Compute the denominator of the weight expression
        # Continuous parameters
        diff_vectors = np.array([[new_particle[param] - old_particle[param] for param in continuous_params] for old_particle in old_particles])
        continuous_kernel_probs = stats.multivariate_normal.logpdf(diff_vectors, mean=np.zeros(len(continuous_params)), cov=cov_matrix, allow_singular=True)

        # Discrete parameters
        discrete_kernel_probs = np.ones(len(old_particles))
        for param in discrete_params:
            
            # Calculate the number of elements in the domain
            lower_bound, upper_bound = priors[param].support()
            n_values = upper_bound - lower_bound + 1

            # Compute discrete kernel probabilities
            transitions = np.array([new_particle[param] != old_particle[param] for old_particle in old_particles])
            discrete_kernel_probs *= np.where(transitions, discrete_transition_prob / (n_values - 1), (1 - discrete_transition_prob))

        kernel_probs = np.exp(continuous_kernel_probs) * discrete_kernel_probs
        denominator = np.sum(old_weights * kernel_probs)
        
        # Calculate the weight for the new particle
        new_weights[i] = prior_prob / denominator

    # Normalize the weights so they sum to 1
    new_weights /= np.sum(new_weights)
    
    return new_weights


def compute_effective_sample_size(weights: np.ndarray) -> float:
    """
    Computes the effective sample size (ESS) of a set of weights.

    Args:
        weights (np.ndarray): A NumPy array of weights for the particles.

    Returns:
        float: The effective sample size, calculated as the reciprocal of the sum of squared weights.
    """
    ess = 1 / np.sum(weights**2)
    return ess


def silverman_rule_of_thumb(neff: int, d: int) -> float:
    """
    Computes the bandwidth factor using Silverman's rule of thumb.

    Args:
        neff (int): The effective sample size.
        d (int): The number of dimensions (parameters).

    Returns:
        float: The bandwidth factor calculated using Silverman's rule of thumb.
    """
    return (4 / neff / (d + 2)) ** (1 / (d + 4))


def weighted_quantile(
        values: Union[np.ndarray, list],
        weights: Union[np.ndarray, list],
        quantile: float
    ) -> float:
    """
    Compute the weighted quantile of a dataset.

    Args:
        values (Union[np.ndarray, list]): Data values for which the quantile is to be computed.
        weights (Union[np.ndarray, list]): Weights associated with the data values.
        quantile (float): The quantile to compute, must be between 0 and 1.

    Returns:
        float: The weighted quantile value.

    Raises:
        ValueError: If `quantile` is not between 0 and 1.
    """
    # Ensure quantile is between 0 and 1
    if not (0 <= quantile <= 1):
        raise ValueError("Quantile must be between 0 and 1.")
    
    values = np.asarray(values)
    weights = np.asarray(weights)
    
    # Ensure that weights sum to 1
    weights /= np.sum(weights)

    # Sort values and weights by values
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute the cumulative sum of weights
    cumulative_weights = np.cumsum(sorted_weights)

    # Find the index where the cumulative weight equals or exceeds the quantile
    quantile_index = np.searchsorted(cumulative_weights, quantile)

    return sorted_values[quantile_index]