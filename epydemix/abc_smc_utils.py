import numpy as np 
from scipy import stats


def initialize_particles(priors, num_particles):
    # Initialize particles from the prior
    particles = [{p: priors[p].rvs() for p in priors} for _ in range(num_particles)]
    weights = np.ones(num_particles) / num_particles
    tolerance = np.inf
    return particles, weights, tolerance


def perturbation_kernel(particle, continuous_params, discrete_params, cov_matrix, priors, p_discrete_transition):
    """
    Perturbs a particle's parameters.

    Parameters:
    - particle: dict, keys are parameter names, values are current values of the parameters
    - continuous_params: list of parameter names that are continuous
    - discrete_params: list of parameter names that are discrete
    - cov_matrix: dict, covariance matrices for the continuous parameters
    - priors: dict, prior distributions for the parameters
    - discrete_jump_probs: dict, probability distributions for the discrete jumps

    Returns:
    - perturbed_particle: dict, the perturbed particle
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
        # The probability distribution must exclude the current value if we are jumping to a different state
        if np.random.rand() < p_discrete_transition:
            # Get the probability distribution for the current parameter
            perturbed_particle[param] = np.random.choice(range(priors[param].support()[0], priors[param].support()[1]))

    return perturbed_particle


def compute_covariance_matrix(particles, continuous_params, weights=None, scaling_factor=1., apply_bandwidth=True):
    """
    Computes the covariance matrix for the continuous parameters in a list of particles.

    Parameters:
    - particles: list of dicts, each dict contains a set of parameters
    - continuous_params: list of str, the names of the continuous parameters

    Returns:
    - covariance_matrix: numpy array, the covariance matrix of the continuous parameters
    """
    # Extract the values of the continuous parameters from each particle
    data = np.array([[particle[param] for param in continuous_params] for particle in particles])

    # Compute the covariance matrix
    if len(continuous_params) == 1:
        covariance_matrix = np.array([[np.var(data)]])
    else:
        covariance_matrix = np.cov(data, rowvar=False)

    # Compute bandwidth factor
    if apply_bandwidth:
        if weights is None:
            weights = np.ones(len(particles)) / len(particles)
        bandwidth_factor = silverman_rule_of_thumb(compute_effective_sample_size(weights), len(continuous_params))
        covariance_matrix *= bandwidth_factor**2

    # Apply scaling factor
    covariance_matrix *= scaling_factor

    return covariance_matrix


def compute_weights(new_particles, old_particles, old_weights, priors, cov_matrix, continuous_params, discrete_params, discrete_transition_prob):
    """
    Computes the weights for new particles in generation t.
    
    Parameters:
    - new_particles: list of dicts, the particles in the current generation t
    - old_particles: list of dicts, the particles from the previous generation t-1
    - old_weights: list of floats, the weights of particles from generation t-1
    - priors: dict, keys are parameter names and values are scipy.stats objects representing the priors
    - cov_matrix: covariance matrix used in the perturbation kernel for continuous parameters
    - continuous_params: list of parameter names that are continuous
    - discrete_params: list of parameter names that are discrete
    - discrete_transition_prob: float, probability of any transition for discrete parameters
    
    Returns:
    - new_weights: list of floats, the computed weights for the new particles
    """
    # Precompute inverse and determinant of covariance matrix for the multivariate normal
    #inv_cov_matrix = np.linalg.inv(cov_matrix)
    #log_det_cov_matrix = np.log(np.linalg.det(cov_matrix))

    new_weights = np.zeros(len(new_particles))
    
    for i, new_particle in enumerate(new_particles):
        # Compute the prior probability of the new particle
        prior_prob = np.prod([priors[param].pdf(new_particle[param]) for param in continuous_params])
        prior_prob *= np.prod([priors[param].pmf(new_particle[param]) for param in discrete_params])

        # Compute the denominator of the weight expression
        diff_vectors = np.array([[new_particle[param] - old_particle[param] for param in continuous_params] for old_particle in old_particles])
        continuous_kernel_probs = stats.multivariate_normal.logpdf(diff_vectors, mean=np.zeros(len(continuous_params)), cov=cov_matrix, allow_singular=True)

        discrete_kernel_probs = np.ones(len(old_particles))
        for param in discrete_params:
            transitions = np.array([new_particle[param] != old_particle[param] for old_particle in old_particles])
            discrete_kernel_probs *= np.where(transitions, discrete_transition_prob, 1 - discrete_transition_prob)

        kernel_probs = np.exp(continuous_kernel_probs) * discrete_kernel_probs
        denominator = np.sum(old_weights * kernel_probs)
        
        # Calculate the weight for the new particle
        new_weights[i] = prior_prob / denominator

    # Normalize the weights so they sum to 1
    new_weights /= np.sum(new_weights)
    
    return new_weights


def compute_effective_sample_size(weights): 
    """
    Computes the effective sample size of a set of weights.

    Parameters:
    - weights: numpy array, the weights of the particles

    Returns:
    - ess: float, the effective sample size
    """
    ess = 1 / np.sum(weights**2)
    return ess


def silverman_rule_of_thumb(neff, d):
    """
    Silverman's rule of thumb.
    """
    return (4 / neff / (d + 2)) ** (1 / (d + 4))
