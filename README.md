# Epydemix, the ABC of Epidemics
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


**Epydemix** is a Python package for epidemic modeling and simulation. It provides tools to create, run, and analyze epidemiological models, allowing users to simulate the spread of infectious diseases using different compartmental models, contact layers, and calibration techniques.

## Features

- **Design and simulate epidemic models** with customizable compartmental frameworks, including widely-used models like SIR, SEIR, and more complex structures tailored to specific scenarios.
- **Load or create detailed real-world population datasets** for over 400 locations worldwide, incorporating demographic and contact patterns.
- **Run simulations** and explore model outcomes and trends over time with powerful visualizations.
- **Effortlessly calibrate your models** to real-world epidemiological data, ensuring accuracy and relevance by fine-tuning key parameters using Approximate Bayesian Computation frameworks.


## Installation

### From TestPyPI (For Testing Purposes)

To install the latest version of `epydemix` from TestPyPI, use the following command:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ epydemix
```

---

### From PyPI (Coming Soon)

Once the package is released on the official PyPI repository, you can install it with:

```bash
pip install epydemix
```

---

### Optional: Installing for Development

If you plan to contribute or modify the package, you can install it in editable mode:

```bash
git clone https://github.com/ngozzi/epydemix.git
cd epydemix
pip install -e .
```


## Quick Start

Once installed, you can start using **epydemix** in your Python scripts or Jupyter notebooks. Below an example to get started.

### Example: Creating and running a simple SIR model

```python
from epydemix import EpiModel 
from epydemix.visualization import plot_quantiles

# Create a new epidemic model
model = EpiModel(predefined_model="SIR")

# Run 100 stochastic simulations specifying time frame 
results = model.run_simulations(start_date="2019-12-01", 
                                end_date="2020-04-01",
                                Nsim=100)

# Plot results 
plot_quantiles(results, columns=["I_total", "S_total", "R_total"])
```

## Epydemix Data

**epydemix** also provides access to a wealth of real-world population and contact matrix data through the [**epydemix_data**](https://github.com/ngozzi/epydemix-data/) module. This dataset allows you to load predefined population structures, including age distribution and contact matrices for over 400 locations globally. You can use this data to create realistic simulations of disease spread in different geographies.

### Example of Loading Population Data

```python
from epydemix.model import load_epydemix_population

# Load population data for the United States with the Mistry 2021 contact matrix
population = load_epydemix_population(
    population_name="United_States",
    contacts_source="mistry_2021",
    layers=["home", "work", "school", "community"]
)

# Use the loaded population in your epidemic model
model.set_population(population=population)
```

Epydemix can load data either locally from a folder or directly from online sources, making it easy to simulate a wide range of epidemic models on real population data.

For more information about the available population and contact matrices and to download the data, please visit the [dedicated repository](https://github.com/ngozzi/epydemix-data/).


### Main Modules

- **`epydemix.model`**: Core classes for building and running epidemic models.
- **`epydemix.calibration`**: Tools for calibrating models to real-world data.
- **`epydemix.visualization`**: Visualization functions for plotting simulation results.
- **`epydemix.utils`**: Utility functions and additional tools.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer at `nic.gozzi@gmail.com`.
