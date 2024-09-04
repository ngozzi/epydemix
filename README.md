# Epydemix, the ABC of epidemics
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


**Epydemix** is a Python package for epidemic modeling and simulation. It provides tools to create, run, and analyze epidemiological models, allowing users to simulate the spread of infectious diseases using different compartmental models, contact layers, and calibration techniques.

## Features

- **Design and simulate epidemic models** with customizable compartmental frameworks, including widely-used models like SIR, SEIR, and more complex structures tailored to specific scenarios.
- **Load or create detailed real-world population datasets** for over 400 locations worldwide, incorporating demographic and contact patterns.
- **Run simulations** and explore model outcomes and trends over time with powerful visualizations.
- **Effortlessly calibrate your models** to real-world epidemiological data, ensuring accuracy and relevance by fine-tuning key parameters using Approximate Bayesian Computation frameworks.


## Installation

You can install **Epydemix** locally for development using `pip`:

1. Clone the repository:

   ```bash
   git clone https://github.com/ngozzi/epydemix.git
   ```

2. Navigate to the project directory:

   ```bash
   cd epydemix
   ```

3. Install the package in editable mode:

   ```bash
   pip install -e .
   ```

This will install the package and its dependencies, allowing you to make changes and immediately see the effects without needing to reinstall.
See [requirements.txt](https://github.com/ngozzi/epydemix/blob/main/requirements.txt)


## Usage

Once installed, you can start using **Epydemix** in your Python scripts or Jupyter notebooks. Below an example to get started.

### Example: Creating and running a simple SIR model

```python
from epydemix import EpiModel 
from epydemix.visualization import plot_quantiles

# Create a new epidemic model
model = EpiModel(def)

# Define parameters
model.add_parameter(name="beta", value=0.3)
model.add_parameter(name="mu", value=0.1)

# Define transitions 
model.add_transition(source="S", target="I", rate="beta", agent="I")
model.add_transition(source="I", target="R", rate="mu")

# Run simulations specifying time frame and initial conditions
results = model.run_simulations(start_date="2019-12-01", end_date="2020-04-01", S=99990, I=10)

# Plot results 
plot_quantiles(results, columns=["I_total", "S_total", "R_total"])
```

## Epydemix Data

**Epydemix** also provides access to a wealth of real-world population and contact matrix data through the **epydemix_data** module. This dataset allows you to load predefined population structures, including age distribution and contact matrices for over 400 locations globally. You can use this data to create realistic simulations of disease spread in different geographies.

### Example of Loading Population Data

```python
from epydemix.population import load_population

# Load population data for the United States with the Mistry 2021 contact matrix
population = load_population(
    population_name="United_States",
    path_to_data="path/to/epydemix_data/United_States",
    contacts_source="mistry_2021",
    layers=["home", "work", "school", "community"]
)

# Use the loaded population in your epidemic model
model.set_custom_population(population=population)
```

Epydemix can load data either locally from a folder or directly from online sources, making it easy to simulate a wide range of epidemic models on real population data.

For more information about the available population data and contact matrices, please refer to the [locations.csv](https://github.com/ngozzi/epydemix/blob/main/epydemix_data/locations.csv) file, and see the [epydemix_data README](https://github.com/ngozzi/epydemix/blob/main/epydemix_data/README.md) for further details.


### Main Modules

- **`epydemix.model`**: Core classes for building and running epidemic models.
- **`epydemix.calibration`**: Tools for calibrating models to real-world data.
- **`epydemix.visualization`**: Visualization functions for plotting simulation results.
- **`epydemix.utils`**: Utility functions and additional tools.


## Contributing

Contributions are welcome! If you would like to contribute, please:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/my-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/my-feature`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer at `nic.gozzi@gmail.com`.
