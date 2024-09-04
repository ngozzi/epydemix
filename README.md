# Epydemix, the ABC of epidemics
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


**Epydemix** is a Python package for epidemic modeling and simulation. It provides tools to create, run, and analyze epidemiological models, allowing users to simulate the spread of infectious diseases using different compartmental models, contact layers, and calibration techniques.

## Features

- Build custom epidemic models using different compartments (e.g., SIR, SEIR).
- Load or create populations with contact layers (e.g., school, work, home, community).
- Run simulations and analyze the results using metrics and visualization tools.
- Calibrate models to real-world data.
- Easily extendable and customizable for different scenarios.

## Installation

You can install **Epydemix** locally for development using `pip`:

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/epydemix.git
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

### Requirements

- Python 3.6 or higher
- `numpy`
- `pandas`
- `matplotlib`

## Usage

Once installed, you can start using **Epydemix** in your Python scripts or Jupyter notebooks. Below are some examples to get started.

### Example 1: Creating a Model with a Default Population

```python
from epydemix.model.epimodel import EpiModel

# Create a new epidemic model with the default population
model = EpiModel(use_default_population=True)

# Run simulations and analyze results
model.run_simulation()
```

### Example 2: Setting a Custom Population

```python
from epydemix.model.epimodel import EpiModel

# Create a new model
model = EpiModel()

# Set a custom population using data from a file
model.set_custom_population(
    population_name="my_custom_population",
    population_data_path="data/population_data.csv",
    contact_layers=["school", "work", "community"]
)

# Run the simulation
model.run_simulation()
```

### Example 3: Plotting Simulation Results

```python
from epydemix.visualization.plotting import plot_results

# Assuming 'results' is the output from a simulation
plot_results(results)
```

## Documentation

For full documentation, including detailed guides on each module, visit the [Documentation](https://github.com/your_username/epydemix/wiki).

### Main Modules

- **`epydemix.model`**: Core classes for building and running epidemic models.
- **`epydemix.calibration`**: Tools for calibrating models to real-world data.
- **`epydemix.visualization`**: Visualization functions for plotting simulation results.
- **`epydemix.utils`**: Utility functions and additional tools.

## Running Tests

You can run the package tests using `pytest`:

1. Navigate to the root of the project:

   ```bash
   cd epydemix
   ```

2. Run the tests:

   ```bash
   pytest tests/
   ```

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
