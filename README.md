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

See [requirements.txt](https://github.com/ngozzi/epydemix/blob/main/requirements.txt)
- Python 3.6 or higher
- `numpy`
- `pandas`
- `matplotlib`
- `multiprocess`
- `scipy`
- `seaborn`
- `evalidate`

## Usage

Once installed, you can start using **Epydemix** in your Python scripts or Jupyter notebooks. Below an example to get started.

### Example: Creating and running a simple SIR model

```python
from epydemix import EpiModel 
from epydemix.visualization import plot_quantiles

# Create a new epidemic model
model = EpiModel(compartments=["S", "I", "R"])

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
