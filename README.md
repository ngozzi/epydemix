# Epydemix, the ABC of Epidemics
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Read the Docs](https://readthedocs.org/projects/epydemix/badge/?version=latest)](https://epydemix.readthedocs.io/en/latest/?badge=latest)


**Epydemix** is a Python package for epidemic modeling. It provides tools to create, calibrate, and analyze epidemic models, allowing users to simulate the spread of infectious diseases using different compartmental models, contact layers, and calibration techniques. It is designed to be used in conjunction with the [epydemix-data](https://github.com/ngozzi/epydemix-data/) package to load population and contact matrix data.


## Installation

### From TestPyPI (For Testing Purposes)

To install the latest version of `epydemix` from TestPyPI, use the following command:

```bash
pip install -i https://test.pypi.org/simple/ epydemix==0.1.0
```

---

### From PyPI (Coming Soon)

Once the package is released on the official PyPI repository, you can install it with:

```bash
pip install epydemix
```

---


## Quick Start

Once installed, you can start using **epydemix** in your Python scripts or Jupyter notebooks. Below an example to get started.

### Example: Creating and running a simple SIR model

```python
from epydemix import EpiModel 
from epydemix.visualization import plot_quantiles

# Defining a basic SIR model
model = EpiModel(
    name='SIR Model',
    compartments=['S', 'I', 'R'],  # Susceptible, Infected, Recovered
)
model.add_transition(source='S', target='I', params=(0.3, "I"), kind='mediated')
model.add_transition(source='I', target='R', params=0.1, kind='spontaneous')

# Running the model
results = model.run_simulations(
    start_date="2024-01-01",
    end_date="2024-04-10",
    Nsim=100)

# Plotting the results
df_quantiles_comps = results.get_quantiles_compartments()
plot_quantiles(df_quantiles_comps, columns=["I_total", "S_total", "R_total"])
```

### Tutorials
We provide a series of tutorials to help you get started with **epydemix**.

- [Tutorial 1](https://github.com/ngozzi/epydemix/blob/main/tutorials/1_Model_Definition_and_Simulation.ipynb): An Introduction to Model Definition and Simulation
- [Tutorial 2](https://github.com/ngozzi/epydemix/blob/main/tutorials/2_Modeling_with_Population_Data.ipynb): Using Population Data from Epydemix Data
- [Tutorial 3](https://github.com/ngozzi/epydemix/blob/main/tutorials/3_Modeling_Interventions.ipynb): Modeling Non-pharmaceutical Interventions
- [Tutorial 4](https://github.com/ngozzi/epydemix/blob/main/tutorials/4_Model_Calibration_part1.ipynb): Model Calibration with ABC (Part 1)
- [Tutorial 5](https://github.com/ngozzi/epydemix/blob/main/tutorials/5_Model_Calibration_part2.ipynb): Model Calibration with ABC (Part 2)
- [Tutorial 6](https://github.com/ngozzi/epydemix/blob/main/tutorials/6_Advanced_Modeling_Features.ipynb): Advanced Modeling Features
- [Tutorial 7](https://github.com/ngozzi/epydemix/blob/main/tutorials/7_Covid-19_Example.ipynb): COVID-19 Case Study


## Epydemix Data

**epydemix** also provides access to a wealth of real-world population and contact matrix data through the [**epydemix_data**](https://github.com/ngozzi/epydemix-data/) module. This dataset allows you to load predefined population structures, including age distribution and contact matrices for over 400 locations globally. You can use this data to create realistic simulations of disease spread in different geographies.

### Example of Loading Population Data

```python
from epydemix.population import load_epydemix_population

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



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer at `epydemix@isi.it`.
