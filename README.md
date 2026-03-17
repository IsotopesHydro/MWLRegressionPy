# MWLRegressionPY

Python package to perform linear regression of data with measurement uncertainties in both variables.
The package was developed in the framework of regression of water stable isotopes in precipitation, but can be used for any X and Y dataset with uncertainties in both variables.

## Installation

Install directly from GitHub using `pip`:

```bash
pip install git+[https://github.com/IsotopesHydro/MWLRegressionPython/MWLRegression.git](https://github.com/IsotopesHydro/MWLRegressionPython/MWLRegression.git)
```

## Overview

A quick overview of some of the functions:

    york_regression: Performs York linear regression which accounts for uncertainties in both X and Y variables, with uncertainties also potentially correlated. The function depends on matplotlib for plotting.

    RegEqualSigma: Performs linear regression on x and y data which has equal measurement uncertainties. The function depends on matplotlib for plotting.

## Usage

Here is an example of how to import the package and run a regression using data from a CSV file.

Expected CSV format:
```bash
Xi   Yi   errXi  errYi 
x1   y1   errx1  erry1 
...  ...  ...    ...   
xn   yn   errxn  erryn 
```

(Note: Headers are not mandatory in the data table, but if your file does not have headers, remember to adjust the pd.read_csv function accordingly).

Python Code:

```bash
import pandas as pd
from MWLRegression import YorkRegression, RegEqualSigma

# Define the path to your data
path_csv = "./your_file.csv"

# Load the data
df = pd.read_csv(path_csv)

# Assign variables
Xi = df['Xi']
Yi = df['Yi']
errXi = df['errXi']
errYi = df['errYi']

# Run the regression
results = YorkRegression(Xi, Yi, errXi, errYi, iter=1000, plot=True)
# or
results = RegEqualSigma(Xi, Yi, errXi, errYi, iter=1000, plot=True)

# The function returns a dictionary. You can print specific results like this:
print("Slope:", results["Slope"])
print("Intercept:", results["Intercept"])
```