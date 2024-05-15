
# Packaging the ML Model of Breast Cancer Classification

#### Problem Statement
- I want to develop a machine learning model to classify whether a given tumor is benign or malignant based on features extracted from breast cancer cell images.
- It is a classification problem where we have to predict whether a tumor is benign or malignant.

#### Data
The data corresponds to a set of financial requests associated with individuals. 

| Variables                 | Description                                                        |
|---------------------------|--------------------------------------------------------------------|
| ID                        | Unique ID number                                                   |
| Diagnosis                 | Diagnosis of the cell (M = malignant, B = benign)                  |
| Radius_mean               | Mean radius of distances from center to points on the perimeter    |
| Texture_mean              | Standard deviation of gray-scale values                            |
| Perimeter_mean            | Mean perimeter                                                     |
| Area_mean                 | Mean area                                                          |
| Smoothness_mean           | Mean local variation in radius lengths                             |
| Compactness_mean          | Mean compactness (perimeter^2 / area - 1.0)                        |
| Concavity_mean            | Mean severity of concave portions of the contour                   |
| Concave_points_mean       | Mean number of concave portions of the contour                     |
| Symmetry_mean             | Mean symmetry                                                      |
| Fractal_dim_mean          | Mean fractal dimension ("coastline approximation" - 1)             |
| Radius_se                 | Standard error for the radius                                      |
| Texture_se                | Standard error for the texture                                     |
| Perimeter_se              | Standard error for the perimeter                                   |
| Area_se                   | Standard error for the area                                        |
| Smoothness_se             | Standard error for the smoothness                                  |
| Compactness_se            | Standard error for the compactness                                 |
| Concavity_se              | Standard error for the concavity                                   |
| Concave_points_se         | Standard error for the concave points                              |
| Symmetry_se               | Standard error for the symmetry                                    |
| Fractal_dim_se            | Standard error for the fractal dimension                           |
| Radius_worst              | "Worst" or largest mean value for radius                           |
| Texture_worst             | "Worst" or largest mean value for texture                          |
| Perimeter_worst           | "Worst" or largest mean value for perimeter                        |
| Area_worst                | "Worst" or largest mean value for area                             |
| Smoothness_worst          | "Worst" or largest mean value for smoothness                       |
| Compactness_worst         | "Worst" or largest mean value for compactness                      |
| Concavity_worst           | "Worst" or largest mean value for concavity                        |
| Concave_points_worst      | "Worst" or largest mean value for concave points                   |
| Symmetry_worst            | "Worst" or largest mean value for symmetry                         |
| Fractal_dimension_worst   | "Worst" or largest mean value for fractal dimension                |

Source: Kaggle

## Running Locally

### Setting the PYTHONPATH Environment Variable

### For Windows Command Prompt

To temporarily set the `PYTHONPATH` for your current session, run the following command:

```cmd
set PYTHONPATH=%PYTHONPATH%;C:\Users\User\Desktop\MLOps\BreastCancerModel\Model\Package
C:\Users\User\Desktop\MLOps\BreastCancerModel\prediction_model


## Virtual Environment
Install virtualenv

```python
python3 -m pip install virtualenv
```

Check version
```python
virtualenv --version
```

Create virtual environment

```python
virtualenv breast_cancer
```

Activate virtual environment

For Linux/Mac
```python
source breast_cancer/bin/activate
```
For Windows
```python
breast_cancer\Scripts\activate
```

Deactivate virtual environment

```python
deactivate
```


## Directory structure

```
BreastCancerModel


├── MANIFEST.in
├── prediction_model
│   ├── config
│   │   ├── config.py
│   │   └── __init__.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── test.csv
│   │   └── train.csv
│   ├── __init__.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── processing
│   │   ├── data_handling.py
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── trained_models
│   │   ├── diagnosis.pkl
│   │   └── __init__.py
│   ├── training_pipeline.py
│   └── VERSION
├── README.md
├── requirements.txt
├── setup.py
└── tests
    ├── pytest.ini
    └── test_prediction.py
```


# Build the Package

1. Go to Project directory and install dependencies
`pip install -r requirements.txt`

2. Create Pickle file after training:
`python prediction_model/training_pipeline.py`

3. Create source distribution and wheel
`python setup.py sdist bdist_wheel`

# Installation of Package

Go to project directory where `setup.py` file is located

1. To install it in editable or developer mode
```python
pip install -e .
```
```.``` refers to current directory

```-e``` refers to --editable mode

2. Normal installation
```python
pip install .
```
```.``` refers to current directory

3. Also can be installed from git as well after pushing to github

```
pip install git+https://github.com/PonzhiAghan/Breast_Cancer_MLOPS.git
```

# Testing the Package Working

1. Remove the PYTHONPATH from environment variables 
2. Go to a separate location which is outside of package directory
3. Create a new virual environment using the commands mentioned above & activate it
4. Before installing, test that you are able to import the package of `prediction_model`
5. Now in the new environment install the package from github
`pip install git+https://github.com/PonzhiAghan/Breast_Cancer_MLOPS.git`
6. Now try importing the prediction_model, you should be able to do it successfully
7. Extras : Run training pipeline using the package, and also conduct the test
