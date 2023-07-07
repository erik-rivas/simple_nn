import pandas as pd

diabetes = pd.read_csv("pima-indians-diabetes.csv")
diabetes.head()
cols_to_norm = [
    "Number_pregnant",
    "Glucose_concentration",
    "Blood_pressure",
    "Triceps",
    "Insulin",
    "BMI",
    "Pedigree",
]

diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(
    lambda x: (x - x.min()) / (x.max() - x.min())
)
