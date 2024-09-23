import pandas as pd
import seaborn
import sklearn
from sklearn.preprocessing import LabelEncoder
import plotly.express
import matplotlib.pyplot as plt
from typing import Literal

#


def view_dataset(dataset: pd.DataFrame) -> None:
    """
    See the basic information about the dataset
    """
    print(".head")
    dataset.head()
    print(".describe")
    dataset.describe()
    print(".info")
    dataset.info()

def remove_outliers(dataset: pd.DataFrame, ignored_columns: list, standard_deviations: int) -> pd.DataFrame:
    """
    Remove outliers from a given dataset. 
    Ignores sepcified columns
    Removes outside of the given standard deviations from the mean
    """
    dataset_without_outliers = dataset.copy()

    for column in dataset_without_outliers:
        if column not in ignored_columns:
            mean = dataset_without_outliers[column].mean()
            standard_deviation = dataset_without_outliers[column].std()
            dataset_without_outliers = dataset_without_outliers[(dataset_without_outliers[column] <= mean+(2*standard_deviation))]
    
    return dataset_without_outliers

def normalise_dataset(dataset: pd.DataFrame, ignored_columns: list) -> pd.DataFrame:
    """
    Normalize the data to values between 0 and 1 to allow for faster computation if dealing with large numbers.
    Uses the Z-score method
    """
    dataset_normalised = dataset.copy()

    for column in dataset_normalised.columns:
        if column not in ignored_columns: 
            dataset_normalised[column] = (dataset_normalised[column] - dataset_normalised[column].mean()) / dataset_normalised[column].std()
    
    return dataset_normalised

_VISUALISATION_TYPES = Literal["histogram", "correlation heatmap", "scatter diagram", "PI"]
def visualise_data(dataset: pd.DataFrame, visualisation_type: _VISUALISATION_TYPES = "histogram"):
    """
    Visualise the dataset using different diagrams
    """
    match visualisation_type:
        case "histogram":
            dataset.hist(bins=50,  figsize=(20,15), grid=False)
            plt.show()
        case "correlation heatmap":
            corr = pd.DataFrame.corr(dataset.loc[:, dataset.columns!="class"])
            corr.style.background_gradient(cmap="coolwarm")
        case "scatter diagram":
            seaborn.pairplot(dataset)
        case "PI":
            plotly.express.pie(dataset, names = "class")
        case _:
            raise ValueError("Invalid visualisation type. Expected one of: histogram, correlation heatmap, scatter diagram")
    
def transform_target_to_int(dataset: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    transforms a target column from a string to integer
    """
    le = LabelEncoder()
    transformed_dataset = dataset.copy()
    transformed_dataset[column] = le.fit_transform(transformed_dataset[column])
    transformed_dataset[column] = transformed_dataset[column].astype(int)

    return transformed_dataset


def main():
    #importing the data
    dataset = pd.read_csv("star_classification.csv")
    view_dataset(dataset)

    #cleaning the data
    dataset_without_outliers = remove_outliers(dataset, ["alpha","delta","MJD","class"], 2)
    dataset_normalised = normalise_dataset(dataset_without_outliers, ["class"])

    #visualising the data
    visualise_data(dataset_normalised, "histogram")
    visualise_data(dataset_normalised, "correlation heatmap")
    visualise_data(dataset_normalised, "scatter diagram")
    visualise_data(dataset_normalised, "PI")

    #preparin the data for machine learning
    pre_processed_dataset = transform_target_to_int(dataset_normalised, "class")
    pre_processed_dataset.to_csv("star_classification_processed.csv", index = False)



if __name__ == "__main__":
    main()