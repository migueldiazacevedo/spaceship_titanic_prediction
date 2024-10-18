from typing import List

import numpy as np
import pandas as pd


def start_pipeline(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to start data preprocessing pipeline. Make a copy of the initial DataFrame

    :param data_df: The DataFrame to process. From Kaggle Competition called "Spaceship Titanic"
                    Must have the following columns: ['PassengerId', 'HomePlanet', 'CryoSleep',
                   'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                   'Name', 'Transported']
                    find data at https://www.kaggle.com/competitions/spaceship-titanic
    :type data_df: pd.DataFrame
    :return: A copy of the initial DataFrame
    :rtype: pd.DataFrame
    """
    return data_df.copy()


def create_split_spaceship_features(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to split string features from the initial DataFrame into new features

    :param data_df: The DataFrame to process. From Kaggle Competition called "Spaceship Titanic"
                    Must have the following columns: ['PassengerId', 'HomePlanet', 'CryoSleep',
                   'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                   'Name', 'Transported']
                    find data at https://www.kaggle.com/competitions/spaceship-titanic
    :type data_df: pd.DataFrame
    :return: the same DataFrame after splitting the string features into new features
    :rtype: pd.DataFrame
    """
    data_df[["GroupID", "PassengerNum"]] = data_df["PassengerId"].str.split(
        "_", expand=True
    )
    data_df[["CabinDeck", "CabinNum", "CabinSide"]] = data_df["Cabin"].str.split(
        "/", expand=True
    )
    data_df[["FirstName", "LastName"]] = data_df["Name"].str.split(" ", expand=True)
    new_index = [
        "GroupID",
        "PassengerNum",
        "FirstName",
        "LastName",
        "Age",
        "HomePlanet",
        "Destination",
        "CabinDeck",
        "CabinNum",
        "CabinSide",
        "CryoSleep",
        "VIP",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "Transported",
    ]
    data_df = data_df.reindex(columns=new_index)
    return data_df


def spaceship_missing_qualitative_values(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace any missing values in columns containing qualitative variables

    :param data_df: The Spaceship Titanic DataFrame (after running create_split_spaceship_features)
    :type data_df: pd.DataFrame
    :return: The same DataFrame with missing values imputed and indicator columns added
    :rtype: pd.DataFrame
    """
    data_df["NameMissing"] = (
        data_df["FirstName"].isnull() & data_df["LastName"].isnull()
    )
    data_df.loc[
        data_df["FirstName"].isnull() & data_df["LastName"].isnull(),
        ["FirstName", "LastName"],
    ] = "Unknown"

    data_df["HomeMissing"] = data_df["HomePlanet"].isnull()
    data_df.loc[data_df["HomePlanet"].isnull(), ["HomePlanet"]] = "Unknown"

    data_df["DestinationMissing"] = data_df["Destination"].isnull()
    data_df.loc[data_df["Destination"].isnull(), ["Destination"]] = "Unknown"

    data_df["CabinMissing"] = data_df["CabinNum"].isnull()
    data_df.loc[
        data_df["CabinNum"].isnull(), ["CabinDeck", "CabinNum", "CabinSide"]
    ] = "Unknown"

    data_df["CryoMissing"] = data_df["CryoSleep"].isnull()
    data_df.loc[data_df["CryoSleep"].isnull(), ["CryoSleep"]] = "Unknown"

    data_df["VIPMissing"] = data_df["VIP"].isnull()
    data_df.loc[data_df["VIP"].isnull(), ["VIP"]] = "Unknown"

    return data_df


def change_spaceship_datatypes(
    data_df: pd.DataFrame,
    cat_features: List[str] = None,
    cat_to: str = None,
    bool_features: List[str] = None,
    bool_to: str = None,
    float_features: List[str] = None,
    float_to: str = None,
    int_features: List[str] = None,
    int_to: str = None,
) -> pd.DataFrame:
    """
    Change datatypes in the spaceship titanic dataframe (or any dataframe) as requested by the User
    :param data_df: any dataframe
    :param cat_features: List of categorical features
    :param cat_to: String indicating what to change feature to
    :param bool_features: List of boolean features
    :param bool_to: String indicating what to change feature to
    :param float_features: List of float features
    :param float_to: String indicating what to change feature to
    :param int_features: List of integer features
    :param int_to: String indicating what to change feature to
    :return: Dataframe with changed datatypes
    """
    if cat_features is not None:
        data_df[cat_features] = data_df[cat_features].astype(cat_to)
    if bool_features is not None:
        data_df[bool_features] = data_df[bool_features].astype(bool_to)
    if float_features is not None:
        data_df[float_features] = data_df[float_features].astype(float_to)
    if int_features is not None:
        data_df[int_features] = data_df[int_features].astype(int_to)

    return data_df


def create_party_size_feature(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to create party size feature
    :param data_df: DataFrame containing column 'GroupID'
    :return: DataFrame containing columns 'PartySize'
    """
    data_df["PartySize"] = (
        data_df.loc[:, ["GroupID"]].groupby("GroupID", observed=False).transform("size")
    )
    return data_df


def create_family_group_feature(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to create Boolean feature based on whether a group is a family (i.e. contains the same last name) or not
    :param data_df: DataFrame containing columns 'LastName' and 'GroupID'
    :return: DataFrame containing boolean column 'FamilyGroupMember'
    """
    data_df["FamilyGroupMember"] = np.where(
        data_df.groupby(["LastName", "GroupID"], observed=False).transform("size") > 1,
        True,
        False,
    )
    return data_df


def create_cabin_bin_feature(data_df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """
    Create an ordinal (float) variable that creates a specific number of bins representing how high the room number is
    :param data_df: DataFrame containing columns ['CabinNum', 'Transported']
    :param n_bins: the number of bins to create
    :return: the same DataFrame including the column 'CabinBin'
    """
    cabin_num_temp = data_df.loc[:, ["CabinNum", "Transported"]]
    cabin_num_temp["CabinNum"] = pd.to_numeric(cabin_num_temp.CabinNum, errors="coerce")
    cabin_num_temp_sorted = cabin_num_temp.sort_values("CabinNum", ascending=True)
    cabin_num_temp_sorted["bin"] = pd.qcut(
        cabin_num_temp_sorted["CabinNum"], n_bins, labels=False
    )
    data_df["CabinBin"] = cabin_num_temp_sorted.sort_index()["bin"]
    return data_df


def create_total_spending_feature(
    data_df, col_names=["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
) -> pd.DataFrame:
    """
    Sum all the spending columns
    :param data_df: a DataFrame
    :param col_names: list of column names to sum
    :return: a DataFrame containing the sum of those columns in ['TotalSpending'] column
    """
    data_df["TotalSpending"] = data_df[col_names].sum(axis=1, skipna=False)
    return data_df


def create_spending_indicator_columns(
    data_df,
    col_names=[
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "TotalSpending",
    ],
):
    """
    Create an indicator column that indicates whether a passenger spent on a luxury amenity or not
    :param data_df: dataframe containing data
    :param col_names: list of column names to create indicator columns from
    :return: a dataframe containing indicator columns
    """
    for col in col_names:
        data_df[f"Yes{col}"] = data_df[col].apply(lambda x: True if x > 0 else False)
    return data_df


def log_transform_spending(
    data_df,
    col_names=[
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "TotalSpending",
    ],
):
    """
    Log transform the luxury amenity spending column
    :param data_df: Pandas DataFrame containing data
    :param col_names: names of columns containing spending on luxury items
    :return: dataframe containing columns of log transformations of spending (with 1 added to each prior to log)
    """
    for col in col_names:
        data_df[f"Log{col}"] = data_df[col].transform(np.log1p)
    return data_df
