import pandas as pd
import numpy as np
from game_data import characters, all_stages
import sys


def parse_spreadsheet(filepath: str) -> pd.DataFrame:
    # read the csv
    df = pd.read_csv("rivals_spreadsheet.tsv", sep="\t")

    # cutting out goals and notes, these are priveleged information!
    if "Notes" in df.columns:
        df = df.drop(columns=["Notes", "Goal"])
        df.to_csv("rivals_spreadsheet.tsv", sep="\t", header=True)

    # Killing incomplete rows, this is important for calculating the regression
    df = df.dropna(subset=["My Char", "My ELO", "Opponent ELO"])

    # validating stages and characters to ensure there are only correct values
    validation_rules = {
        "Opponent Char": characters,
        "G1 Stage": all_stages,
        "G2 Stage": all_stages,
        "G3 Stage": all_stages + [None, np.nan],
        "G2 char (if different)": characters + [None, np.nan],
        "G3 char (if different)": characters + [None, np.nan],
    }

    valid_rows = pd.Series(True, index=df.index)

    for column, allowed_values in validation_rules.items():
        invalid_rows = ~df[column].isin(allowed_values)

        if invalid_rows.any():
            for index, value in df.loc[invalid_rows, column].items():
                print(
                    f"Error: Value '{value}' in column '{column}' at row {index} is not allowed. This row will be removed for the current analysis",
                    file=sys.stderr,
                )
                # print(*args, file=sys.stderr, **kwargs)

        valid_rows &= ~invalid_rows

    df = df.loc[valid_rows].reset_index(drop=True)

    df["Datetime"] = pd.to_datetime(df["Date"])
    df["Row Index"] = df.index + 1

    # Imputing Opponent characters for games 2 & 3
    df["G2 char (if different)"] = df.apply(
        lambda row: (
            row["Opponent Char"]
            if pd.notnull(row["G2 Stage"]) and pd.isnull(row["G2 char (if different)"])
            else row["G2 char (if different)"]
        ),
        axis=1,
    )

    df["G3 char (if different)"] = df.apply(
        lambda row: (
            row["Opponent Char"]
            if pd.notnull(row["G3 Stage"]) and pd.isnull(row["G3 char (if different)"])
            else row["G3 char (if different)"]
        ),
        axis=1,
    )

    df = df.rename(
        columns={
            "Opponent Char": "G1 Char",
            "G2 char (if different)": "G2 Char",
            "G3 char (if different)": "G3 Char",
        }
    )

    df["Main"] = df.apply(
        lambda row: (
            row["G1 Char"]
            if (row["G2 Char"] == row["G1 Char"] and pd.isnull(row["G3 Char"]))
            or (row["G2 Char"] == row["G1 Char"] and row["G3 Char"] == row["G1 Char"])
            else "Multiple"
        ),
        axis=1,
    )

    return df


# calculating longer dataframe for game-by-game statistics
def calculate_gamewise_df(full_df: pd.DataFrame) -> pd.DataFrame:
    long_df = full_df.melt(
        id_vars=[
            "Date",
            "Time",
            "My ELO",
            "Opponent ELO",
        ],
        value_vars=[
            "G1 Stage",
            "G1 Stock Diff",
            "G1 Char",
            "G2 Stage",
            "G2 Stock Diff",
            "G2 Char",
            "G3 Stage",
            "G3 Stock Diff",
            "G3 Char",
        ],
        var_name="Game Info",
        value_name="Value",
    )

    long_df["Game"] = long_df["Game Info"].str.extract(r"(G\d)")
    long_df["Attribute"] = long_df["Game Info"].str.extract(r"(Stage|Stock Diff|Char)")
    long_df = long_df.pivot(
        index=[
            "Date",
            "Time",
            "My ELO",
            "Opponent ELO",
            "Game",
        ],
        columns="Attribute",
        values="Value",
    ).reset_index()

    long_df.dropna(subset=["Char", "Stage", "Stock Diff"], inplace=True)
    long_df.reset_index(inplace=True, drop=True)

    # calculating winrates for stages
    long_df["Win"] = long_df["Stock Diff"] > 0
    return long_df


# calculating winrate for each character
def calculate_set_winrates(full_df: pd.DataFrame) -> pd.DataFrame:
    winrate_df = (
        full_df.groupby("Main")
        .agg(
            Wins=("Win/Loss", lambda x: (x == "W").sum()),
            Total_Matches=("Win/Loss", "count"),
        )
        .assign(WinRate=lambda x: (x["Wins"] / x["Total_Matches"]) * 100)
        .reset_index()
    )
    return winrate_df


# double bar graph for stages
def calculate_stage_winrates(gamewise_df: pd.DataFrame) -> pd.DataFrame:
    stage_winrate_df = (
        gamewise_df.groupby("Stage")
        .agg(
            Wins=("Win", lambda x: (x == True).sum()),
            Total_Matches=("Win", "count"),
        )
        .assign(WinRate=lambda x: (x["Wins"] / x["Total_Matches"]) * 100)
        .reset_index()
    )
    return stage_winrate_df
