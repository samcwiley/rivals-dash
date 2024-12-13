import polars as pl
import sys
from datetime import datetime
from game_data import characters, all_stages

pl.Config.set_tbl_rows(1000)
pl.Config.set_tbl_cols(100)


def parse_spreadsheet(filepath: str) -> pl.DataFrame:
    df = pl.read_csv(filepath, separator="\t")
    # removing notes and game goals, these are priveleged information!
    if "Notes" in df.columns:
        df = df.drop(["Notes", "Goal"])
        df.write_csv(filepath, separator="\t")
    # removing rows where it seems the set didn't happen, e.g. game bugs where it crashes or they forfeit before game 1 starts
    # these null values must be dropped so we can calculate the linear regression
    df = df.drop_nulls(["My Char", "My ELO", "Opponent ELO"])
    # validating data to make sure there are no invalid characters or stages listed
    validation_rules = {
        "Opponent Char": characters,
        "G1 Stage": all_stages,
        "G2 Stage": all_stages,
        "G3 Stage": all_stages,
        "G2 char (if different)": characters,
        "G3 char (if different)": characters,
    }

    valid_rows = pl.Series([True] * len(df))

    for column, allowed_values in validation_rules.items():
        if column in df.columns:
            is_valid = df[column].is_in(allowed_values) | df[column].is_null()

            invalid_rows = ~is_valid

            if invalid_rows.any():
                invalid_indices = invalid_rows.to_numpy().nonzero()[0]
                for index in invalid_indices:
                    value = df[column][index]
                    print(
                        f"Error: Value '{value}' in column '{column}' at row {index} is not allowed. This row will be removed for the current analysis",
                        file=sys.stderr,
                    )

            valid_rows &= is_valid

    df = df.filter(valid_rows)

    # Impute Opponent characters for games 2 & 3
    df = df.with_columns(
        [
            pl.when(
                (pl.col("G2 Stage").is_not_null())
                & (pl.col("G2 char (if different)").is_null())
            )
            .then(pl.col("Opponent Char"))
            .otherwise(pl.col("G2 char (if different)"))
            .alias("G2 char (if different)"),
            pl.when(
                (pl.col("G3 Stage").is_not_null())
                & (pl.col("G3 char (if different)").is_null())
            )
            .then(pl.col("Opponent Char"))
            .otherwise(pl.col("G3 char (if different)"))
            .alias("G3 char (if different)"),
        ]
    )

    df = df.rename(
        {
            "Opponent Char": "G1 Char",
            "G2 char (if different)": "G2 Char",
            "G3 char (if different)": "G3 Char",
        }
    )
    # Figuring out their main for set winrate stats
    df = df.with_columns(
        pl.when(
            ((pl.col("G2 Char") == pl.col("G1 Char")) & pl.col("G3 Char").is_null())
            | (
                (pl.col("G2 Char") == pl.col("G1 Char"))
                & (pl.col("G3 Char") == pl.col("G1 Char"))
            )
        )
        .then(pl.col("G1 Char"))
        .otherwise(pl.lit("Multiple"))
        .alias("Main")
    )
    # Adding a row index for counting sets
    df = df.with_row_count(name="Row Index").with_columns(
        (pl.col("Row Index") + 1).alias("Row Index")
    )

    # imputing who chose the stage for each game
    df = df.with_columns(
        [
            pl.lit("Picks/Bans").alias("G1 Stage_Choice"),
            pl.when(pl.col("G1 Stock Diff") < 0)
            .then(pl.lit("My Counterpick"))
            .otherwise(pl.lit("Their Counterpick"))
            .alias("G2 Stage_Choice"),
            pl.when(pl.col("G3 Stage").is_not_null())
            .then(
                pl.when(pl.col("G2 Stock Diff") < 0)
                .then(pl.lit("My Counterpick"))
                .otherwise(pl.lit("Their Counterpick"))
            )
            .otherwise(pl.lit(""))
            .alias("G3 Stage_Choice"),
        ]
    )

    return df


def calculate_gamewise_df(full_df: pl.DataFrame) -> pl.DataFrame:
    long_df = full_df.melt(
        id_vars=["Date", "Time", "My ELO", "Opponent ELO"],
        value_vars=[
            "G1 Stage",
            "G1 Stock Diff",
            "G1 Char",
            "G1 Stage_Choice",
            "G2 Stage",
            "G2 Stock Diff",
            "G2 Char",
            "G2 Stage_Choice",
            "G3 Stage",
            "G3 Stock Diff",
            "G3 Char",
            "G3 Stage_Choice",
        ],
        variable_name="Game Info",
        value_name="Value",
    )

    long_df = long_df.with_columns(
        [
            long_df["Game Info"].str.extract(r"(G\d)").alias("Game"),
            long_df["Game Info"]
            .str.extract(r"(Stage_Choice|Stock Diff|Char|Stage)")
            .alias("Attribute"),
        ]
    )

    long_df = long_df.pivot(
        index=["Date", "Time", "My ELO", "Opponent ELO", "Game"],
        columns="Attribute",
        values="Value",
    )

    long_df = long_df.drop_nulls(["Char", "Stage", "Stock Diff", "Stage_Choice"])

    long_df = long_df.with_columns(
        (long_df["Stock Diff"].cast(pl.Float64) > 0).alias("Win")
    )

    return long_df


def calculate_set_winrates(full_df: pl.DataFrame) -> pl.DataFrame:
    winrate_df = (
        full_df.group_by("Main")
        .agg(
            [
                (pl.col("Win/Loss") == "W").sum().alias("Wins"),
                pl.col("Win/Loss").count().alias("Total_Matches"),
            ]
        )
        .with_columns((pl.col("Wins") / pl.col("Total_Matches") * 100).alias("WinRate"))
        .sort("Main")
    )
    return winrate_df


def calculate_stage_winrates(gamewise_df: pl.DataFrame) -> pl.DataFrame:
    stage_winrate_df = (
        gamewise_df.group_by("Stage")
        .agg(
            [
                (pl.col("Win") == True).sum().alias("Wins"),
                pl.col("Win").count().alias("Total_Matches"),
                (pl.col("Stage_Choice") == "Picks/Bans").sum().alias("Picks_Bans"),
                (pl.col("Stage_Choice") == "My Counterpick")
                .sum()
                .alias("My_Counterpick"),
                (pl.col("Stage_Choice") == "Their Counterpick")
                .sum()
                .alias("Their_Counterpick"),
            ]
        )
        .with_columns((pl.col("Wins") / pl.col("Total_Matches") * 100).alias("WinRate"))
    )
    return stage_winrate_df
