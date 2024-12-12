import polars as pl
import sys
from datetime import datetime
from game_data import characters, all_stages

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(100)


def parse_spreadsheet_polars(filepath: str) -> pl.DataFrame:
    df = pl.read_csv(filepath, separator="\t")

    if "Notes" in df.columns:
        df = df.drop(["Notes", "Goal"])
        df.write_csv(filepath, separator="\t", has_header=True)

    df = df.drop_nulls(["My Char", "My ELO", "Opponent ELO"])

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

    df = df.with_row_count(name="Row Index")

    return df


def calculate_gamewise_df_polars(full_df: pl.DataFrame) -> pl.DataFrame:
    long_df = full_df.melt(
        id_vars=["Date", "Time", "My ELO", "Opponent ELO"],
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
        variable_name="Game Info",
        value_name="Value",
    )

    long_df = long_df.with_columns(
        [
            long_df["Game Info"].str.extract(r"(G\d)").alias("Game"),
            long_df["Game Info"]
            .str.extract(r"(Stage|Stock Diff|Char)")
            .alias("Attribute"),
        ]
    )

    long_df = long_df.pivot(
        index=["Date", "Time", "My ELO", "Opponent ELO", "Game"],
        columns="Attribute",
        values="Value",
    )

    long_df = long_df.drop_nulls(["Char", "Stage", "Stock Diff"])

    long_df = long_df.with_columns(
        (long_df["Stock Diff"].cast(pl.Float64) > 0).alias("Win")
    )

    return long_df


def calculate_set_winrates_polars(full_df: pl.DataFrame) -> pl.DataFrame:
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


def calculate_stage_winrates_polars(gamewise_df: pl.DataFrame) -> pl.DataFrame:
    stage_winrate_df = (
        gamewise_df.group_by("Stage")
        .agg(
            [
                (pl.col("Win") == True).sum().alias("Wins"),
                pl.col("Win").count().alias("Total_Matches"),
            ]
        )
        .with_columns((pl.col("Wins") / pl.col("Total_Matches") * 100).alias("WinRate"))
    )
    return stage_winrate_df
