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

    return df
