import polars as pl
import sys
from datetime import datetime
from game_data import characters, all_stages, character_icons

pl.Config.set_tbl_rows(1000)
pl.Config.set_tbl_cols(100)


def parse_spreadsheet(filepath: str) -> pl.DataFrame:
    df = pl.read_csv(filepath, separator="\t")
    # removing notes and game goals, these are priveleged information!
    if "Notes" in df.columns:
        df = df.drop(["Notes", "Goal", "Opponent Name"])
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
    df = df.with_columns(
        pl.col("Date").str.strptime(pl.Date, format="%m/%d/%Y").alias("Date")
    )
    df = df.with_columns((pl.col("My ELO") - pl.col("Opponent ELO")).alias("ELO Diff"))
    df = df.with_columns(pl.col("Main").replace(character_icons).alias("Icon_Path"))

    return df


def calculate_gamewise_df(full_df: pl.DataFrame) -> pl.DataFrame:
    long_df = full_df.melt(
        id_vars=["Date", "Time", "My ELO", "Opponent ELO", "Main"],
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
        index=["Date", "Time", "My ELO", "Opponent ELO", "Game", "Main"],
        columns="Attribute",
        values="Value",
    )

    long_df = long_df.drop_nulls(["Char", "Stage", "Stock Diff", "Stage_Choice"])

    long_df = long_df.with_columns(
        (long_df["Stock Diff"].cast(pl.Float64) > 0).alias("Win")
    )

    return long_df


def calculate_set_character_winrates(full_df: pl.DataFrame) -> pl.DataFrame:
    winrate_df = (
        full_df.group_by("Main")
        .agg(
            [
                (pl.col("Win/Loss") == "W").sum().alias("Wins"),
                pl.col("Win/Loss").count().alias("Total_Matches"),
            ]
        )
        .with_columns(
            ((pl.col("Wins") / pl.col("Total_Matches") * 100).round(2)).alias("WinRate")
        )
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
                ((pl.col("Stage_Choice") == "Picks/Bans") & (pl.col("Win") == True))
                .sum()
                .alias("Picks_Bans_Wins"),
                (pl.col("Stage_Choice") == "My Counterpick")
                .sum()
                .alias("My_Counterpick"),
                ((pl.col("Stage_Choice") == "My Counterpick") & (pl.col("Win") == True))
                .sum()
                .alias("My_Counterpick_Wins"),
                (pl.col("Stage_Choice") == "Their Counterpick")
                .sum()
                .alias("Their_Counterpick"),
                (
                    (pl.col("Stage_Choice") == "Their Counterpick")
                    & (pl.col("Win") == True)
                )
                .sum()
                .alias("Their_Counterpick_Wins"),
            ]
        )
        .with_columns(
            [
                ((pl.col("Wins") / pl.col("Total_Matches") * 100).round(2)).alias(
                    "WinRate"
                ),
                (
                    (pl.col("Picks_Bans_Wins") / pl.col("Picks_Bans") * 100).round(2)
                ).alias("Pick/Ban_Winrate"),
                (
                    (
                        pl.col("My_Counterpick_Wins") / pl.col("My_Counterpick") * 100
                    ).round(2)
                ).alias("My_Counterpick_Winrate"),
                (
                    (
                        pl.col("Their_Counterpick_Wins")
                        / pl.col("Their_Counterpick")
                        * 100
                    ).round(2)
                ).alias("Their_Counterpick_Winrate"),
            ]
        )
    )
    stage_winrate_df = stage_winrate_df.with_columns(
        [
            pl.when(pl.col("WinRate").is_not_nan())
            .then((pl.col("WinRate").cast(str) + " %"))
            .otherwise(pl.lit("N/A"))
            .alias("WinRate_str"),
            pl.when(pl.col("Pick/Ban_Winrate").is_not_nan())
            .then((pl.col("Pick/Ban_Winrate").cast(str) + " %"))
            .otherwise(pl.lit("N/A"))
            .alias("Pick/Ban_Winrate"),
            pl.when(pl.col("My_Counterpick_Winrate").is_not_nan())
            .then((pl.col("My_Counterpick_Winrate").cast(str) + " %"))
            .otherwise(pl.lit("N/A"))
            .alias("My_Counterpick_Winrate"),
            pl.when(pl.col("Their_Counterpick_Winrate").is_not_nan())
            .then((pl.col("Their_Counterpick_Winrate").cast(str) + " %"))
            .otherwise(pl.lit("N/A"))
            .alias("Their_Counterpick_Winrate"),
        ]
    )

    return stage_winrate_df


def calculate_game_character_winrates(gamewise_df: pl.DataFrame) -> pl.DataFrame:
    gamewise_df = gamewise_df.with_columns(
        pl.when(pl.col("Main") == pl.col("Char"))
        .then(pl.lit("Main"))
        .otherwise(pl.lit("Counterpick"))
        .alias("Character_Pick")
    )

    character_winrate_df = (
        gamewise_df.group_by("Char")
        .agg(
            [
                (pl.col("Win") == True).sum().alias("Wins"),
                pl.col("Win").count().alias("Total_Matches"),
            ]
        )
        .with_columns(
            ((pl.col("Wins") / pl.col("Total_Matches") * 100).round(2)).alias("WinRate")
        )
    )

    main_vs_counterpick_df = gamewise_df.group_by(["Char", "Character_Pick"]).agg(
        [
            (pl.col("Win") == True).sum().alias("Wins"),
            pl.col("Win").count().alias("Total_Games"),
        ]
    )

    main_vs_counterpick_df = main_vs_counterpick_df.pivot(
        values=["Total_Games", "Wins"],
        index="Char",
        columns="Character_Pick",
        aggregate_function="sum",
    ).fill_null(0)

    final_df = character_winrate_df.join(main_vs_counterpick_df, on="Char", how="left")

    final_df = final_df.with_columns(
        ((pl.col("Wins_Main") / pl.col("Total_Games_Main") * 100).round(2)).alias(
            "WinRate_Main"
        ),
        (
            (
                pl.col("Wins_Counterpick") / pl.col("Total_Games_Counterpick") * 100
            ).round(2)
        ).alias("WinRate_Counterpick"),
        ((pl.col("Total_Games_Main") / pl.col("Total_Matches") * 100).round(2)).alias(
            "Percent_Main"
        ),
        (
            (pl.col("Total_Games_Counterpick") / pl.col("Total_Matches") * 100).round(2)
        ).alias("Percent_Counterpick"),
    )

    final_df = final_df.with_columns(
        [
            pl.when(pl.col("WinRate").is_not_nan())
            .then((pl.col("WinRate").cast(str) + " %"))
            .otherwise(pl.lit("N/A"))
            .alias("WinRate_str"),
            pl.when(pl.col("WinRate_Main").is_not_nan())
            .then((pl.col("WinRate_Main").cast(str) + " %"))
            .otherwise(pl.lit("N/A")),
            pl.when(pl.col("WinRate_Counterpick").is_not_nan())
            .then((pl.col("WinRate_Counterpick").cast(str) + " %"))
            .otherwise(pl.lit("N/A")),
            pl.when(pl.col("Percent_Main").is_not_nan())
            .then((pl.col("Percent_Main").cast(str) + " %"))
            .otherwise(pl.lit("N/A")),
            pl.when(pl.col("Percent_Counterpick").is_not_nan())
            .then((pl.col("Percent_Counterpick").cast(str) + " %"))
            .otherwise(pl.lit("N/A")),
        ]
    )

    return final_df
