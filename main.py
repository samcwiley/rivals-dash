import dash
from dash import dcc, html
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.express as px
import sys

from game_data import stages, characters, starter_stages, counter_stages, all_stages


def add_50_percent_line(fig: go.Figure):
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        xref="paper",
        y0=50,
        y1=50,
        yref="y2",
        line=dict(color="red", width=2, dash="dash"),
    )


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


df = parse_spreadsheet("rivals_spreadsheet.tsv")


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


winrate_df = calculate_set_winrates(df)


# calculating longer dataframe for game-by-game statistics
def calculate_gamewise_df(full_df: pd.DataFrame) -> pd.DataFrame:
    long_df = df.melt(
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


long_df = calculate_gamewise_df(df)


# double bar graph for stages
def calculate_stage_winrates(gamewise_df: pd.DataFrame) -> pd.DataFrame:
    stage_winrate_df = (
        long_df.groupby("Stage")
        .agg(
            Wins=("Win", lambda x: (x == True).sum()),
            Total_Matches=("Win", "count"),
        )
        .assign(WinRate=lambda x: (x["Wins"] / x["Total_Matches"]) * 100)
        .reset_index()
    )
    return stage_winrate_df


stage_winrate_df = calculate_stage_winrates(long_df)


def double_bar_plot(
    x_axis: pd.Series,
    y1_axis: pd.Series,
    y1_name: str,
    y1_axis_label: str,
    y2_axis: pd.Series,
    y2_name: str,
    y2_axis_label: str,
) -> go.Figure:
    double_bar = go.Figure(
        data=[
            go.Bar(
                name=y1_name,
                x=x_axis,
                y=y1_axis,
                yaxis="y",
                offsetgroup=1,
            ),
            go.Bar(
                name=y2_name,
                x=x_axis,
                y=y2_axis,
                yaxis="y2",
                offsetgroup=2,
            ),
        ],
        layout={
            "yaxis": {"title": y1_axis_label},
            "yaxis2": {"title": y2_axis_label, "overlaying": "y", "side": "right"},
        },
    )
    double_bar.update_layout(barmode="group")
    add_50_percent_line(double_bar)
    return double_bar


stage_bar = double_bar_plot(
    x_axis=stage_winrate_df["Stage"],
    y1_axis=stage_winrate_df["Total_Matches"],
    y1_name="Number of Matches",
    y1_axis_label="Frequency of Stage",
    y2_axis=stage_winrate_df["WinRate"],
    y2_name="Winrate",
    y2_axis_label="Winrate",
)


# Calculate regression line and statistics for ELO
x = df["My ELO"].values.reshape(-1, 1)
y = df["Opponent ELO"].values

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)
m = model.coef_[0]
b = model.intercept_
r2 = r2_score(y, y_pred)

elo_scatter = go.Figure()

# Scatter plot of actual data
elo_scatter.add_trace(
    go.Scatter(
        y=df["Opponent ELO"],
        x=df["My ELO"],
        mode="markers",
        name="Data Points",
        marker=dict(color="blue", size=8),
    )
)

elo_scatter.add_trace(
    go.Scatter(
        x=df["My ELO"],
        y=y_pred,
        mode="lines",
        name=f"Best Fit: y = {m:.2f}x + {b:.2f} (R² = {r2:.2f})",
        line=dict(color="red", width=2, dash="dash"),
    )
)
elo_scatter.update_layout(
    title="My ELO vs. Opponent ELO",
    xaxis_title="My ELO",
    yaxis_title="Opponent ELO",
    legend_title="Legend",
    template="plotly_white",
)

stage_winrate_df["Stage_Width"] = stage_winrate_df["Stage"].map(
    lambda stage: stages[stage].width
)

stage_scatter = go.Figure()
stage_scatter.add_trace(
    go.Scatter(
        x=stage_winrate_df["Stage_Width"],
        y=stage_winrate_df["WinRate"],
        mode="markers+text",
        name="Data Points",
        marker=dict(color="blue", size=8),
        text=stage_winrate_df["Stage"],
        textposition="top center",
    )
)
stage_scatter.update_layout(
    title="Stage Width vs. Winrate",
    xaxis_title="Stage Width",
    yaxis_title="Winrate",
    legend_title="Legend",
    template="plotly_white",
)

# double bar graph for # matchups and winrate against each character
matchup_bar = double_bar_plot(
    x_axis=winrate_df["Main"],
    y1_axis=winrate_df["Total_Matches"],
    y1_name="Number of Matches",
    y1_axis_label="Frequency of Opponent Characters",
    y2_axis=winrate_df["WinRate"],
    y2_name="Winrate",
    y2_axis_label="Winrate",
)


app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("ELO Analysis Dashboard"),
        # Line plot of ELO over time
        dcc.Graph(
            id="line-plot",
            figure=px.line(
                df,
                x="Row Index",
                y="My ELO",
                title="My ELO Over Time",
                labels={"Row Index": "Sets Played", "My ELO": "My ELO"},
            ),
        ),
        # Scatter plot of My ELO vs. Opponent ELO
        dcc.Graph(id="scatter-plot", figure=elo_scatter),
        # Matchup bar plot
        dcc.Graph(id="character-bar", figure=matchup_bar),
        # stage winrate double bar plot
        dcc.Graph(id="stage_winrate_double_plot", figure=stage_bar),
        dcc.Graph(
            id="stage_winrate_scatter",
            figure=stage_scatter,
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
