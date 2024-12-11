import dash
from dash import dcc, html
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.express as px

from game_data import stages, characters, starter_stages, counter_stages


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


# read the csv
df = pd.read_csv("rivals_spreadsheet.tsv", sep="\t")

# cutting out goals and notes, these are priveleged information!
if "Notes" in df.columns:
    df = df.drop(columns=["Notes", "Goal"])
    df.to_csv("rivals_spreadsheet.tsv", sep="\t", header=True)

# Killing incomplete rows, this is important for calculating the regression
df = df.dropna(subset=["My Char", "My ELO", "Opponent ELO"])

df["Datetime"] = pd.to_datetime(df["Date"])
df["Row Index"] = df.index + 1

# calculating winrate for each character
winrate_df = (
    df.groupby("Opponent Char")
    .agg(
        Wins=("Win/Loss", lambda x: (x == "W").sum()),
        Total_Matches=("Win/Loss", "count"),
    )
    .assign(WinRate=lambda x: (x["Wins"] / x["Total_Matches"]) * 100)
    .reset_index()
)

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

# calculating longer dataframe for game-by-game statistics
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

# double bar graph for stages

stage_winrate_df = (
    long_df.groupby("Stage")
    .agg(
        Wins=("Win", lambda x: (x == True).sum()),
        Total_Matches=("Win", "count"),
    )
    .assign(WinRate=lambda x: (x["Wins"] / x["Total_Matches"]) * 100)
    .reset_index()
)

stage_bar = go.Figure(
    data=[
        go.Bar(
            name="Number of Matches",
            x=stage_winrate_df["Stage"],
            y=stage_winrate_df["Total_Matches"],
            yaxis="y",
            offsetgroup=1,
        ),
        go.Bar(
            name="Winrate",
            x=stage_winrate_df["Stage"],
            y=stage_winrate_df["WinRate"],
            yaxis="y2",
            offsetgroup=2,
        ),
    ],
    layout={
        "yaxis": {"title": "Frequency of Stage"},
        "yaxis2": {"title": "Winrate", "overlaying": "y", "side": "right"},
    },
)
stage_bar.update_layout(barmode="group")
add_50_percent_line(stage_bar)


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
matchup_bar = go.Figure(
    data=[
        go.Bar(
            name="Number of Matches",
            x=winrate_df["Opponent Char"],
            y=winrate_df["Total_Matches"],
            yaxis="y",
            offsetgroup=1,
        ),
        go.Bar(
            name="Winrate",
            x=winrate_df["Opponent Char"],
            y=winrate_df["WinRate"],
            yaxis="y2",
            offsetgroup=2,
        ),
    ],
    layout={
        "yaxis": {"title": "Frequency of Opponent Characters"},
        "yaxis2": {"title": "Winrate", "overlaying": "y", "side": "right"},
    },
)
matchup_bar.update_layout(barmode="group")
add_50_percent_line(matchup_bar)


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
        # stage winrate plot
        # dcc.Graph(id="stage_winrate_plot", figure=stage_winrate_plot),
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
