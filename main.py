import dash
from dash import dcc, html
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.express as px
from plotly.subplots import make_subplots

df = pd.read_csv("rivals_spreadsheet.tsv", sep="\t")

# cutting out goals and notes, these are priveleged information!
if "Notes" in df.columns:
    df = df.drop(columns=["Notes", "Goal", "Achieved?"])
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

# for making horizontal 50% line on bar graph
matchup_bar.add_shape(
    type="line",
    x0=0,
    x1=1,
    xref="paper",
    y0=50,
    y1=50,
    yref="y2",
    line=dict(color="red", width=2, dash="dash"),
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
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
