import dash
from dash import dcc, html
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys

from graph_utils import double_bar_plot, scatterplot_with_regression
from game_data import stages
from df_utils import (
    parse_spreadsheet,
    calculate_gamewise_df,
    calculate_set_winrates,
    calculate_stage_winrates,
)


df = parse_spreadsheet("rivals_spreadsheet.tsv")
winrate_df = calculate_set_winrates(df)
long_df = calculate_gamewise_df(df)
stage_winrate_df = calculate_stage_winrates(long_df)

stage_bar = double_bar_plot(
    x_axis=stage_winrate_df["Stage"],
    y1_axis=stage_winrate_df["Total_Matches"],
    y1_name="Number of Matches",
    y1_axis_label="Frequency of Stage",
    y2_axis=stage_winrate_df["WinRate"],
    y2_name="Winrate",
    y2_axis_label="Winrate",
)


elo_scatter = scatterplot_with_regression(
    independent=df["My ELO"],
    dependent=df["Opponent ELO"],
    title="My ELO vs. Opponent ELO",
    x_title="My ELO",
    y_title="Opponent ELO",
)

stage_winrate_df["Stage_Width"] = stage_winrate_df["Stage"].map(
    lambda stage: stages[stage].width
)

stage_scatter = scatterplot_with_regression(
    independent=stage_winrate_df["Stage_Width"],
    dependent=stage_winrate_df["WinRate"],
    title="Stage Width vs. Winrate",
    x_title="Stage Width",
    y_title="Winrate",
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
