import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys

from graph_utils import *
from game_data import stages, characters
from df_utils import *
from df_utils_polars import *


df_polars = parse_spreadsheet_polars("rivals_spreadsheet.tsv")
df_pandas = parse_spreadsheet("rivals_spreadsheet.tsv")
# print(df_polars.shape)
# print(df_pandas.shape)
# print(df_pandas.columns)
# print(df_polars.columns)
winrate_df = calculate_set_winrates(df_pandas)
winrate_df_polars = calculate_set_winrates_polars(df_polars)
# print(winrate_df.head)
# print(winrate_df_polars.head)
gamewise_df = calculate_gamewise_df(df_pandas)
gamewise_df_polars = calculate_gamewise_df_polars(df_polars)
# print(gamewise_df.shape)
# print(gamewise_df.head)
# print(gamewise_df_polars.shape)
# print(gamewise_df_polars.head)
stage_winrate_df = calculate_stage_winrates(gamewise_df)
stage_winrate_df_polars = calculate_stage_winrates_polars(gamewise_df_polars)
# print(stage_winrate_df_polars.head)
# print(stage_winrate_df.head)

stage_bar = double_bar_plot(
    title="Stage Winrates (pandas)",
    x_axis=stage_winrate_df["Stage"],
    y1_axis=stage_winrate_df["Total_Matches"],
    y1_name="Number of Matches",
    y1_axis_label="Frequency of Stage",
    y2_axis=stage_winrate_df["WinRate"],
    y2_name="Winrate",
    y2_axis_label="Winrate",
)

stage_bar_polars = double_bar_plot_polars(
    title="Stage Winrates (polars)",
    x_axis=stage_winrate_df_polars["Stage"],
    y1_axis=stage_winrate_df_polars["Total_Matches"],
    y1_name="Number of Matches",
    y1_axis_label="Frequency of Stage",
    y2_axis=stage_winrate_df_polars["WinRate"],
    y2_name="Winrate",
    y2_axis_label="Winrate",
)


elo_scatter = scatterplot_with_regression(
    independent=df_pandas["My ELO"],
    dependent=df_pandas["Opponent ELO"],
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
    title="Character Matchup Winrates",
    x_axis=winrate_df["Main"],
    y1_axis=winrate_df["Total_Matches"],
    y1_name="Number of Matches",
    y1_axis_label="Frequency of Opponent Characters",
    y2_axis=winrate_df["WinRate"],
    y2_name="Winrate",
    y2_axis_label="Winrate",
)


app = dash.Dash(__name__)

char_options = ["All Characters"] + characters


# Callback to update stage winrate plot
@app.callback(Output("stage-bar-plot", "figure"), [Input("character-filter", "value")])
def update_graph(selected_character):
    if selected_character == "All Characters":
        filtered_df = gamewise_df
    else:
        filtered_df = gamewise_df[gamewise_df["Char"] == selected_character]

    stage_winrate_df = calculate_stage_winrates(filtered_df)

    figure = double_bar_plot(
        title=f"Stage Winrates Against {selected_character}",
        x_axis=stage_winrate_df["Stage"],
        y1_axis=stage_winrate_df["Total_Matches"],
        y1_name="Number of Matches",
        y1_axis_label="Frequency of Stage",
        y2_axis=stage_winrate_df["WinRate"],
        y2_name="Winrate",
        y2_axis_label="Winrate",
    )
    return figure


app.layout = html.Div(
    [
        html.H1("ELO Analysis Dashboard"),
        # Line plot of ELO over time
        dcc.Graph(
            id="line-plot",
            figure=px.line(
                df_pandas,
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
        dcc.Graph(id="stage_bar_polars", figure=stage_bar_polars),
        dcc.Graph(
            id="stage_winrate_scatter",
            figure=stage_scatter,
        ),
        # For filtering character bar plot
        dcc.Dropdown(
            id="character-filter",
            options=[{"label": char, "value": char} for char in char_options],
            value="All Characters",
            placeholder="Select a character",
        ),
        dcc.Graph(id="stage-bar-plot"),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
