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
# df_pandas = parse_spreadsheet("rivals_spreadsheet.tsv")
# winrate_df = calculate_set_winrates(df_pandas)
winrate_df_polars = calculate_set_winrates_polars(df_polars)
gamewise_df_polars = calculate_gamewise_df_polars(df_polars)
stage_winrate_df_polars = calculate_stage_winrates_polars(gamewise_df_polars)


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


elo_scatter_polars = scatterplot_with_regression_polars(
    independent=df_polars["My ELO"],
    dependent=df_polars["Opponent ELO"],
    title="My ELO vs. Opponent ELO",
    x_title="My ELO",
    y_title="Opponent ELO",
)


stages_df = pl.DataFrame(
    {
        "Stage": list(stages.keys()),
        "Stage_Width": [stages[stage].width for stage in stages],
    }
)
stage_winrate_df_polars = stage_winrate_df_polars.join(stages_df, on="Stage")

stage_scatter_polars = scatterplot_with_regression_polars(
    independent=stage_winrate_df_polars["Stage_Width"],
    dependent=stage_winrate_df_polars["WinRate"],
    title="Stage Width vs. Winrate (Polars)",
    x_title="Stage Width",
    y_title="Winrate",
)

# double bar graph for # matchups and winrate against each character

matchup_bar_polars = double_bar_plot_polars(
    title="Character Matchup Winrates (Polars)",
    x_axis=winrate_df_polars["Main"],
    y1_axis=winrate_df_polars["Total_Matches"],
    y1_name="Number of Matches",
    y1_axis_label="Number of Matches",
    y2_axis=winrate_df_polars["WinRate"],
    y2_name="Winrate",
    y2_axis_label="Winrate",
)


app = dash.Dash(__name__)

char_options = ["All Characters"] + characters


# Callback to update stage winrate plot
@app.callback(
    Output("stage-bar-plot-polars", "figure"), [Input("character-filter", "value")]
)
def update_graph_polars(selected_character):
    if selected_character == "All Characters":
        filtered_df = gamewise_df_polars
    else:
        filtered_df = gamewise_df_polars.filter(pl.col("Char") == selected_character)

    stage_winrate_df = calculate_stage_winrates_polars(filtered_df)

    figure = double_bar_plot_polars(
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
        # dcc.Graph(
        # id="line-plot",
        # figure=px.line(
        # df_pandas,
        # x="Row Index",
        # y="My ELO",
        # title="My ELO Over Time",
        # labels={"Row Index": "Sets Played", "My ELO": "My ELO"},
        # ),
        # ),
        dcc.Graph(
            id="line-plot-polars",
            figure=px.line(
                df_polars,
                x="Row Index",
                y="My ELO",
                title="My ELO Over Time",
                labels={"Row Index": "Sets Played", "My ELO": "My ELO"},
            ),
        ),
        # Scatter plot of My ELO vs. Opponent ELO
        dcc.Graph(id="elo-scatter-polars", figure=elo_scatter_polars),
        # Matchup bar plot
        dcc.Graph(id="character-bar-polars", figure=matchup_bar_polars),
        # stage winrate double bar plot
        dcc.Graph(id="stage_bar_polars", figure=stage_bar_polars),
        dcc.Graph(id="stage_winrate_scatter_polars", figure=stage_scatter_polars),
        dcc.Dropdown(
            id="character-filter",
            options=[{"label": char, "value": char} for char in char_options],
            value="All Characters",
            placeholder="Select a character",
        ),
        dcc.Graph(id="stage-bar-plot-polars"),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
