import dash
from dash import dcc, html, Input, Output
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys

from graph_utils import *
from game_data import stages, characters, character_icons
from df_utils import *


setwise_df = parse_spreadsheet("rivals_spreadsheet.tsv")
character_set_winrate_df = calculate_set_character_winrates(setwise_df)
gamewise_df = calculate_gamewise_df(setwise_df)
stage_winrate_df = calculate_stage_winrates(gamewise_df)
# print(stage_winrate_df)
character_game_winrate_df = calculate_game_character_winrates(gamewise_df)

stage_bar = double_bar_plot_stages(
    title="Stage Winrates",
    stage_winrate_df=stage_winrate_df,
    # x_axis=stage_winrate_df["Stage"],
    # y1_axis=stage_winrate_df["Total_Matches"],
    y1_name="Number of Matches",
    y1_axis_label="Frequency of Stage",
    # y2_axis=stage_winrate_df["WinRate"],
    y2_name="Winrate",
    y2_axis_label="Winrate",
)


elo_scatter = scatterplot_with_icons(
    independent=setwise_df["My ELO"],
    dependent=setwise_df["Opponent ELO"],
    title="My ELO vs. Opponent ELO",
    x_title="My ELO",
    y_title="Opponent ELO",
    df=setwise_df,
)

stage_dimension_scatter = make_stage_scatter(
    stage_winrate_df=stage_winrate_df,
    title="Stage Width vs. Winrate",
    x_title="Stage Width",
    y_title="Winrate",
    # dependent=stage_winrate_df["WinRate"],
    independent_var="Stage_Width",
)

# double bar graph for # matchups and winrate against each character

matchup_bar = character_setwise_bar_plot(
    title="Character Matchup Winrates",
    x_axis=character_set_winrate_df["Main"],
    y1_axis=character_set_winrate_df["Total_Matches"],
    y1_name="Number of Matches",
    y1_axis_label="Number of Matches",
    y2_axis=character_set_winrate_df["WinRate"],
    y2_name="Winrate",
    y2_axis_label="Winrate",
)

elo_plot = make_elo_line_plot(
    x=setwise_df["Row Index"],
    y=setwise_df["My ELO"],
    title="ELO Over Time",
    x_label="By Set",
    y_label="ELO",
    df=setwise_df,
)

"""elo_histogram = make_elo_histogram(
    x=setwise_df["ELO Diff"], x_label="ELO Difference", y_label="Count of Sets"
)"""
elo_histogram = make_elo_mirror_histogram(
    setwise_df=setwise_df,
    x_label="ELO Difference",
    y_label="Counts",
    title="ELO Histogram",
)

boxplot = go.Figure(
    go.Box(
        x=setwise_df["ELO Diff"],
        boxpoints="all",
        jitter=0.3,
        pointpos=0,
        marker=dict(color="green"),
        name="ELO Diff",
        customdata=setwise_df[
            ["Main", "Win/Loss", "Breakdown", "My ELO", "Opponent ELO"]
        ],
        hovertemplate=(
            "Opponent Main: %{customdata[0]}<br>"
            "Opponent ELO: %{customdata[4]}<br>"
            "My ELO: %{customdata[3]}<br>"
            "Set Outcome: %{customdata[1]}<br>"
            "Game Breakdown: %{customdata[2]}<br>"
            "<extra></extra>"
        ),
    )
)

# Add layout details
boxplot.update_layout(
    title="Box-and-Whisker Plot of ELO Diff",
    xaxis_title="ELO Diff",
    template="plotly_white",
)


app = dash.Dash(__name__)

char_options = ["All Characters"] + characters


@app.callback(Output("stage-bar-plot", "figure"), [Input("character-filter", "value")])
def update_stage_bar_graph(selected_character):
    if selected_character == "All Characters":
        filtered_df = gamewise_df
    else:
        filtered_df = gamewise_df.filter(pl.col("Char") == selected_character)

    stage_winrate_df = calculate_stage_winrates(filtered_df)

    figure = double_bar_plot_stages(
        title=f"Stage Winrates Against {selected_character}",
        stage_winrate_df=stage_winrate_df,
        y1_name="Number of Matches",
        y1_axis_label="Frequency of Stage",
        y2_name="Winrate",
        y2_axis_label="Winrate",
    )
    return figure


@app.callback(Output("elo-line-plot", "figure"), [Input("elo-line-filter", "value")])
def update_elo_line(date_vs_set):
    if date_vs_set == "By Set":
        elo_plot = make_elo_line_plot(
            x=setwise_df["Row Index"],
            y=setwise_df["My ELO"],
            title="ELO Over Time",
            x_label="Set Number",
            y_label="ELO",
            df=setwise_df,
        )
    else:
        elo_plot = elo_double_line_plot(
            setwise_df=setwise_df, title="ELO Over Time", x_label="Date", y_label="ELO"
        )
    return elo_plot


@app.callback(
    Output("character-bar", "figure"), [Input("character-set-game-filter", "value")]
)
def update_character_bars(character_set_game):
    if character_set_game == "By Set":
        matchup_bar = character_setwise_bar_plot(
            title="Character Matchup Winrates By Set",
            x_axis=character_set_winrate_df["Main"],
            y1_axis=character_set_winrate_df["Total_Matches"],
            y1_name="Number of Sets",
            y1_axis_label="Number of Sets",
            y2_axis=character_set_winrate_df["WinRate"],
            y2_name="Winrate",
            y2_axis_label="Winrate",
        )
        matchup_bar.update_layout(xaxis_title="Character (Main)")
    else:
        matchup_bar = character_gamewise_bar_plot(
            title="Character Matchup Winrates By Game",
            x_axis=character_game_winrate_df["Char"],
            y1_axis=character_game_winrate_df["Total_Matches"],
            y1_name="Number of Games",
            y1_axis_label="Number of Games",
            y2_axis=character_game_winrate_df["WinRate"],
            y2_name="Winrate",
            y2_axis_label="Winrate",
            df=character_game_winrate_df,
        )
        matchup_bar.update_layout(xaxis_title="Character")
    return matchup_bar


@app.callback(
    Output("stage-dimension-scatter", "figure"), [Input("stage-stat-selector", "value")]
)
def update_stage_dimension_scatter(stage_dimension):
    stage_dimension_scatter = make_stage_scatter(
        stage_winrate_df=stage_winrate_df,
        title=f"Stage {stage_dimension} vs. Winrate",
        x_title=f"Stage {stage_dimension}",
        y_title="Winrate",
        # dependent=stage_winrate_df["WinRate"],
        independent_var=stage_dimension,
    )
    return stage_dimension_scatter


app.layout = html.Div(
    [
        html.H1("ELO Analysis Dashboard"),
        dcc.Tabs(
            id="tabs",
            value="tab-elo",
            children=[
                dcc.Tab(
                    label="ELO Data",
                    value="tab-elo",
                    children=[
                        html.H2("ELO Line Plot"),
                        dcc.Dropdown(
                            id="elo-line-filter",
                            options=["By Set", "By Date"],
                            value="By Set",
                        ),
                        dcc.Graph(
                            id="elo-line-plot",
                            figure=elo_plot,
                        ),
                        dcc.Graph(id="elo-scatter", figure=elo_scatter),
                        html.Div(
                            children=[
                                dcc.Graph(
                                    id="elo-histogram",
                                    figure=elo_histogram,
                                    style={"width": "48%", "display": "inline-block"},
                                ),
                                dcc.Graph(
                                    id="elo-boxplot",
                                    figure=boxplot,
                                    style={"width": "48%", "display": "inline-block"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justify-content": "space-between",
                            },
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Character Data",
                    value="tab-character",
                    children=[
                        dcc.Dropdown(
                            id="character-set-game-filter",
                            options=["By Set", "By Game"],
                            value="By Set",
                        ),
                        dcc.Graph(id="character-bar", figure=matchup_bar),
                    ],
                ),
                dcc.Tab(
                    label="Stage Data",
                    value="tab-stage",
                    children=[
                        dcc.Dropdown(
                            id="character-filter",
                            options=[
                                {"label": char, "value": char} for char in char_options
                            ],
                            value="All Characters",
                            placeholder="Select a character",
                        ),
                        dcc.Graph(id="stage-bar-plot", figure=stage_bar),
                        dcc.Dropdown(
                            id="stage-stat-selector",
                            options={
                                "Stage_Width": "Width",
                                "Top_Blast": "Top Blastzone",
                                "Side_Blast": "Side Blastzone",
                                "Bot_Blast": "Bottom Blastzone",
                            },
                            value="Stage_Width",
                            placeholder="Select a stage dimension",
                        ),
                        dcc.Graph(
                            id="stage-dimension-scatter", figure=stage_dimension_scatter
                        ),
                    ],
                ),
            ],
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
