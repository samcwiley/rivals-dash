import dash
from dash import dcc, html, Input, Output
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys

from graph_utils import *
from game_data import stages, characters
from df_utils import *


setwise_df = parse_spreadsheet("rivals_spreadsheet.tsv")
character_winrate_df = calculate_set_winrates(setwise_df)
gamewise_df = calculate_gamewise_df(setwise_df)
stage_winrate_df = calculate_stage_winrates(gamewise_df)


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


elo_scatter = scatterplot_with_regression(
    independent=setwise_df["My ELO"],
    dependent=setwise_df["Opponent ELO"],
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
stage_winrate_df = stage_winrate_df.join(stages_df, on="Stage")

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
    x_axis=character_winrate_df["Main"],
    y1_axis=character_winrate_df["Total_Matches"],
    y1_name="Number of Matches",
    y1_axis_label="Number of Matches",
    y2_axis=character_winrate_df["WinRate"],
    y2_name="Winrate",
    y2_axis_label="Winrate",
)

elo_plot = make_line_plot(
    x=setwise_df["Row Index"],
    y=setwise_df["My ELO"],
    title="ELO Over Time",
    x_label="By Set",
    y_label="ELO",
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
        elo_plot = make_line_plot(
            x=setwise_df["Row Index"],
            y=setwise_df["My ELO"],
            title="ELO Over Time",
            x_label="Set Number",
            y_label="ELO",
        )
    else:
        elo_plot = elo_double_line_plot(
            setwise_df=setwise_df, title="ELO Over Time", x_label="Date", y_label="ELO"
        )
    return elo_plot


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
                    ],
                ),
                dcc.Tab(
                    label="Character Data",
                    value="tab-character",
                    children=[
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
                        dcc.Graph(id="stage_winrate_scatter", figure=stage_scatter),
                    ],
                ),
            ],
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
