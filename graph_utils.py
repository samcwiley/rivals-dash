from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import polars as pl
import plotly.graph_objects as go
from game_data import all_stages, character_icons
from PIL import Image
import numpy as np


def double_bar_plot_stages(
    title: str,
    stage_winrate_df: pl.DataFrame,
    # x_axis: pl.Series,
    # y1_axis: pl.Series,
    y1_name: str,
    y1_axis_label: str,
    # y2_axis: pl.Series,
    y2_name: str,
    y2_axis_label: str,
) -> go.Figure:
    customdata = stage_winrate_df[
        [
            "Picks_Bans",
            "My_Counterpick",
            "Their_Counterpick",
            "Pick/Ban_Winrate",
            "My_Counterpick_Winrate",
            "Their_Counterpick_Winrate",
        ]
    ].to_numpy()
    double_bar = go.Figure(
        data=[
            go.Bar(
                name=y1_name,
                x=stage_winrate_df["Stage"].to_list(),
                y=stage_winrate_df["Total_Matches"].to_list(),
                yaxis="y",
                offsetgroup=1,
                customdata=customdata,
                hovertemplate=(
                    "Stage: %{x}<br>"
                    "Total Matches: %{y}<br>"
                    "Picks/Bans: %{customdata[0]}<br>"
                    "My Counterpick: %{customdata[1]}<br>"
                    "Their Counterpick: %{customdata[2]}<br>"
                    "<extra></extra>"
                ),
            ),
            go.Bar(
                name=y2_name,
                x=stage_winrate_df["Stage"].to_list(),
                y=stage_winrate_df["WinRate"].to_list(),
                yaxis="y2",
                offsetgroup=2,
                customdata=customdata,
                hovertemplate=(
                    "Stage: %{x}<br>"
                    "Winrate: %{y}%<br>"
                    "Picks/Bans Winrate: %{customdata[3]}<br>"
                    "My Counterpick Winrate: %{customdata[4]}<br>"
                    "Their Counterpick Winrate: %{customdata[5]}<br>"
                    "<extra></extra>"
                ),
            ),
        ],
        layout={
            "yaxis": {"title": y1_axis_label},
            "yaxis2": {"title": y2_axis_label, "overlaying": "y", "side": "right"},
        },
    )
    double_bar.update_layout(
        barmode="group",
        title=title,
        xaxis={
            "categoryorder": "category ascending",
        },
        yaxis2_range=[0, 100],
    )
    add_50_percent_line(double_bar)
    return double_bar


def character_double_bar_plot(
    title: str,
    x_axis: pl.Series,
    y1_axis: pl.Series,
    y1_name: str,
    y1_axis_label: str,
    y2_axis: pl.Series,
    y2_name: str,
    y2_axis_label: str,
    # customdata,
) -> go.Figure:
    double_bar = go.Figure(
        data=[
            go.Bar(
                name=y1_name,
                x=x_axis.to_list(),
                y=y1_axis.to_list(),
                yaxis="y",
                offsetgroup=1,
                # customdata=customdata,
                hovertemplate=(
                    "Opponent Character: %{x}<br>"
                    "Total Matches: %{y}%<br>"
                    "<extra></extra>"
                ),
            ),
            go.Bar(
                name=y2_name,
                x=x_axis.to_list(),
                y=y2_axis.to_list(),
                yaxis="y2",
                offsetgroup=2,
                # customdata=customdata,
                hovertemplate=(
                    "Opponent Character: %{x}<br>"
                    "Winrate: %{y}%<br>"
                    "<extra></extra>"
                ),
            ),
        ],
        layout={
            "yaxis": {"title": y1_axis_label},
            "yaxis2": {"title": y2_axis_label, "overlaying": "y", "side": "right"},
        },
    )
    double_bar.update_layout(
        barmode="group",
        title=title,
        xaxis={
            "categoryorder": "category ascending",
        },
        yaxis2_range=[0, 100],
    )
    add_50_percent_line(double_bar)
    return double_bar


def scatterplot_with_regression(
    independent: pl.Series, dependent: pl.Series, title: str, x_title: str, y_title: str
) -> go.Figure:
    x = independent.to_numpy().reshape(-1, 1)
    y = dependent.to_numpy()

    model = LinearRegression()
    model.fit(x, y)

    y_pred = model.predict(x)
    m = model.coef_[0]
    b = model.intercept_
    r2 = r2_score(y, y_pred)

    scatter = go.Figure()

    scatter.add_trace(
        go.Scatter(
            x=independent.to_list(),
            y=dependent.to_list(),
            mode="markers",
            name="Data Points",
            marker=dict(color="blue", size=8),
        )
    )
    scatter.add_trace(
        go.Scatter(
            x=independent.to_list(),
            y=y_pred.tolist(),
            mode="lines",
            name=f"Best Fit: y = {m:.2f}x + {b:.2f} (R² = {r2:.2f})",
            line=dict(color="red", width=2, dash="dash"),
        )
    )

    scatter.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title="Legend",
        template="plotly_white",
    )
    return scatter


def make_elo_histogram(
    x: pl.Series, x_label: str, title: str, y_label: str
) -> go.Figure:

    histogram = go.Figure(
        go.Histogram(
            x=x,
            nbinsx=20,
            marker=dict(color="blue"),
        )
    )

    histogram.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
    )
    return histogram


def make_elo_mirror_histogram(
    setwise_df: pl.DataFrame, x_label: str, y_label: str, title: str
) -> go.Figure:
    wins = setwise_df.filter(pl.col("Win/Loss") == "W").select(["ELO Diff"]).to_series()
    losses = (
        setwise_df.filter(pl.col("Win/Loss") == "L").select(["ELO Diff"]).to_series()
    )

    elo_min = min(wins.min(), losses.min())
    elo_max = max(wins.max(), losses.max())

    bin_edges = np.arange(elo_min // 10 * 10, elo_max // 10 * 10 + 10, 10)

    win_counts, _ = np.histogram(wins, bins=bin_edges)
    loss_counts, _ = np.histogram(losses, bins=bin_edges)

    loss_counts = -loss_counts

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=bin_edges[:-1],
            y=win_counts,
            width=10,
            marker_color="blue",
            name="ELO Dist. of Sets Won",
            offset=0,
        )
    )

    fig.add_trace(
        go.Bar(
            x=bin_edges[:-1],
            y=loss_counts,
            width=10,
            marker_color="red",
            name="ELO Dist. of Sets Lost",
            offset=0,
        )
    )

    fig.update_layout(
        title="Mirrored Histogram of ELO Differences",
        xaxis_title="ELO Difference",
        yaxis_title="Count",
        barmode="overlay",
        bargap=0,
    )
    return fig


def scatterplot_with_icons(
    independent: pl.Series,
    dependent: pl.Series,
    title: str,
    x_title: str,
    y_title: str,
    df: pl.DataFrame,
) -> go.Figure:
    x = independent.to_numpy().reshape(-1, 1)
    y = dependent.to_numpy()

    model = LinearRegression()
    model.fit(x, y)

    y_pred = model.predict(x)
    m = model.coef_[0]
    b = model.intercept_
    r2 = r2_score(y, y_pred)

    scatter = go.Figure()
    customdata = df[["Main", "Breakdown", "Win/Loss"]].to_numpy()
    scatter.add_trace(
        go.Scatter(
            x=independent.to_list(),
            y=dependent.to_list(),
            mode="markers",
            name="Data Points",
            marker=dict(color="blue", size=8),
            customdata=customdata,
            hovertemplate=(
                "Opponent Main: %{customdata[0]}<br>"
                "Opponent ELO: %{y}<br>"
                "My ELO: %{x}<br>"
                "Set Outcome: %{customdata[2]}<br>"
                "Game Breakdown: %{customdata[1]}<br>"
                "<extra></extra>"
            ),
        )
    )
    scatter.add_trace(
        go.Scatter(
            x=independent.to_list(),
            y=y_pred.tolist(),
            mode="lines",
            name=f"Best Fit: y = {m:.2f}x + {b:.2f} (R² = {r2:.2f})",
            line=dict(color="red", width=2, dash="dash"),
        )
    )
    for x, y, png in zip(scatter.data[0].x, scatter.data[0].y, df["Icon_Path"]):
        scatter.add_layout_image(
            x=x,
            y=y,
            source=Image.open(png),
            xref="x",
            yref="y",
            sizex=40,
            sizey=40,
            xanchor="center",
            yanchor="middle",
        )

    scatter.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title="Legend",
        template="plotly_white",
    )
    return scatter


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


def make_line_plot(
    x: pl.Series, y: pl.Series, title: str, x_label: str, y_label: str
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=title))

    fig.update_layout(
        title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white"
    )
    return fig


def make_elo_line_plot(
    x: pl.Series, y: pl.Series, title: str, x_label: str, y_label: str, df: pl.DataFrame
) -> go.Figure:
    customdata = df[["Main", "My ELO", "Breakdown", "Win/Loss"]].to_numpy()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=title,
            customdata=customdata,
            hovertemplate=(
                "Opponent Main: %{customdata[0]}<br>"
                "Opponent ELO: %{y}<br>"
                "My ELO: %{customdata[1]}<br>"
                "Set Outcome: %{customdata[3]}<br>"
                "Game Breakdown: %{customdata[2]}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
    )
    return fig


def elo_double_line_plot(
    setwise_df: pl.DataFrame, title: str, x_label: str, y_label: str
) -> go.Figure:
    datewise_df = setwise_df.group_by("Date").agg(
        [
            pl.col("My ELO").min().alias("Minimum ELO"),
            pl.col("My ELO").max().alias("Maximum ELO"),
            pl.col("My ELO").mean().round().alias("Average ELO"),
        ]
    )
    datewise_df = datewise_df.sort("Date")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=datewise_df["Date"],
            y=datewise_df["Maximum ELO"],
            mode="lines",
            name="Maximum ELO",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=datewise_df["Date"],
            y=datewise_df["Average ELO"],
            mode="lines",
            name="Average ELO",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=datewise_df["Date"],
            y=datewise_df["Minimum ELO"],
            mode="lines",
            name="Minimum ELO",
        )
    )

    fig.update_layout(
        title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white"
    )
    return fig
