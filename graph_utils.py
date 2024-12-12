from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import plotly.graph_objects as go


def double_bar_plot(
    title: str,
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
    double_bar.update_layout(barmode="group", title=title)
    add_50_percent_line(double_bar)
    return double_bar


def scatterplot_with_regression(
    independent: pd.Series, dependent: pd.Series, title: str, x_title: str, y_title: str
) -> go.Figure:
    x = independent.values.reshape(-1, 1)
    y = dependent.values

    model = LinearRegression()
    model.fit(x, y)

    y_pred = model.predict(x)
    m = model.coef_[0]
    b = model.intercept_
    r2 = r2_score(y, y_pred)

    scatter = go.Figure()

    # Scatter plot of actual data
    scatter.add_trace(
        go.Scatter(
            y=dependent,
            x=independent,
            mode="markers",
            name="Data Points",
            marker=dict(color="blue", size=8),
        )
    )

    scatter.add_trace(
        go.Scatter(
            x=independent,
            y=y_pred,
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