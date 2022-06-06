import plotly.graph_objects as go
import plotly.express as px
import plotly

from ism_model.utils.dataset import Data


def plot_metrics_3d(data_path: str):
    # load dataset
    df = Data(source_file=data_path).load()

    # Create figure
    fig = go.Figure()

    # Add surface trace
    fig.add_trace(
        go.Heatmap(
            x=df["a"],
            y=df["b"],
            z=df["corr_metrics"].tolist(),
            colorscale="Viridis"
        )
    )

    fig.add_trace(
        go.Heatmap(
            x=df["a"],
            y=df["b"],
            z=df["MAPE_metrics"].tolist(),
            colorscale="Viridis"
        )
    )

    fig.add_trace(
        go.Heatmap(
            x=df["a"],
            y=df["alpha_k"],
            z=df["corr_metrics"].tolist(),
            colorscale="Viridis"
        )
    )

    fig.add_trace(
        go.Heatmap(
            x=df["a"],
            y=df["alpha_k"],
            z=df["MAPE_metrics"].tolist(),
            colorscale="Viridis"
        )
    )

    fig.add_trace(
        go.Heatmap(
            x=df["a"],
            y=df["mu"],
            z=df["corr_metrics"].tolist(),
            colorscale="Viridis"
        )
    )

    fig.add_trace(
        go.Heatmap(
            x=df["a"],
            y=df["mu"],
            z=df["MAPE_metrics"].tolist(),
            colorscale="Viridis"
        )
    )

    fig.add_trace(
        go.Heatmap(
            x=df["b"],
            y=df["alpha_k"],
            z=df["corr_metrics"].tolist(),
            colorscale="Viridis"
        )
    )

    fig.add_trace(
        go.Heatmap(
            x=df["b"],
            y=df["alpha_k"],
            z=df["MAPE_metrics"].tolist(),
            colorscale="Viridis"
        )
    )

    fig.add_trace(
        go.Heatmap(
            x=df["b"],
            y=df["mu"],
            z=df["corr_metrics"].tolist(),
            colorscale="Viridis"
        )
    )

    fig.add_trace(
        go.Heatmap(
            x=df["b"],
            y=df["mu"],
            z=df["MAPE_metrics"].tolist(),
            colorscale="Viridis"
        )
    )

    fig.add_trace(
        go.Heatmap(
            x=df["alpha_k"],
            y=df["mu"],
            z=df["corr_metrics"].tolist(),
            colorscale="Viridis"
        )
    )

    fig.add_trace(
        go.Heatmap(
            x=df["alpha_k"],
            y=df["mu"],
            z=df["MAPE_metrics"].tolist(),
            colorscale="Viridis"
        )
    )

    # Update plot sizing
    fig.update_layout(
        width=800,
        height=900,
        autosize=False,
        margin=dict(t=100, b=0, l=0, r=0),
    )

    # Update 3D scene options
    fig.update_scenes(
        aspectratio=dict(x=1, y=1, z=0.7),
        aspectmode="manual"
    )

    # Add dropdowns
    button_layer_1_height = 1.08
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        label="a",
                        method="update",
                        args=[
                            {"visible": [
                                True, True, True, True,
                                True, True, False, False,
                                False, False, False, False,
                            ]},
                            {"title": "b"}
                        ]
                    ),
                    dict(
                        label="b",
                        method="update",
                        args=[
                            {"visible": [
                                False, False, False, False,
                                False, False, True, True,
                                False, False, True, True,
                            ]},
                            {"title": "b"}
                        ]
                    ),
                    dict(
                        label="alpha_k",
                        method="update",
                        args=[
                            {"visible": [
                                False, False, False, False,
                                False, False, False, False,
                                False, False, True, True,
                            ]},
                            {"title": "b"}
                        ]
                    ),
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=button_layer_1_height,
                yanchor="top"
            ),
            dict(
                buttons=list([
                    dict(
                        label="b",
                        method="update",
                        args=[
                            {"visible": [
                                True, True, False, False,
                                False, False, False, False,
                                False, False, False, False,
                            ]},
                            {"title": "b"}
                        ]
                    ),
                    dict(
                        label="alpha_k",
                        method="update",
                        args=[
                            {"visible": [
                                True, True, True, True,
                                True, True, True, True,
                                True, True, True, True,
                            ]},
                            {"title": "b"}
                        ]
                    ),
                    dict(
                        label="mu",
                        method="update",
                        args=[
                            {"visible": [
                                False, False, False, False,
                                False, False, False, False,
                                True, True, True, True,
                            ]},
                            {"title": "b"}
                        ]
                    ),
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.37,
                xanchor="left",
                y=button_layer_1_height,
                yanchor="top"
            ),
            dict(
                buttons=list([
                    dict(
                        label="corr",
                        method="update",
                        args=[
                            {"visible": [
                                True, False, True, False,
                                True, False, True, False,
                                True, False, True, False,
                            ]},
                            {"title": "b"}
                        ]
                    ),
                    dict(
                        label="MAPE",
                        method="update",
                        args=[
                            {"visible": [
                                False, True, False, True,
                                False, True, False, True,
                                False, True, False, True,
                            ]},
                            {"title": "b"}
                        ]
                    ),
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.58,
                xanchor="left",
                y=button_layer_1_height,
                yanchor="top"
            ),
        ]
    )

    fig.update_layout(
        annotations=[
            dict(text="x", x=0, xref="paper", y=1.06, yref="paper",
                 align="left", showarrow=False),
            dict(text="y", x=0.25, xref="paper", y=1.07,
                 yref="paper", showarrow=False),
        ])

    fig.show()


def plot_metrics_3d_new(x, y, z, df):
    plotly.offline.init_notebook_mode()

    fig = px.scatter_3d(df, x=x, y=y, z=z, color=z)

    fig.show()
