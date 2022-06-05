from cProfile import label
import matplotlib.pyplot as plt
from typing import List, Dict
import numpy as np
import pandas as pd


def plot_time_series(
    data_y: Dict[str, List[float]],
    colors: List[str],
    title: str,
    x_label: str,
    y_label: str,
    save_path: str
) -> None:
    plt.figure(figsize=(8, 6))
    plt.title(title)
    for key, color in zip(data_y.keys(), colors):
        plt.plot(
            list(range(len(data_y[key]))),
            data_y[key],
            color=color,
            label=key
        )
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(save_path)
    return None


def plot_metrics(
    # data_x: List[float],
    # data_y: List[float],
    # shell: np.array,
    pareto_points_x: List[float],
    pareto_points_y: List[float],
    not_pareto_points_x: List[float],
    not_pareto_points_y: List[float],
    title: str,
    x_label: str,
    y_label: str,
    save_path: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.title(title)
    ax.scatter(
        x=not_pareto_points_x,
        y=not_pareto_points_y,
        c='orange',
        s=25,
        label="Not Pareto front"
    )
    ax.scatter(
        x=pareto_points_x,
        y=pareto_points_y,
        c='green',
        s=35,
        marker="P",
        label="Pareto front"
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(save_path)
    return None


def plot_tube_predictions(
    trace_real: List[float],
    traces_predicted: np.array,
    title: str,
    x_label: str,
    y_label: str,
    save_path: str
) -> None:
    x = np.arange(len(trace_real))

    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.plot(
        x,
        trace_real,
        color='red',
        label="Real"
    )
    plt.plot(
            x,
            traces_predicted[0:1, :].reshape(-1,),
            color='orange',
            alpha=0.25,
            label="Prediction"
        )
    for trace in traces_predicted[1:, :]:
        plt.plot(
            x,
            trace,
            color='orange',
            alpha=0.25
        )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(save_path)
    pass
