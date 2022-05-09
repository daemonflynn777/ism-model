import matplotlib.pyplot as plt
from typing import List, Dict


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
    data_x: List[float],
    data_y: List[float],
    title: str,
    x_label: str,
    y_label: str,
    save_path: str
) -> None:
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.scatter(
        x=data_x,
        y=data_y
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(save_path)
    return None
