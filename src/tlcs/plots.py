from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt


def save_data_and_plot(  # noqa: PLR0913
    data: Sequence[float | int],
    filename: str,
    xlabel: str,
    ylabel: str,
    out_folder: Path,
    dpi: int = 96,
) -> None:
    """Save numeric data to a text file and a corresponding line plot image.

    Args:
        data: Sequence of numeric values to plot and save.
        filename: Base name (without extension) for output files.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        out_folder: Directory where output files are written.
        dpi: Resolution in dots per inch for the saved plot image.
    """
    out_folder.mkdir(parents=True, exist_ok=True)

    plot_file = out_folder / f"plot_{filename}.png"
    data_file = out_folder / f"plot_{filename}_data.txt"

    min_val = min(data)
    max_val = max(data)
    margin_min = 0.05 * abs(min_val)
    margin_max = 0.05 * abs(max_val)

    plt.rcParams.update({"font.size": 24})

    fig, ax = plt.subplots(figsize=(20, 11.25))
    ax.plot(data)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.margins(0)
    ax.set_ylim(min_val - margin_min, max_val + margin_max)
    fig.tight_layout()
    fig.savefig(plot_file, dpi=dpi)
    plt.close(fig)

    data_text = "\n".join(str(value) for value in data) + "\n"
    data_file.write_text(data_text, encoding="utf-8")
