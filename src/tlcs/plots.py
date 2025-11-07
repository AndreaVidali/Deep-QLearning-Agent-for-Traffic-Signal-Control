from pathlib import Path

import matplotlib.pyplot as plt


def save_data_and_plot(
    data: list[float],
    filename: str,
    xlabel: str,
    ylabel: str,
    out_folder: Path,
    dpi: int = 96,
) -> None:
    """
    Produce a plot of performance of the agent over the session and save the relative data to txt.
    """
    out_folder.mkdir(parents=True, exist_ok=True)

    min_val = min(data)
    max_val = max(data)

    plt.rcParams.update({"font.size": 24})

    plt.figure()
    plt.plot(data)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.margins(0)
    plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.tight_layout()
    fig.savefig(out_folder / f"plot_{filename}.png", dpi=dpi)
    plt.close("all")

    with open(out_folder / f"plot_{filename}_data.txt", "w", encoding="utf-8") as file:
        for value in data:
            file.write(f"{value}\n")
