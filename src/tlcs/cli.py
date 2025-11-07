from pathlib import Path
from typing import Annotated

import typer

from tlcs.main import testing_session, training_session

app = typer.Typer(
    help="Train and run TLCS traffic light control simulations.",
    add_completion=False,
    no_args_is_help=True,
)


TRAINING_CONFIG_FILE = Path("settings/training_settings.yaml")
TESTING_CONFIG_FILE = Path("settings/testing_settings.yaml")

DEFAULT_OUT_PATH = Path("model")


@app.command(help="Train a new TLCS model using the specified configuration file.")
def train(
    config_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="Path to the YAML file containing training parameters.",
        ),
    ] = TRAINING_CONFIG_FILE,
    out_path: Annotated[
        Path,
        typer.Option(help="Directory where training outputs and trained model will be saved."),
    ] = DEFAULT_OUT_PATH,
):
    training_session(
        config_file=config_file,
        out_path=out_path,
    )


@app.command(help="Run a simulation test using a trained TLCS model.")
def test(
    config_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="Path to the YAML file containing testing parameters.",
        ),
    ] = TESTING_CONFIG_FILE,
    model_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="Path to the directory containing the trained TLCS model to test.",
        ),
    ] = DEFAULT_OUT_PATH,
):
    testing_session(config_file=config_file, model_path=model_path)


if __name__ == "__main__":
    app()
