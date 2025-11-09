from pathlib import Path
from typing import Annotated

import typer

from tlcs.constants import (
    DEFAULT_MODEL_PATH,
    SETTINGS_PATH,
    TESTING_SETTINGS_FILE,
    TRAINING_SETTINGS_FILE,
)
from tlcs.main import testing_session, training_session

app = typer.Typer(
    help="Train and run TLCS traffic light control simulations.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command(name="train", help="Train a new TLCS model using the specified settings file.")
def cmd_train(
    settings_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="Path to the YAML file containing training parameters.",
        ),
    ] = SETTINGS_PATH / TRAINING_SETTINGS_FILE,
    out_path: Annotated[
        Path,
        typer.Option(help="Directory where training outputs and trained model will be saved."),
    ] = DEFAULT_MODEL_PATH,
):
    model_files = list(out_path.glob("*.pt"))
    if out_path.exists() and any(model_files):
        typer.echo(f"⚠️  A trained model already exists in '{out_path}'.")
        confirm = typer.confirm(
            "Continuing will overwrite the existing model. Do you want to continue?",
            default=False,
        )
        if not confirm:
            typer.echo("Training cancelled.")
            raise typer.Abort

    training_session(
        settings_file=settings_file,
        out_path=out_path,
    )


@app.command(name="test", help="Run a simulation test using a trained TLCS model.")
def cmd_test(
    settings_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="Path to the YAML file containing testing parameters.",
        ),
    ] = SETTINGS_PATH / TESTING_SETTINGS_FILE,
    model_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="Path to the directory containing the trained TLCS model to test.",
        ),
    ] = DEFAULT_MODEL_PATH,
):
    testing_session(settings_file=settings_file, model_path=model_path)


if __name__ == "__main__":
    app()
