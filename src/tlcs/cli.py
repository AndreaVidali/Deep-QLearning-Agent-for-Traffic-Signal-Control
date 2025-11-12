from pathlib import Path
from typing import Annotated

import typer

from tlcs.constants import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TEST_FOLDER,
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


def check_training_path(out_path: Path) -> None:
    if out_path.exists():
        typer.echo(f"⚠️  A folder already exists with path '{out_path}'.")
        confirm = typer.confirm(
            "Continuing will possibly overwrite the existing training data and model."
            " Do you want to continue?",
            default=False,
        )
        if not confirm:
            typer.echo("Training cancelled.")
            raise typer.Abort


def check_testing_path(model_path: Path, test_name: str) -> None:
    model_files = list(model_path.glob("*.pt"))

    if not any(model_files):
        typer.echo(f"Model file (*.pt) not found in model path: '{model_path}'.")
        raise typer.Abort

    test_folder = model_path / test_name

    if test_folder.exists():
        typer.echo(f"⚠️  The test folder '{test_folder}' already exists.")
        confirm = typer.confirm(
            "Continuing will overwrite the content of the test folder. Do you want to continue?",
            default=False,
        )
        if not confirm:
            typer.echo("Testing cancelled.")
            raise typer.Abort


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
    check_training_path(out_path)

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
    test_name: Annotated[
        str,
        typer.Option(
            help="The name of the test folder.",
        ),
    ] = DEFAULT_TEST_FOLDER,
):
    check_testing_path(model_path=model_path, test_name=test_name)
    testing_session(settings_file=settings_file, model_path=model_path, test_name=test_name)


if __name__ == "__main__":
    app()
