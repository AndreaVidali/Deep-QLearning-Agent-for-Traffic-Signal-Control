from pathlib import Path
from typing import Annotated

import typer

from tlcs.constants import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SETTINGS_PATH,
    DEFAULT_TEST_FOLDER,
    TESTING_SETTINGS_FILE,
    TRAINING_SETTINGS_FILE,
)
from tlcs.main import testing_session, training_session

app = typer.Typer(
    help="Train and run TLCS.",
    add_completion=False,
    no_args_is_help=True,
)


def _confirm_overwrite_directory(
    directory: Path,
    overwrite_message: str,
    cancel_message: str,
) -> None:
    """Ask the user to confirm overwriting an existing directory.

    Args:
        directory: The directory that potentially will be overwritten.
        overwrite_message: The message prompting the user to confirm overwrite.
        cancel_message: The message displayed if the user cancels.
    """
    typer.echo(f"⚠️  The folder '{directory}' already exists.")
    confirm = typer.confirm(overwrite_message, default=False)
    if not confirm:
        typer.echo(cancel_message)
        raise typer.Abort


def check_training_path(out_path: Path) -> None:
    """Ensure training output path is safe to use.

    Args:
        out_path: Directory where training outputs and model will be saved.
    """
    if out_path.exists():
        _confirm_overwrite_directory(
            directory=out_path,
            overwrite_message=(
                "Continuing will possibly overwrite the existing training data and model. "
                "Do you want to continue?"
            ),
            cancel_message="Training cancelled.",
        )


def check_testing_path(model_path: Path, test_name: str) -> None:
    """Ensure testing model path and test folder are valid.

    Args:
        model_path: Path to the directory containing the trained model.
        test_name: Name of the test folder to create within the model directory.
    """
    model_files = list(model_path.glob("*.pt"))

    if not model_files:
        typer.echo(f"Model file (*.pt) not found in model path: '{model_path}'.")
        raise typer.Abort

    test_folder = model_path / test_name

    if test_folder.exists():
        _confirm_overwrite_directory(
            directory=test_folder,
            overwrite_message=(
                "Continuing will overwrite the content of the test folder. Do you want to continue?"
            ),
            cancel_message="Testing cancelled.",
        )


@app.command(name="train", help="Train a new TLCS model using the specified settings file.")
def cmd_train(
    settings_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="Path to the YAML file containing training parameters.",
        ),
    ] = DEFAULT_SETTINGS_PATH / TRAINING_SETTINGS_FILE,
    out_path: Annotated[
        Path,
        typer.Option(
            help="Directory where training outputs and trained model will be saved.",
        ),
    ] = DEFAULT_MODEL_PATH,
) -> None:
    """CLI command to train a TLCS model.

    Args:
        settings_file: Path to the YAML file with training parameters.
        out_path: Output directory for training artifacts and the trained model.
    """
    check_training_path(out_path)
    training_session(settings_file=settings_file, out_path=out_path)


@app.command(name="test", help="Run a simulation test using a trained TLCS model.")
def cmd_test(
    settings_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="Path to the YAML file containing testing parameters.",
        ),
    ] = DEFAULT_SETTINGS_PATH / TESTING_SETTINGS_FILE,
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
) -> None:
    """CLI command to run a simulation test using a trained TLCS model.

    Args:
        settings_file: Path to the YAML file with testing parameters.
        model_path: Path to the directory containing the trained model.
        test_name: Name of the test folder created under the model directory.
    """
    check_testing_path(model_path=model_path, test_name=test_name)
    testing_session(settings_file=settings_file, model_path=model_path, test_name=test_name)


if __name__ == "__main__":
    app()
