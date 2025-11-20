import numpy as np
from numpy.typing import NDArray

from tlcs.constants import (
    ROUTES_FILE,
    ROUTES_FILE_HEADER,
    STRAIGHT_ROUTES,
    TURN_ROUTES,
)


def _map_to_interval(values: NDArray, new_min: int, new_max: int) -> NDArray:
    """Linearly map values to the interval [new_min, new_max].

    Falls back to a constant array if the input has zero range.

    Args:
        values: Input array of values to be re-scaled.
        new_min: Minimum value of the target interval.
        new_max: Maximum value of the target interval.

    Returns:
        Array of values mapped to the target interval, same shape as `values`.
    """
    old_min = float(values.min())
    old_max = float(values.max())
    return np.interp(values, (old_min, old_max), (new_min, new_max))


def _get_car_row(route_id: str, car_i: int, step: int) -> str:
    """Build the XML row describing a single vehicle.

    Args:
        route_id: Identifier of the route the vehicle will follow.
        car_i: Index of the vehicle in the episode.
        step: Simulation step at which the vehicle departs.

    Returns:
        XML snippet representing the vehicle element.
    """
    return f'    <vehicle id="{route_id}_{car_i}" type="standard_car" route="{route_id}" depart="{step}" departLane="random" departSpeed="10" />'  # noqa: E501


def generate_routefile(
    seed: int,
    n_cars_generated: int,
    max_steps: int,
    turn_chance: float,
) -> None:
    """Generate a SUMO route file for one simulation episode.

    Car departure times follow a Weibull distribution re-scaled to [0, max_steps].
    A fraction of cars go straight and the rest turn, controlled by turn_chance.

    Args:
        seed: Random seed for reproducible generation.
        n_cars_generated: Number of cars to generate in the episode.
        max_steps: Maximum simulation step for car departures.
        turn_chance: Probability to select a turn route rather than a straight route.
    """
    rng = np.random.default_rng(seed)

    # Generate departure timings according to a Weibull distribution
    timings = np.sort(rng.weibull(2.0, size=n_cars_generated))

    # Rescale the distribution to the interval [0, max_steps]
    generated_steps = _map_to_interval(timings, new_min=0, new_max=max_steps)

    # Round to integer steps -> effective depart times for each car
    depart_steps = np.rint(generated_steps).astype(int)

    ROUTES_FILE.parent.mkdir(parents=True, exist_ok=True)

    with ROUTES_FILE.open("w", encoding="utf-8") as routes_file:
        print(ROUTES_FILE_HEADER, file=routes_file)

        for car_i, step in enumerate(depart_steps):
            routes_selected = TURN_ROUTES if rng.random() < turn_chance else STRAIGHT_ROUTES
            route_id = rng.choice(routes_selected)
            car_row = _get_car_row(route_id=route_id, car_i=car_i, step=step)

            print(car_row, file=routes_file)

        print("</routes>", file=routes_file)
