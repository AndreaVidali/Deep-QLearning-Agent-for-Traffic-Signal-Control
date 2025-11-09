import numpy as np

from tlcs.constants import (
    ROUTES_FILE,
    ROUTES_FILE_HEADER,
    STRAIGHT_CHANCHE,
    STRAIGHT_ROUTES,
    TURN_ROUTES,
)


def _map_to_interval(values: np.ndarray, new_min: int, new_max: int) -> np.ndarray:
    """
    Linearly map an array of values to [new_min, new_max].
    Falls back to a constant array if the input has zero range.
    """
    old_min = float(np.min(values))
    old_max = float(np.max(values))
    return np.interp(values, (old_min, old_max), (new_min, new_max))


def _get_car_row(route_id: str, car_i: int, step: int) -> str:
    return f'    <vehicle id="{route_id}_{car_i}" type="standard_car" route="{route_id}" depart="{step}" departLane="random" departSpeed="10" />'  # noqa: E501


def _get_route_id(rng: np.random.Generator, routes: list[str]) -> str:
    route_idx = int(rng.integers(0, len(routes)))
    return routes[route_idx]


def generate_routefile(seed: int, n_cars_generated: int, max_steps: int) -> None:
    """
    Generate a SUMO route file for one episode.

    Cars are generated according to a Weibull distribution and re-scaled to fit
    [0, max_steps]. About 75% go straight, 25% turn.
    """
    rng = np.random.default_rng(seed)

    # the generation of cars is distributed according to a weibull distribution
    timings = np.sort(rng.weibull(2, size=n_cars_generated).astype(float))

    # reshape the distribution to fit the interval 0:max_steps
    generated_steps = _map_to_interval(timings, new_min=0, new_max=max_steps)

    # round every value to int -> effective steps when a car will be generated
    depart_steps = np.rint(generated_steps).astype(int).tolist()

    ROUTES_FILE.parent.mkdir(parents=True, exist_ok=True)

    with ROUTES_FILE.open("w", encoding="utf-8") as routes:
        print(ROUTES_FILE_HEADER, file=routes)

        for car_i, step in enumerate(depart_steps):
            routes_selected = STRAIGHT_ROUTES if rng.random() < STRAIGHT_CHANCHE else TURN_ROUTES
            route_id = _get_route_id(rng, routes_selected)
            car_row = _get_car_row(route_id=route_id, car_i=car_i, step=step)

            print(car_row, file=routes)

        print("</routes>", file=routes)
