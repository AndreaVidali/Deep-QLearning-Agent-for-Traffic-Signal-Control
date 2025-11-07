from pathlib import Path

import numpy as np

ROUTE_FILE = Path("TLCS/intersection/episode_routes.rou.xml")

HEADER = """<routes>
    <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

    <route id="W_N" edges="W2TL TL2N"/>
    <route id="W_E" edges="W2TL TL2E"/>
    <route id="W_S" edges="W2TL TL2S"/>
    <route id="N_W" edges="N2TL TL2W"/>
    <route id="N_E" edges="N2TL TL2E"/>
    <route id="N_S" edges="N2TL TL2S"/>
    <route id="E_W" edges="E2TL TL2W"/>
    <route id="E_N" edges="E2TL TL2N"/>
    <route id="E_S" edges="E2TL TL2S"/>
    <route id="S_W" edges="S2TL TL2W"/>
    <route id="S_N" edges="S2TL TL2N"/>
    <route id="S_E" edges="S2TL TL2E"/>"""

STRAIGHT_ROUTES = ["W_E", "E_W", "N_S", "S_N"]
TURN_ROUTES = ["W_N", "W_S", "N_W", "N_E", "E_N", "E_S", "S_W", "S_E"]

STRAIGHT_CHANCHE = 0.75


def _map_to_interval(values: np.ndarray, new_min: int, new_max: int) -> np.ndarray:
    """
    Linearly map an array of values to [new_min, new_max].
    Falls back to a constant array if the input has zero range.
    """
    old_min = float(np.min(values))
    old_max = float(np.max(values))
    return np.interp(values, (old_min, old_max), (new_min, new_max))


def _get_car_row(route_id: str, car_i: int, step: int) -> str:
    return f'    <vehicle id="{route_id}_{car_i}" type="standard_car" route="{route_id}" depart="{step}" departLane="random" departSpeed="10" />'


def _get_route_id(rng: np.random.Generator, routes: list[str]) -> str:
    route_idx = int(rng.integers(0, len(routes)))
    return routes[route_idx]


def generate_routefile(seed: int, n_cars_generated: int, max_steps: int) -> None:
    """
    Generate a SUMO route file for one episode.

    Cars are generated according to a Weibull distribution and re-scaled to fit
    [0, max_steps]. About 75% go straight, 25% turn.
    """
    if n_cars_generated <= 0:
        raise ValueError("n_cars_generated must be > 0")
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    rng = np.random.default_rng(seed)

    # the generation of cars is distributed according to a weibull distribution
    timings = np.sort(rng.weibull(2, size=n_cars_generated).astype(float))

    # reshape the distribution to fit the interval 0:max_steps
    generated_steps = _map_to_interval(timings, new_min=0, new_max=max_steps)

    # round every value to int -> effective steps when a car will be generated
    depart_steps = np.rint(generated_steps).astype(int).tolist()

    ROUTE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(ROUTE_FILE, "w", encoding="utf-8") as routes:
        print(HEADER, file=routes)

        for car_i, step in enumerate(depart_steps):
            # 75% straight, 25% turning
            routes_selected = STRAIGHT_ROUTES if rng.random() < STRAIGHT_CHANCHE else TURN_ROUTES
            route_id = _get_route_id(rng, routes_selected)
            car_row = _get_car_row(route_id=route_id, car_i=car_i, step=step)

            print(car_row, file=routes)

        print("</routes>", file=routes)
