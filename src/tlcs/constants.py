from pathlib import Path

# TODO WARNING: you need to also edit environemnt.net.xml
# TODO based on environemnt.net.xml

SETTINGS_PATH = Path("settings")
TRAINING_SETTINGS_FILE = Path("training_settings.yaml")
TESTING_SETTINGS_FILE = Path("testing_settings.yaml")

DEFAULT_MODEL_PATH = Path("model")
MODEL_FILE = Path("trained_model.pt")

DEFAULT_TEST_FOLDER = "test"

ROUTES_FILE = Path("intersection/episode_routes.rou.xml")

ROUTES_FILE_HEADER = """<routes>
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
    <route id="S_E" edges="S2TL TL2E"/>"""  # noqa: E501

STRAIGHT_ROUTES = ("W_E", "E_W", "N_S", "S_N")
TURN_ROUTES = ("W_N", "W_S", "N_W", "N_E", "E_N", "E_S", "S_W", "S_E")

STRAIGHT_CHANCHE = 0.75  # 75% go straight, 25% turn

# based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


ACTION_TO_TL_PHASE = {
    0: PHASE_NS_GREEN,
    1: PHASE_NSL_GREEN,
    2: PHASE_EW_GREEN,
    3: PHASE_EWL_GREEN,
}

TL_GREEN_TO_YELLOW = {
    PHASE_NS_GREEN: PHASE_NS_YELLOW,
    PHASE_NSL_GREEN: PHASE_NSL_YELLOW,
    PHASE_EW_GREEN: PHASE_EW_YELLOW,
    PHASE_EWL_GREEN: PHASE_EWL_YELLOW,
}

# phase codes based on environment.net.xml
ROAD_MAX_LENGTH = 750

LANE_DISTANCE_TO_CELL = {
    7: 0,
    14: 1,
    21: 2,
    28: 3,
    40: 4,
    60: 5,
    100: 6,
    160: 7,
    400: 8,
    750: 9,
}
