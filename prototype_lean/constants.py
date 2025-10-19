from enum import Enum

class WebUIState(str, Enum):
    INIT_STATE = "Initial Iteration"
    MAIN_STATE = "Main Loop"
    GENERATING_STATE = "Generating"

class ScoreMode(str, Enum):
    SLIDERS = "sliders"