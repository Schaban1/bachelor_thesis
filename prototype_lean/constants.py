from enum import Enum
from pathlib import Path

class WebUIState(str, Enum):
    INIT_STATE = "Initial Iteration"
    MAIN_STATE = "Main Loop"
    GENERATING_STATE = "Generating"

class ScoreMode(str, Enum):
    SLIDERS = "sliders"

RESOURCES_DIR = Path(__file__).resolve().parent / "resources"