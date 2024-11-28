from enum import Enum

class RecommendationType(Enum):
    POINT = "Single point generation"
    WEIGHTED_AXES = "Single point generation with weighted axes"
    FUNCTION_BASED = "Function-based generation"

class WebUIState(Enum):
    INIT_STATE = "Initial iteration"
    MAIN_STATE = "Main loop iteration"
    GENERATING_STATE = "Generating"
