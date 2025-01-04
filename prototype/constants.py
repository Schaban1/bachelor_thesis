from enum import Enum

class RecommendationType(Enum):
    RANDOM = "Random Recommendations"
    POINT = "Single point generation"
    WEIGHTED_AXES = "Single point generation with weighted axes"
    EMA_WEIGHTED_AXES = "Single point generation with weighted axes using exponential moving average"
    FUNCTION_BASED = "Function-based generation"

class WebUIState(Enum):
    INIT_STATE = "Initial iteration"
    MAIN_STATE = "Main loop iteration"
    GENERATING_STATE = "Generating"
    PLOT_STATE = "Interactive plot"

class ScoreMode(Enum):
    SLIDER = "slider"
    EMOJI = "emoji"
