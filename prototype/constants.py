from enum import Enum

class RecommendationType(Enum):
    POINT = "Single point generation"
    WEIGHTED_AXES = "Single point generation with weighted axes"
    FUNCTION_BASED = "Function-based generation"

class OptimizationType(Enum):
    MAX_PREF = "Maximum preference optimization"
    WEIGHTED_SUM = "Weighted sum optimization"
    GAUSSIAN_PROCESS = "Gaussian process regression"

class WebUIState(Enum):
    INIT_STATE = "Initial iteration"
    MAIN_STATE = "Main loop iteration"
    GENERATING_STATE = "Generating"
