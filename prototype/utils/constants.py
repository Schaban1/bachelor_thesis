from enum import Enum

class RecommendationTypes(Enum):
    POINT = "Single point generation"
    WEIGHTED_AXES = "Single point generation with weighted axes"
    FUNCTION_BASED = "Function-based generation"

class WebUIStates(Enum):
    INITIAL_ITERATION = "Initial iteration"
    MAIN_LOOP_ITERATION = "Main loop iteration"
    GENERATING = "Generating"
