from enum import Enum

class RecommendationType(Enum):
    BASELINE = "Random Recommendations varying Latents only."
    RANDOM = "Random Recommendations varying Latents and Embeddings."
    EMA_DIRICHLET = "Random Recommendations with a EMA User-Profile"
    SIMPLE = "Intelligent prompt generation based on probabilistic User-Profile"
    HYPERSPHERICAL_RANDOM = "Recommendations (randomly sampled) lie on an orthonormal basis on a circumscribed hypersphere"
    HYPERSPHERICAL_MOVING_CENTER = "Recommendations (around a moving center) lie on an orthonormal basis on a circumscribed hypersphere"
    HYPERSPHERICAL_BAYESIAN = "Recommendations (chosen with Bayesian optimization) lie on an orthonormal basis on a circumscribed hypersphere"
    # Outdated
    WEIGHTED_AXES = "Single point generation with weighted axes"
    EMA_WEIGHTED_AXES = "Single point generation with weighted axes using exponential moving average"
    FUNCTION_BASED = "Function-based generation"
    DIVERSE_DIRICHLET = "Dirichlet-based with multiple preference-based centers"

class WebUIState(Enum):
    INIT_STATE = "Initial iteration"
    MAIN_STATE = "Main loop iteration"
    GENERATING_STATE = "Generating"
    PLOT_STATE = "Interactive plot"

class ScoreMode(Enum):
    SLIDER = "slider"
    EMOJI = "emoji"
