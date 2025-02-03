from nicegui import ui as ngUI
import torch

from prototype.constants import ScoreMode


class Scorer:
    """
    This class handles the scoring logic in the WebUI.
    """

    def __init__(self, webUI):
        """
        Initializes the scorer and automatically registers scoring functions based on the WebUI scoring mode.

        Args:
            webUI: The parent WebUI instance that uses this component.
        """
        self.webUI = webUI

        self.scores_slider = [None for _ in range(self.webUI.num_images_to_generate * self.webUI.first_iteration_images_factor)] # For convenience already initialized here
        self.scores_toggles = [None for _ in range(self.webUI.num_images_to_generate * self.webUI.first_iteration_images_factor)] # For convenience already initialized here

        self.init_score_mode()
    
    def init_score_mode(self):
        """
        Registers some functions based on the current self.webUI.score_mode.
        """
        if self.webUI.score_mode == ScoreMode.SLIDER.value:
            self.build_scorer = self.build_slider
            self.get_scores = self.get_scores_slider
            self.reset_scorers = self.reset_sliders
        elif self.webUI.score_mode == ScoreMode.EMOJI.value:
            self.build_scorer = self.build_emoji_toggle
            self.get_scores = self.get_scores_emoji_toggles
            self.reset_scorers = self.reset_emoji_toggles
        else:
            print(f"Unknown score mode: {self.webUI.score_mode}")
    
    def build_slider(self, idx):
        """
        Registers a slider object at position idx.

        Args:
            idx: The index of the slider.
        """
        self.scores_slider[idx] = ngUI.slider(min=0, max=10, value=0, step=0.1)
        ngUI.label().bind_text_from(self.scores_slider[idx], 'value')
    
    def build_emoji_toggle(self, idx):
        """
        Registers a toggle object at position idx.

        Args:
            idx: The index of the toggle object.
        """
        self.scores_toggles[idx] = ngUI.toggle({0: '😢1', 1: '🙁2', 2: '😐3', 3: '😄4', 4: '😍5'}, value=0).props('toggle-color=grey-8 rounded')
    
    def get_scores_slider(self):
        """
        Get the normalized scores provided by the user with the sliders.

        Returns:
            The normalized scores as a one-dim tensor of shape (num_images_to_generate).
        """
        scores = torch.FloatTensor([slider.value for slider in self.scores_slider if slider.visible])
        normalized_scores = scores / 10
        return normalized_scores
    
    def get_scores_emoji_toggles(self):
        """
        Get the normalized scores provided by the user with the emoji toggle buttons.

        Returns:
            The normalized scores as a one-dim tensor of shape (num_images_to_generate).
        """
        scores = torch.FloatTensor([toggle.value for toggle in self.scores_toggles if toggle.visible])
        normalized_scores = scores / 4
        return normalized_scores

    def reset_sliders(self):
        """
        Reset the value of the score sliders to the default value.
        """
        [slider.set_value(0) for slider in self.scores_slider]
    
    def reset_emoji_toggles(self):
        """
        Reset the value of the score toggles to the default value.
        """
        [toggle.set_value(0) for toggle in self.scores_toggles]
