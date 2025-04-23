from nicegui import ui as ngUI

from prototype.webuserinterface.components.ui_component import UIComponent

generator_batch_count = 1


class LoadingUI(UIComponent):
    """
    Contains the code for the loading UI.
    """

    def build_userinterface(self):
        """
        Builds the UI for the generating state.
        """
        with ngUI.column().classes('mx-auto items-center').bind_visibility_from(self.webUI, 'is_generating', value=True):
            ngUI.label('Generating images...').style('font-size: 200%;')
            ngUI.space().classes('m-4')
            self.loading_progress = ngUI.linear_progress(value=0, show_value=False, color='#323232').props('size=md;')
        self.webUI.generator.callback = update_progress
    
    def reset_progress_bar(self):
        """
        Resets the progress bar to 0.
        """
        self.loading_progress.set_value(0)
    
def update_progress(pipe, step_index, timestep, callback_kwargs, current_step, num_embeddings, loading_progress, batch_size, num_steps):
    """
    This function serves as the callback_function for the StableDiffusion-Pipeline to show the generation progress on the UI.

    Args:
        pipe: StableDiffusion-Pipeline
        step_index: Index of the callback step
        timestep: Current timestep
        callback_kwargs: Provided by the pipeline
        current_step: Current step in the generation process
        num_embeddings: Number of embeddings
        loading_progress: The linear_progress instance of the UI to update
        batch_size: The size of the batch
        num_steps: Number of inference steps

    Returns:
        callback_kwargs that were input
    """
    loading_progress.set_value((current_step+(step_index/num_steps*batch_size))/num_embeddings)
    return callback_kwargs
