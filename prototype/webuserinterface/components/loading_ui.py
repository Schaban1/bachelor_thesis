from nicegui import ui as ngUI
import math

from prototype.webuserinterface.components.ui_component import UIComponent

current_step = 0
generator_batch_count = 1
loading_progress = None


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
            global loading_progress
            loading_progress = ngUI.linear_progress(value=0, show_value=False, color='#323232').props('instant-feedback=True')
        self.update_loading_batch_count()
        self.webUI.generator.callback = update_progess
    
    def update_loading_batch_count(self):
        """
        Updates the generator_batch_count variable, that indicates the number of batches processed by the generator.
        This is used to determine the progress bar steps.
        """
        global generator_batch_count
        if self.webUI.iteration < 2:
            generator_batch_count = int(math.ceil((self.webUI.num_images_to_generate*self.webUI.first_iteration_images_factor)/self.webUI.args.generator.batch_size))
        else:
            generator_batch_count = int(math.ceil(self.webUI.num_images_to_generate/self.webUI.args.generator.batch_size))
    
    def reset_progress_bar(self):
        """
        Resets the progress bar to 0.
        """
        global current_step
        current_step = 0
        loading_progress.set_value(0)
    
def update_progess(pipe, step_index, timestep, callback_kwargs):
    """
    This function serves as the callback_function for the StableDiffusion-Pipeline to show the generation progress on the UI.

    Args:
        pipe: StableDiffusion-Pipeline
        step_index: Index of the callback step
        timestep: Current timestep
        callback_kwargs: Provided by the pipeline

    Returns:
        callback_kwargs that were input
    """
    global current_step
    global generator_batch_count
    global loading_progress
    current_step += 1
    loading_progress.set_value(current_step/(pipe.num_timesteps * generator_batch_count))
    return callback_kwargs
