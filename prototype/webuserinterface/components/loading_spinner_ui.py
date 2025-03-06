from nicegui import ui as ngUI
import math

from prototype.webuserinterface.components.ui_component import UIComponent

current_step = 0
generator_num_batches = 1
loading_progress = None


class LoadingSpinnerUI(UIComponent):
    """
    Contains the code for the loading spinner UI.
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
        global generator_num_batches
        generator_num_batches = int(math.ceil(self.webUI.num_images_to_generate/self.webUI.args.generator.batch_size))

        self.webUI.generator.callback = update_progess
    
def update_progess(pipe, step_index, timestep, callback_kwargs):
    global current_step
    global generator_num_batches
    global loading_progress
    current_step += 1
    loading_progress.set_value(current_step/(pipe.num_timesteps * generator_num_batches))
    if current_step == (pipe.num_timesteps * generator_num_batches):
        current_step = 0
    return callback_kwargs
