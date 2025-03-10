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
        global generator_batch_count
        if self.webUI.iteration < 2:
            generator_batch_count = int(math.ceil((self.webUI.num_images_to_generate*self.webUI.first_iteration_images_factor)/self.webUI.args.generator.batch_size))
        else:
            generator_batch_count = int(math.ceil(self.webUI.num_images_to_generate/self.webUI.args.generator.batch_size))
    
    def reset_progress_bar(self):
        global current_step
        current_step = 0
        loading_progress.set_value(0)
    
def update_progess(pipe, step_index, timestep, callback_kwargs):
    global current_step
    global generator_batch_count
    global loading_progress
    current_step += 1
    loading_progress.set_value(current_step/(pipe.num_timesteps * generator_batch_count))
    return callback_kwargs
