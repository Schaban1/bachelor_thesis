from nicegui import ui as ngUI

from prototype.constants import WebUIState, RecommendationType, ScoreMode

class DebugMenu(ngUI.element):
    """
    Contains the code for the debug menu.
    """

    def __init__(self, webUI, bg_color=(0,0,0,0.5)):
        super().__init__(tag='div')
        self.webUI = webUI
        self.style('position: fixed; display: block; width: 100%; height: 100%;'
                   'top: 0; left: 0; right: 0; bottom: 0; z-index: 2;'
                   'background-color:' + f'rgba{bg_color};')
        input_props = 'input-style="color: white" label-color=white dense'
        checkbox_style = 'color: white;'
        with self:
            with ngUI.element('div').style("margin-left: 8px;").classes("h-screen flex"):
                with ngUI.column().classes('p-0 gap-0'):
                    ngUI.label("Debug UI Info").style("font-size: 50px; color: white;")
                    ngUI.input(label="session_id").props(input_props).bind_value(self.webUI, 'session_id')
                    ngUI.select({s: s.value for s in WebUIState}, label='state').props(input_props).bind_value(self.webUI, 'state')
                    ngUI.checkbox('is_initial_iteration').style(checkbox_style).bind_value(self.webUI, 'is_initial_iteration')
                    ngUI.checkbox('is_main_loop_iteration').style(checkbox_style).bind_value(self.webUI, 'is_main_loop_iteration')
                    ngUI.checkbox('is_generating').style(checkbox_style).bind_value(self.webUI, 'is_generating')
                    ngUI.checkbox('is_interactive_plot').style(checkbox_style).bind_value(self.webUI, 'is_interactive_plot')
                    ngUI.input(label="user_prompt").props(input_props).bind_value(self.webUI, 'user_prompt')
                    ngUI.select({t: t.value for t in RecommendationType}, label='recommendation_type').props(input_props).bind_value(self.webUI, 'recommendation_type')
                    ngUI.number(label="num_images_to_generate", min=0, precision=0, step=1).props(input_props).bind_value(self.webUI, 'num_images_to_generate', forward=int)
                    ngUI.select({m.value: m.value for m in ScoreMode}, label='score_mode').props(input_props).bind_value(self.webUI, 'score_mode')
                    ngUI.number(label="user_profile_host_beta", min=0, precision=0, step=1).props(input_props).bind_value(self.webUI, 'user_profile_host_beta', forward=int)
                    ngUI.input(label="image_display_size").props(input_props).bind_value(self.webUI, 'image_display_size')
                    ngUI.number(label="active_image", min=0, precision=0, step=1).props(input_props).bind_value(self.webUI, 'active_image', forward=int)
                    ngUI.input(label="save_path").props(input_props).bind_value(self.webUI, 'save_path')
                ngUI.space()
                with ngUI.column().classes('p-0 gap-0'):
                    ngUI.label("Debug Generator Info").style("font-size: 50px; color: white;")
                    ngUI.number(label="height", min=0, precision=0, step=8, on_change=self.webUI.generator.load_generator_utils).props(input_props).bind_value(self.webUI.generator, 'height', forward=int)
                    ngUI.number(label="width", min=0, precision=0, step=8, on_change=self.webUI.generator.load_generator_utils).props(input_props).bind_value(self.webUI.generator, 'width', forward=int)
                    ngUI.number(label="batch_size", min=0, precision=0, step=1, on_change=self.webUI.generator.load_generator_utils).props(input_props).bind_value(self.webUI.generator, 'batch_size', forward=int)
                    ngUI.checkbox('random_latents').style(checkbox_style).bind_value(self.webUI.generator, 'random_latents')
                    ngUI.number(label="num_inference_steps", min=0, precision=0, step=1, on_change=self.webUI.generator.load_generator_utils).props(input_props).bind_value(self.webUI.generator, 'num_inference_steps', forward=int)
                    ngUI.number(label="guidance_scale", min=0, step=0.01, on_change=self.webUI.generator.load_generator_utils).props(input_props).bind_value(self.webUI.generator, 'guidance_scale')
                    ngUI.number(label="n_images", min=0, precision=0, step=1, on_change=self.webUI.generator.load_generator_utils).props(input_props).bind_value(self.webUI.generator, 'n_images', forward=int)
                    ngUI.checkbox('use_negative_prompt').style(checkbox_style).bind_value(self.webUI.generator, 'use_negative_prompt')
        self.toggle_visibility()
    
    def toggle_visibility(self):
        self.set_visibility(not self.visible)
