from nicegui import ui as ngUI
import asyncio

from prototype.webuserinterface.components.ui_component import UIComponent
from prototype.constants import RecommendationType, WebUIState


class InitialIterationUI(UIComponent):
    
    def build_userinterface(self):
        """
        Builds the UI for the initial iteration state.
        """
        with ngUI.column().classes('mx-auto items-center').bind_visibility_from(self.webUI, 'is_initial_iteration', value=True):
            ngUI.input(label='Your prompt:', on_change=self.on_user_prompt_input, validation={'Please type in a prompt!': lambda value: len(value) > 0}).props("size=100")
            ngUI.space().classes('w-full h-[2vh]')
            ngUI.select({t: t.value for t in RecommendationType}, value=RecommendationType.POINT, on_change=self.on_recommendation_type_select).props('popup-content-class="max-w-[200px]"')
            ngUI.space().classes('w-full h-[2vh]')
            ngUI.button('Generate images', on_click=self.on_generate_images_button_click)
    
    def on_user_prompt_input(self, new_user_prompt):
        """
        Updates the user_prompt class variable on input in the text field.

        Args:
            new_user_prompt: Input of the text field in the initial iteration state.
        """
        self.webUI.user_prompt = new_user_prompt.value
    
    def on_recommendation_type_select(self, new_recommendation_type):
        """
        Updates the recommendation_type class variable on selection in the select menu.

        Args:
            new_recommendation_type: Selection of the select menu in the initial iteration state.
        """
        self.webUI.recommendation_type = new_recommendation_type.value
    
    async def on_generate_images_button_click(self):
        """
        Initializes the user profile host with the initial user prompt and generates the first images.
        """
        if not self.webUI.user_prompt:
            ngUI.notify('Please type in a prompt!')
            return
        self.webUI.change_state(WebUIState.GENERATING_STATE)
        ngUI.notify('Generating images...')
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.webUI.init_user_profile_host)
        await loop.run_in_executor(None, self.webUI.generate_images)
        self.webUI.update_image_displays()
        self.webUI.change_state(WebUIState.MAIN_STATE)
        self.webUI.update_active_image()
        self.webUI.keyboard.active = True
