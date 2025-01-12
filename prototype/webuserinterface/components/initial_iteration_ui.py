from nicegui import ui as ngUI
import asyncio

from prototype.webuserinterface.components.ui_component import UIComponent
from prototype.constants import RecommendationType, WebUIState


class InitialIterationUI(UIComponent):
    """
    Contains the code for the initial iteration UI.
    """
    
    def build_userinterface(self):
        """
        Builds the UI for the initial iteration state.
        """
        with ngUI.column().classes('mx-auto items-center').bind_visibility_from(self.webUI, 'is_initial_iteration', value=True):
            ngUI.input(label='Your prompt:', validation={'Please type in a prompt!': lambda value: len(value) > 0}).props("size=100").bind_value(self.webUI, 'user_prompt')
            ngUI.space().classes('w-full h-[2vh]')
            ngUI.select({t: t.value for t in RecommendationType}, value=RecommendationType.POINT).props('popup-content-class="max-w-[200px]"').bind_value(self.webUI, 'recommendation_type')
            ngUI.space().classes('w-full h-[2vh]')
            ngUI.button('Generate images', on_click=self.on_generate_images_button_click)
    
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
        self.webUI.debug_menu.set_user_profile_updater()
        self.webUI.update_active_image()
