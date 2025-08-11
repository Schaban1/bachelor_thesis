import asyncio
import random
from nicegui import ui as ngUI

from prototype.constants import RecommendationType, WebUIState
from prototype.webuserinterface.components.ui_component import UIComponent


class InitialIterationUI(UIComponent):
    """
    Contains the code for the initial iteration UI.
    """
    
    def build_userinterface(self):
        """
        Builds the UI for the initial iteration state.
        """
        with ngUI.column().classes('mx-auto items-center').bind_visibility_from(self.webUI, 'is_initial_iteration', value=True):
            ngUI.label('Get inspiration based on an initial prompt.').classes('mt-8').style('font-size: 200%; font-weight: bold;')
            

            with ngUI.card() \
                        .classes('self-center no-box-shadow bg-grey-3 p-0 m-0 mt-4 gap-0') \
                        .style('border-radius: 30px;') \
                        .tight():
                with ngUI.column().classes('items-stretch p-0 gap-0'):
                    self.prompt_field = ngUI.input(placeholder='Type in your prompt') \
                                        .props("size=80 autofocus borderless dense item-aligned") \
                                        .style('font-size: 16px;') \
                                        .bind_value(self.webUI, 'user_prompt') \
                                        .on('keypress.enter', self.on_generate_images_button_click)
                    with self.prompt_field.add_slot("append"):
                        with ngUI.row().classes('p-0 gap-0'):
                            ngUI.button(icon='start', on_click=self.on_generate_images_button_click) \
                                .props('flat fab color=black') \
                                .tooltip('Generate images')
                            #with ngUI.button(icon='more_vert').props('flat fab color=black'):
                            #    with ngUI.menu():
                            #        ngUI.switch("Blind Mode") \
                            #            .classes('ml-2 mr-8') \
                            #            .props('color=grey-8 checked-icon=visibility_off unchecked-icon=visibility') \
                            #            .tooltip('Randomly selects a recommendation type and keeps it hidden') \
                            #            .bind_value(self.webUI, "blind_mode")
                    ngUI.separator().bind_visibility_from(self.webUI, 'blind_mode', value=False)
                    self.recommendation_field = ngUI.select({t: t.value for t in
                                                             [#RecommendationType.BASELINE,
                                                              #RecommendationType.SIMPLE,
                                                              #RecommendationType.RANDOM,
                                                              #RecommendationType.EMA_DIRICHLET,
                                                              #RecommendationType.HYPERSPHERICAL_RANDOM,
                                                              RecommendationType.HYPERSPHERICAL_MOVING_CENTER,
                                                              RecommendationType.HYPERSPHERICAL_BAYESIAN]}) \
                        .props('size=80 borderless dense item-aligned color=secondary popup-content-class="max-w-[200px]"') \
                                            .bind_value(self.webUI, 'recommendation_type') \
                                            .bind_visibility_from(self.webUI, 'blind_mode', value=False)
                    with self.recommendation_field.add_slot("prepend"):
                        ngUI.icon('settings_suggest').classes('mr-2')
            ngUI.space().classes('w-full h-[2vh]')
            self.generate_button = ngUI.button('Generate images', on_click=self.on_generate_images_button_click) \
                                        .style('font-weight: bold;') \
                                        .props('icon-right="start" color=grey-8 unelevated rounded')
    
    async def on_generate_images_button_click(self):
        """
        Initializes the user profile host with the initial user prompt and generates the first images.
        """
        if not self.webUI.user_prompt:
            ngUI.notify('Please type in a prompt!')
            return
        if self.webUI.blind_mode:
            self.setup_blind_mode()
        self.webUI.change_state(WebUIState.GENERATING_STATE)
        self.webUI.iteration += 1
        ngUI.notify('Generating images...')
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.webUI.init_user_profile_host)
        await loop.run_in_executor(None, self.webUI.generate_images)
        self.webUI.update_image_displays()
        self.webUI.change_state(WebUIState.MAIN_STATE)
        self.webUI.debug_menu.set_user_profile_updater()
        self.webUI.main_loop_ui.set_user_profile_host_beta_updater()
        self.webUI.loading_ui.reset_progress_bar()
        self.webUI.update_active_image()
    
    def setup_blind_mode(self):
        """
        Setups blind mode by selecting a random recommender.
        """
        # seed with system time to get different recommender each time for each user
        random.seed()
        self.webUI.recommendation_type = random.choice([t for t in RecommendationType])
