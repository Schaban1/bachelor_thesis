import os
from nicegui import ui as ngUI
from nicegui import binding
from PIL import Image
import torch
import asyncio

from prototype.utils.constants import RecommendationType, WebUIState
from prototype.user_profile_host import UserProfileHost
from prototype.generator.generator import Generator

class WebUI:
    is_initial_iteration = binding.BindableProperty()
    is_main_loop_iteration = binding.BindableProperty()
    is_generating = binding.BindableProperty()

    """
    This class implements a interactive web user interface for an image generation system.
    """
    def __init__(self):
        # State variables
        self.state = WebUIState.INIT_STATE
        self.is_initial_iteration = False
        self.is_main_loop_iteration = False
        self.is_generating = False
        # Other modules
        self.user_profile_host = None # Initialized after initial iteration
        self.user_profile_host_beta = 20
        self.generator = Generator()
        # Provided by the user / system
        self.user_prompt = ""
        self.recommendation_type = RecommendationType.POINT
        self.num_images_to_generate = 5
        # Lists / UI components
        self.images = [Image.new('RGB', (512, 512)) for _ in range(self.num_images_to_generate)] # For convenience already initialized here
        self.images_display = [None for _ in range(self.num_images_to_generate)] # For convenience already initialized here
        self.scores_slider = [None for _ in range(self.num_images_to_generate)] # For convenience already initialized here

        # TODO: Could be used for a save image function
        self.save_path = f"{os.getcwd()}/prototype/output"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self):
        """
        This function runs the Web UI indefenitely.
        """
        self.update_state_variables()
        self.build_userinterface()
        ngUI.run(title='Image Generation System Demo')
    
    def change_state(self, new_state: WebUIState):
        """
        Updates the current state of the Web UI.

        Args:
            new_state: The updated state of the Web UI.
        """
        self.state = new_state
        self.update_state_variables()
    
    def update_state_variables(self):
        """
        Updates the boolean state variables (used for component visibility) based on the current state of the web UI.
        """
        self.is_initial_iteration = self.state == WebUIState.INIT_STATE
        self.is_main_loop_iteration = self.state == WebUIState.MAIN_STATE
        self.is_generating = self.state == WebUIState.GENERATING_STATE
    
    def build_userinterface(self):
        """
        Builds the complete user interface using NiceGUI.

        UI Structure:
        - Webis demo template top half.
        - Content based on the current state. Either the initial prompt input, the main loop with the user preferences or the loading spinner.
        - Some empty space so the footer doesnt look weird on high resolution devices.
        - Webis demo template bottom half/footer.
        """
        webis_template_top, webis_template_bottom = self.get_webis_demo_template_html()
        with ngUI.column().classes('w-full').style('font-family:"Product Sans","Noto Sans","Verdana", sans-serif'):
            ngUI.html(webis_template_top).classes('w-full')
            self.build_initial_userinterface()
            self.build_main_loop_userinterface()
            self.build_loading_spinner_userinterface()
            ngUI.space().classes('w-full h-[calc(80vh-2rem)]')
            ngUI.html(webis_template_bottom).classes('w-full')
    
    def build_initial_userinterface(self):
        """
        Builds the UI for the initial iteration state.
        """
        with ngUI.column().classes('mx-auto items-center').bind_visibility_from(self, 'is_initial_iteration', value=True):
            ngUI.input(label='Your prompt:', on_change=self.on_user_prompt_input).props("size=100")
            ngUI.space().classes('w-full h-[2vh]')
            ngUI.select({t: t.value for t in RecommendationType}, value=RecommendationType.POINT, on_change=self.on_recommendation_type_select).props('popup-content-class="max-w-[200px]"')
            ngUI.space().classes('w-full h-[2vh]')
            ngUI.button('Generate images', on_click=self.on_generate_images_button_click)
    
    def build_main_loop_userinterface(self):
        """
        Builds the UI for the main loop iteration state.
        """
        with ngUI.column().classes('mx-auto items-center').bind_visibility_from(self, 'is_main_loop_iteration', value=True):
            ngUI.label('Please rate these images based on your satisfaction from 0 to 10 using the sliders.').style('font-size: 200%;')
            for i in range(self.num_images_to_generate):
                self.images_display[i] = ngUI.interactive_image(self.images[i]).classes('w-1028')
                self.scores_slider[i] = ngUI.slider(min=0, max=10, value=5, step=0.1)
                ngUI.label().bind_text_from(self.scores_slider[i], 'value')
            ngUI.button('Submit scores', on_click=self.on_submit_scores_button_click)
    
    def build_loading_spinner_userinterface(self):
        """
        Builds the UI for the generating state.
        """
        with ngUI.column().classes('mx-auto items-center').bind_visibility_from(self, 'is_generating', value=True):
            ngUI.label('Generating images...')
            ngUI.spinner(size='lg')
    
    def on_user_prompt_input(self, new_user_prompt):
        """
        Updates the user_prompt class variable on input in the text field.

        Args:
            new_user_prompt: Input of the text field in the initial iteration state.
        """
        self.user_prompt = new_user_prompt.value
    
    def on_recommendation_type_select(self, new_recommendation_type):
        """
        Updates the recommendation_type class variable on selection in the select menu.

        Args:
            new_recommendation_type: Selection of the select menu in the initial iteration state.
        """
        self.recommendation_type = new_recommendation_type.value
    
    def init_user_profile_host(self):
        """
        Initializes the user profile host with the initial user prompt.
        """
        self.user_profile_host = UserProfileHost(
            original_prompt=self.user_prompt,
            add_ons=None,
        )
    
    def generate_images(self):
        """
        Generates images by passing the recommended embeddings from the user profile host to the generator and saving the generated 
        images of the generator in self.images.
        """
        embeddings = self.user_profile_host.generate_recommendations(num_recommendations=self.num_images_to_generate, beta=self.user_profile_host_beta)
        self.images = self.generator.generate_image(embeddings)
    
    def update_image_displays(self):
        """
        Updates the image displays with the current images in self.images.
        """
        [self.images_display[i].set_source(self.images[i]) for i in range(self.num_images_to_generate)]
    
    async def on_generate_images_button_click(self):
        """
        Initializes the user profile host with the initial user prompt and generates the first images.
        """
        self.change_state(WebUIState.GENERATING_STATE)
        ngUI.notify('Generating images...')
        self.init_user_profile_host()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.generate_images)
        self.update_image_displays()
        self.change_state(WebUIState.MAIN_STATE)
    
    def get_scores(self):
        """
        Get the normalized scores provided by the user with the sliders.

        Returns:
            The normalized scores as a one-dim tensor of shape (num_images_to_generate).
        """
        scores = torch.FloatTensor([slider.value for slider in self.scores_slider])
        normalized_scores = scores / 10
        return normalized_scores
    
    def update_user_profile(self):
        """
        Call the user profile host to update the user profile using provided scores of the current iteration.
        """
        normalized_scores = self.get_scores()
        self.user_profile_host.fit_user_profile(preferences=normalized_scores)
        self.user_profile_host_beta -= 1
    
    async def on_submit_scores_button_click(self):
        """
        Updates the user profile with the user scores and generates the next images.
        """
        self.update_user_profile()
        ngUI.notify('Scores submitted!')
        self.change_state(WebUIState.GENERATING_STATE)
        ngUI.notify('Generating new images...')
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.generate_images)
        self.update_image_displays()
        self.change_state(WebUIState.MAIN_STATE)
    
    def get_webis_demo_template_html(self):
        """
        Returns the webis html template for demo web applications.

        Returns:
            A tuple of the top half of the webis html template until the demo content and the bottom half/footer.
        """
        with open("./prototype/resources/webis_template_top.html") as f:
            webis_template_top = f.read()
        with open("./prototype/resources/webis_template_bottom.html") as f:
            webis_template_bottom = f.read()
        return webis_template_top, webis_template_bottom
    

if __name__ in {"__main__", "__mp_main__"}:
    ui = WebUI()
    ui.run()
