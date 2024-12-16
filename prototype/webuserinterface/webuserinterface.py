import os
from nicegui import ui as ngUI
from nicegui import binding
from nicegui.events import KeyEventArguments
from PIL import Image
import torch
from functools import partial
import asyncio
import threading
import secrets

from prototype.constants import RecommendationType, WebUIState
from prototype.user_profile_host import UserProfileHost
from prototype.generator.generator import Generator


class WebUI:
    """
    This class implements a interactive web user interface for an image generation system.
    """
    is_initial_iteration = binding.BindableProperty()
    is_main_loop_iteration = binding.BindableProperty()
    is_generating = binding.BindableProperty()
    user_prompt = binding.BindableProperty()
    recommendation_type = binding.BindableProperty()

    @classmethod
    async def create(cls, args):
        """
        This method should be used instead of the __init__-method to create an object of the WebUI-class.
        Usage: ui = await WebUI.create(...) inside an async function.

        Args:
            args: The config args as an omegaconf.DictConfig object.

        Returns:
            Created object of type WebUI.
        """
        self = cls()
        self.session_id = secrets.token_urlsafe(4)
        # Args of global config
        self.args = args
        # State variables
        self.state = None
        self.is_initial_iteration = False
        self.is_main_loop_iteration = False
        self.is_generating = False
        # Provided by the user / system
        self.user_prompt = ""
        self.recommendation_type = RecommendationType.POINT
        self.num_images_to_generate = self.args.num_recommendations
        # Other modules
        self.user_profile_host = None # Initialized after initial iteration
        self.user_profile_host_beta = 20
        self.generator = Generator(
            n_images=self.num_images_to_generate,
            cache_dir=self.args.path.cache_dir,
            device=self.args.device,
            **self.args.generator        
        )
        # Lists / UI components
        self.image_display_size = (256, 256)
        self.images = [Image.new('RGB', self.image_display_size) for _ in range(self.num_images_to_generate)] # For convenience already initialized here
        self.images_display = [None for _ in range(self.num_images_to_generate)] # For convenience already initialized here
        self.scores_toggles = [None for _ in range(self.num_images_to_generate)] # For convenience already initialized here
        self.active_image = 0
        self.scores_slider = [None for _ in range(self.num_images_to_generate)] # For convenience already initialized here
        self.submit_button = None
        # Image saving
        self.save_path = f"{self.args.path.images_save_dir}/{self.session_id}"
        self.num_images_saved = 0

        self.queue_lock = threading.Lock()
        self.keyboard = None
        return self

    def run(self):
        """
        This function runs the Web UI indefinitely.
        """
        self.change_state(WebUIState.INIT_STATE)
        self.build_userinterface()
    
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
        self.keyboard = ngUI.keyboard(on_key=self.handle_key, active=False)
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
            ngUI.input(label='Your prompt:', on_change=self.on_user_prompt_input, validation={'Please type in a prompt!': lambda value: len(value) > 0}).props("size=100")
            ngUI.space().classes('w-full h-[2vh]')
            ngUI.select({t: t.value for t in RecommendationType}, value=RecommendationType.POINT, on_change=self.on_recommendation_type_select).props('popup-content-class="max-w-[200px]"')
            ngUI.space().classes('w-full h-[2vh]')
            ngUI.button('Generate images', on_click=self.on_generate_images_button_click)
    
    def build_main_loop_userinterface(self):
        """
        Builds the UI for the main loop iteration state.
        """
        ngUI.html('<style>.multi-line-notification { white-space: pre-line; }</style>')
        with ngUI.column().classes('mx-auto items-center').bind_visibility_from(self, 'is_main_loop_iteration', value=True):
            with ngUI.row().classes('mx-auto items-center'):
                ngUI.label('Please rate these images based on your satisfaction.').style('font-size: 200%;')
                ngUI.button(icon='o_info', on_click=lambda: ngUI.notify(
                    'Keyboard Controls:\n'
                    'Left/Right arrow: Navigate through images\n'
                    '1-5: Score current image\n'
                    's: Save current image\n'
                    'Enter: Submit scores',
                    multi_line=True,
                    classes='multi-line-notification'
                )).props('flat fab color=black')
            with ngUI.row().classes('mx-auto items-center'):
                ngUI.label(f'Your selected recommendation type:').style('font-size: 150%; font-weight: bold;')
                ngUI.label(self.recommendation_type).style('font-size: 150%;').bind_text_from(self, 'recommendation_type')
            ngUI.label(f'Your initial prompt:').style('font-size: 150%; font-weight: bold;')
            ngUI.label(self.user_prompt).style('font-size: 150%;').bind_text_from(self, 'user_prompt')
            with ngUI.row().classes('mx-auto items-center'):
                for i in range(self.num_images_to_generate):
                    with ngUI.column().classes('mx-auto items-center'):
                        self.images_display[i] = ngUI.interactive_image(self.images[i]).style(f'width: {self.image_display_size[0]}px; height: {self.image_display_size[1]}px; object-fit: scale-down')
                        with self.images_display[i]:
                            ngUI.button(icon='o_save', on_click=partial(self.on_save_button_click, self.images_display[i])).props('flat fab color=white').classes('absolute bottom-0 right-0 m-2')
                        self.scores_toggles[i] = ngUI.toggle({0: '😢1', 1: '🙁2', 2: '😐3', 3: '😄4', 4: '😍5'}, value=0).props('rounded')
                        #self.scores_slider[i] = ngUI.slider(min=0, max=10, value=0, step=0.1)
                        #ngUI.label().bind_text_from(self.scores_slider[i], 'value')
            ngUI.space()
            self.submit_button = ngUI.button('Submit scores', on_click=self.on_submit_scores_button_click)
            with ngUI.row().classes('w-full justify-end'):
                ngUI.button('Restart process', on_click=self.on_restart_process_button_click, color='red')
    
    def build_loading_spinner_userinterface(self):
        """
        Builds the UI for the generating state.
        """
        with ngUI.column().classes('mx-auto items-center').bind_visibility_from(self, 'is_generating', value=True):
            ngUI.label('Generating images...').style('font-size: 200%;')
            ngUI.spinner(size='128px')
    
    def handle_key(self, e: KeyEventArguments):
        """
        Handles key events.

        Args:
            e: KeyEvent args.
        """
        if e.key.arrow_right and e.action.keydown:
            self.update_active_image(self.active_image + 1)
        if e.key.arrow_left and e.action.keydown:
            self.update_active_image(self.active_image - 1)
        if e.key in ['1', '2', '3', '4', '5'] and e.action.keydown:
            self.on_number_keystroke(e.key.number)
        if e.key == 's' and e.action.keydown:
            self.on_save_button_click(self.images_display[self.active_image])
        if e.key.enter and e.action.keydown:
            self.submit_button.run_method('click')
    
    def update_active_image(self, idx=0):
        """
        Updates the active image and its visuals on the UI.

        Args:
            idx: The image index of the new active image.
        """
        idx = idx % self.num_images_to_generate
        self.images_display[self.active_image].style('border-color: transparent')
        self.active_image = idx
        self.images_display[idx].style('border-width: 4px; border-color: red')
    
    def on_number_keystroke(self, key):
        """
        Updates the score for the active image upon typing one of the valid number keys.

        Args:
            key: The number of the key typed.
        """
        self.scores_toggles[self.active_image].value = key - 1
        self.update_active_image(self.active_image + 1)
    
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
            recommendation_type=self.recommendation_type,
            cache_dir=self.args.path.cache_dir,
            stable_dif_pipe=self.generator.pipe,
            **self.args.recommender
        )
    
    def generate_images(self):
        """
        Generates images by passing the recommended embeddings from the user profile host to the generator and saving the generated 
        images of the generator in self.images.
        """
        with self.queue_lock:
            embeddings, latents = self.user_profile_host.generate_recommendations(num_recommendations=self.num_images_to_generate, beta=self.user_profile_host_beta)
            self.images = self.generator.generate_image(embeddings, latents)
    
    def update_image_displays(self):
        """
        Updates the image displays with the current images in self.images.
        """
        [self.images_display[i].set_source(self.images[i]) for i in range(self.num_images_to_generate)]
    
    async def on_generate_images_button_click(self):
        """
        Initializes the user profile host with the initial user prompt and generates the first images.
        """
        if not self.user_prompt:
            ngUI.notify('Please type in a prompt!')
            return
        self.change_state(WebUIState.GENERATING_STATE)
        ngUI.notify('Generating images...')
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.init_user_profile_host)
        await loop.run_in_executor(None, self.generate_images)
        self.update_image_displays()
        self.change_state(WebUIState.MAIN_STATE)
        self.update_active_image()
        self.keyboard.active = True
    
    def get_scores_slider(self):
        """
        Get the normalized scores provided by the user with the sliders.

        Returns:
            The normalized scores as a one-dim tensor of shape (num_images_to_generate).
        """
        scores = torch.FloatTensor([slider.value for slider in self.scores_slider])
        normalized_scores = scores / 10
        return normalized_scores
    
    def get_scores_toggles(self):
        """
        Get the normalized scores provided by the user with the emoji toggle buttons.

        Returns:
            The normalized scores as a one-dim tensor of shape (num_images_to_generate).
        """
        scores = torch.FloatTensor([toggle.value for toggle in self.scores_toggles])
        normalized_scores = scores / 4
        return normalized_scores
    
    def reset_sliders(self):
        """
        Reset the value of the score sliders to the default value.
        """
        [slider.set_value(0) for slider in self.scores_slider]
    
    def reset_toggles(self):
        """
        Reset the value of the score toggles to the default value.
        """
        [toggle.set_value(0) for toggle in self.scores_toggles]
    
    def update_user_profile(self):
        """
        Call the user profile host to update the user profile using provided scores of the current iteration.
        """
        normalized_scores = self.get_scores_toggles()
        self.user_profile_host.fit_user_profile(preferences=normalized_scores)
        self.user_profile_host_beta -= 1
    
    def on_save_button_click(self, image_display):
        """
        Saves the displayed image where the save button is located in the images save dir.

        Args:
            image_display: The image display containing the image to save.
        """
        image_to_save = image_display.source
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        file_name = f"image_{self.num_images_saved}.png"
        image_to_save.save(f"{self.save_path}/{file_name}")
        self.num_images_saved += 1
        ngUI.notify(f"Image saved in {self.save_path}/{file_name}!")
    
    async def on_submit_scores_button_click(self):
        """
        Updates the user profile with the user scores and generates the next images.
        """
        self.update_user_profile()
        ngUI.notify('Scores submitted!')
        self.change_state(WebUIState.GENERATING_STATE)
        self.keyboard.active = False
        ngUI.notify('Generating new images...')
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.generate_images)
        self.update_image_displays()
        self.reset_toggles()
        self.change_state(WebUIState.MAIN_STATE)
        self.update_active_image()
        self.keyboard.active = True
    
    def on_restart_process_button_click(self):
        """
        Restarts the process by starting with the initial iteration again.
        """
        self.change_state(WebUIState.INIT_STATE)
        self.keyboard.active = False
        self.reset_toggles()
        self.user_profile_host = None
    
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
