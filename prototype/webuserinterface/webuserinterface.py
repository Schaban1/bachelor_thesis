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

from prototype.constants import RecommendationType, WebUIState, ScoreMode
from prototype.user_profile_host import UserProfileHost
from prototype.generator.generator import Generator
from prototype.utils import seed_everything
from prototype.webuserinterface.components.initial_iteration_ui import InitialIterationUI
from prototype.webuserinterface.components.main_loop_ui import MainLoopUI
from prototype.webuserinterface.components.loading_spinner_ui import LoadingSpinnerUI
from prototype.webuserinterface.components.plot_ui import PlotUI


class WebUI:
    """
    This class implements a interactive web user interface for an image generation system.
    """
    is_initial_iteration = binding.BindableProperty()
    is_main_loop_iteration = binding.BindableProperty()
    is_generating = binding.BindableProperty()
    is_interactive_plot = binding.BindableProperty()
    user_prompt = binding.BindableProperty()
    recommendation_type = binding.BindableProperty()
    user_profile_host_beta = binding.BindableProperty()

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
        loading_label = ngUI.label("Starting session...")
        await ngUI.context.client.connected()
        # Args of global config
        self.args = args
        seed_everything(self.args.random_seed)
        self.queue_lock = threading.Lock()
        # Generate id for this session
        self.session_id = secrets.token_urlsafe(4)
        # State variables
        self.state = None
        self.is_initial_iteration = False
        self.is_main_loop_iteration = False
        self.is_generating = False
        self.is_interactive_plot = False
        # Provided by the user / system
        self.user_prompt = ""
        self.recommendation_type = RecommendationType.POINT
        self.num_images_to_generate = self.args.num_recommendations
        self.score_mode = self.args.score_mode
        self.init_score_mode()

        # Other modules
        self.user_profile_host = None # Initialized after initial iteration
        self.user_profile_host_beta = self.args.user_profile_host.beta
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.init_generator)

        # Lists / UI components
        self.image_display_size = tuple(self.args.image_display_size)
        self.images = [Image.new('RGB', self.image_display_size) for _ in range(self.num_images_to_generate)] # For convenience already initialized here
        self.images_display = [None for _ in range(self.num_images_to_generate)] # For convenience already initialized here
        self.scores_toggles = [None for _ in range(self.num_images_to_generate)] # For convenience already initialized here
        self.active_image = 0
        self.scores_slider = [None for _ in range(self.num_images_to_generate)] # For convenience already initialized here
        self.submit_button = None
        # Image saving
        self.save_path = f"{self.args.path.images_save_dir}/{self.session_id}"
        self.num_images_saved = 0

        self.keyboard = None
        # Remove loading label
        loading_label.delete()
        loading_label = None
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
        self.is_interactive_plot = self.state == WebUIState.PLOT_STATE
    
    def init_score_mode(self):
        """
        Registers some functions based on the current self.score_mode.
        """
        if self.score_mode == ScoreMode.SLIDER.value:
            self.build_scorer = self.build_slider
            self.get_scores = self.get_scores_slider
            self.reset_scorers = self.reset_sliders
        elif self.score_mode == ScoreMode.EMOJI.value:
            self.build_scorer = self.build_emoji_toggle
            self.get_scores = self.get_scores_emoji_toggles
            self.reset_scorers = self.reset_emoji_toggles
        else:
            print(f"Unknown score mode: {self.score_mode}")
    
    def init_generator(self):
        """
        Initializes the generator and performs a warm-start.
        """
        self.generator = Generator(
            n_images=self.num_images_to_generate,
            cache_dir=self.args.path.cache_dir,
            device=self.args.device,
            **self.args.generator        
        )
        with self.queue_lock:
            self.generator.generate_image(torch.zeros(1, 77, 768))
    
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
            InitialIterationUI(self)
            MainLoopUI(self)
            LoadingSpinnerUI(self)
            PlotUI(self)
            ngUI.space().classes('w-full h-[calc(80vh-2rem)]')
            ngUI.html(webis_template_bottom).classes('w-full')
    
    def build_slider(self, idx):
        """
        Registers a slider object at position idx.

        Args:
            idx: The index of the slider.
        """
        self.scores_slider[idx] = ngUI.slider(min=0, max=10, value=0, step=0.1)
        ngUI.label().bind_text_from(self.scores_slider[idx], 'value')
    
    def build_emoji_toggle(self, idx):
        """
        Registers a toggle object at position idx.

        Args:
            idx: The index of the toggle object.
        """
        self.scores_toggles[idx] = ngUI.toggle({0: '😢1', 1: '🙁2', 2: '😐3', 3: '😄4', 4: '😍5'}, value=0).props('rounded')
    
    def handle_key(self, e: KeyEventArguments):
        """
        Handles key events.

        Args:
            e: KeyEvent args.
        """
        if self.score_mode == ScoreMode.EMOJI.value:
            if e.key.arrow_right and e.action.keydown:
                self.update_active_image(self.active_image + 1)
            if e.key.arrow_left and e.action.keydown:
                self.update_active_image(self.active_image - 1)
            if e.key == 's' and e.action.keydown:
                self.on_save_button_click(self.images_display[self.active_image])
            if e.key.enter and e.action.keydown:
                self.submit_button.run_method('click')
            if e.key.number in [1, 2, 3, 4, 5] and e.action.keydown:
                self.on_number_keystroke(e.key.number)
    
    def update_active_image(self, idx=0):
        """
        Updates the active image and its visuals on the UI (currently only used in emoji ScoreMode).

        Args:
            idx: The image index of the new active image.
        """
        if self.score_mode == ScoreMode.EMOJI.value:
            idx = idx % self.num_images_to_generate
            self.images_display[self.active_image].style('border-color: lightgray')
            self.active_image = idx
            self.images_display[idx].style('border-color: red')
    
    def on_number_keystroke(self, key):
        """
        Updates the score for the active image upon typing one of the valid number keys.

        Args:
            key: The number of the key typed.
        """
        self.scores_toggles[self.active_image].value = key - 1
        self.update_active_image(self.active_image + 1)
    
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
            n_recommendations=self.num_images_to_generate,
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
    
    def get_scores_slider(self):
        """
        Get the normalized scores provided by the user with the sliders.

        Returns:
            The normalized scores as a one-dim tensor of shape (num_images_to_generate).
        """
        scores = torch.FloatTensor([slider.value for slider in self.scores_slider])
        normalized_scores = scores / 10
        return normalized_scores
    
    def get_scores_emoji_toggles(self):
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
    
    def reset_emoji_toggles(self):
        """
        Reset the value of the score toggles to the default value.
        """
        [toggle.set_value(0) for toggle in self.scores_toggles]
    
    def update_user_profile(self):
        """
        Call the user profile host to update the user profile using provided scores of the current iteration.
        """
        normalized_scores = self.get_scores()
        self.user_profile_host.fit_user_profile(preferences=normalized_scores)
        self.user_profile_host_beta -= 1
    
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
