from nicegui import ui as ngUI
from nicegui import binding
from nicegui.events import KeyEventArguments
from PIL import Image
import torch
import asyncio
import threading
import secrets

from prototype.constants import RecommendationType, WebUIState, ScoreMode
from prototype.user_profile_host import UserProfileHost
from prototype.generator.generator import Generator
from prototype.utils import seed_everything
from prototype.webuserinterface.components import InitialIterationUI, MainLoopUI, LoadingSpinnerUI, PlotUI, Scorer, DebugMenu


class WebUI:
    """
    This class implements a interactive web user interface for an image generation system.
    """
    session_id = binding.BindableProperty()
    state = binding.BindableProperty()
    is_initial_iteration = binding.BindableProperty()
    is_main_loop_iteration = binding.BindableProperty()
    is_generating = binding.BindableProperty()
    is_interactive_plot = binding.BindableProperty()
    user_prompt = binding.BindableProperty()
    recommendation_type = binding.BindableProperty()
    num_images_to_generate = binding.BindableProperty()
    score_mode = binding.BindableProperty()
    image_display_width = binding.BindableProperty()
    image_display_height = binding.BindableProperty()
    active_image = binding.BindableProperty()
    save_path = binding.BindableProperty()
    blind_mode = binding.BindableProperty()

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
        self.recommendation_type = RecommendationType.RANDOM
        self.num_images_to_generate = self.args.num_recommendations
        assert self.num_images_to_generate%2 == 0, "We need an even num images to generate (num_recommendations)!"

        self.score_mode = self.args.score_mode
        self.scorer = Scorer(self)

        # Other modules
        self.user_profile_host = None # Initialized after initial iteration
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.init_generator)

        # Lists / UI components
        self.image_display_width, self.image_display_height = tuple(self.args.image_display_size)
        self.prev_images = []
        self.images = [Image.new('RGB', (self.image_display_width, self.image_display_height)) for _ in range(self.num_images_to_generate)] # For convenience already initialized here
        self.images_display = [None for _ in range(self.num_images_to_generate)] # For convenience already initialized here
        self.active_image = 0
        self.submit_button = None
        # Image saving
        self.save_path = f"{self.args.path.images_save_dir}/{self.session_id}"
        self.num_images_saved = 0

        self.blind_mode = False

        # Set UI root & load debug menu
        self.root = ngUI.column().classes('w-full').style('font-family:"Product Sans","Noto Sans","Verdana", sans-serif')
        self.debug_menu = DebugMenu(self)

        self.keyboard = ngUI.keyboard(on_key=self.handle_key)
        # Remove loading label
        loading_label.delete()
        loading_label = None
        return self

    def run(self):
        """
        This function starts the Web UI.
        """
        self.change_state(WebUIState.INIT_STATE)
        self.root.clear()
        self.build_userinterface()

    def reload_userinterface(self):
        """
        Reloads the UI.
        """
        self.root.clear()
        self.scorer = Scorer(self)
        self.images = self.images[:min(len(self.images), self.num_images_to_generate)] \
                    + [Image.new('RGB', (self.image_display_width, self.image_display_height)) for _ in range(self.num_images_to_generate - min(len(self.images), self.num_images_to_generate))]
        self.images_display = [None for _ in range(self.num_images_to_generate)]
        self.build_userinterface()

    # <---------- Updating State ---------->
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

    # <------------------------------------>
    # <---------- Building UI ---------->
    def build_userinterface(self):
        """
        Builds the complete user interface using NiceGUI.

        UI Structure:
        - Webis demo template top half.
        - Content based on the current state. Either the initial prompt input, the main loop with the user preferences, the loading spinner or the plot.
        - Some empty space so the footer doesnt look weird on high resolution devices.
        - Webis demo template bottom half/footer.
        """
        webis_template_top, webis_template_bottom = self.get_webis_demo_template_html()
        with self.root:
            ngUI.html(webis_template_top).classes('w-full')
            InitialIterationUI(self)
            self.main_loop_ui = MainLoopUI(self)
            LoadingSpinnerUI(self)
            self.plot_ui = PlotUI(self)
            ngUI.space().classes('w-full h-[calc(80vh-2rem)]')
            ngUI.html(webis_template_bottom).classes('w-full')

    # <--------------------------------->
    # <---------- Initialize other non-UI components ---------->
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
        if self.args.generator_warm_start:
            with self.queue_lock:
                self.generator.generate_image(torch.zeros(1, 77, 768))

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

    # <------------------------------------------------------->
    # <---------- Keyboard controls ---------->
    def handle_key(self, e: KeyEventArguments):
        """
        Handles key events.

        Args:
            e: KeyEvent args.
        """
        if e.key.f9 and e.action.keydown:
            self.debug_menu.toggle_visibility()
        if self.score_mode == ScoreMode.EMOJI.value and self.state == WebUIState.MAIN_STATE:
            if e.key.arrow_right and e.action.keydown:
                self.update_active_image(self.active_image + 1)
            if e.key.arrow_left and e.action.keydown:
                self.update_active_image(self.active_image - 1)
            if e.key == 's' and e.action.keydown:
                self.main_loop_ui.on_save_button_click(self.images_display[self.active_image])
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
        self.scorer.scores_toggles[self.active_image].value = key - 1
        self.update_active_image(self.active_image + 1)

    # <--------------------------------------->
    # <---------- Image generation & User profile ---------->
    def generate_images(self):
        """
        Generates images by passing the recommended embeddings from the user profile host to the generator and saving the generated 
        images of the generator in self.images.
        """
        with self.queue_lock:
            embeddings, latents = self.user_profile_host.generate_recommendations(num_recommendations=self.num_images_to_generate)
            self.images = self.generator.generate_image(embeddings, latents)
            self.prev_images.extend(self.images)

    def update_image_displays(self):
        """
        Updates the image displays with the current images in self.images.
        """
        [self.images_display[i].set_source(self.images[i]) for i in range(self.num_images_to_generate)]
        self.reload_userinterface()

    def update_user_profile(self):
        """
        Call the user profile host to update the user profile using provided scores of the current iteration.
        """
        normalized_scores = self.scorer.get_scores()
        self.user_profile_host.fit_user_profile(preferences=normalized_scores)

    # <----------------------------------------------------->
    # <---------- Misc. ---------->
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

    # <--------------------------->
