import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL
from nicegui import ui as ngUI

from prototype.utils import ProducerConsumer
from prototype.webuserinterface import WebUI
from prototype.generator.generator import Generator

global_args = None
pipe = None
queue_lock = ProducerConsumer()  # QLock()


@ngUI.page('/demo')
async def start_demo_instance():
    """
    Creates a new instance of the WebUI and runs it.
    This instance is private with the user and not shared.
    """
    global global_args
    global pipe
    global generator
    ui = await WebUI.create(global_args, pipe, generator, queue_lock)
    ui.run()


@ngUI.page('/')
def start():
    """
    Just redirects to '/demo', because '/' is the auto-index page.
    """
    ngUI.navigate.to('/demo')


class App:
    """
    The entry point into the application.
    """

    def __init__(self, args):
        global global_args
        global_args = args
        self.device = torch.device("cuda") if (
                global_args.device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")

        # Initialize a central StableDiffusionPipeline for all sessions
        global pipe
        pipe = StableDiffusionPipeline.from_pretrained(
            global_args.hf_model_name,
            #safety_checker=None,
            requires_safety_checker=True,
            cache_dir=global_args.path.cache_dir,
            torch_dtype=torch.bfloat16,
        ).to(device=self.device)

        pipe.unet = torch.compile(pipe.unet, backend="cudagraphs")
        pipe.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device=pipe.device, dtype=pipe.dtype)
        pipe.vae = torch.compile(pipe.vae, backend="cudagraphs")

        global generator
        generator = Generator(
                cache_dir=args.path.cache_dir,
                device=args.device,
                hf_model_name=args.hf_model_name,
                pipe=pipe,
                **args.generator,
            )

    def start(self):
        """
        Start the application.
        """
        global global_args
        ngUI.run(title='Image Generation System Demo', port=global_args.port,
                 reconnect_timeout=global_args.reconnect_timeout)
        start()
