from nicegui import ui as ngUI
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL
import threading

from prototype.webuserinterface import WebUI

import collections, time

class QLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.waiters = collections.deque()
        self.count = 0

    def acquire(self):
        self.lock.acquire()
        if self.count:
            new_lock = threading.Lock()
            new_lock.acquire()
            self.waiters.append(new_lock)
            self.lock.release()
            new_lock.acquire()
            self.lock.acquire()
        self.count += 1
        self.lock.release()

    def release(self):
        with self.lock:
            if not self.count:
                raise ValueError("lock not acquired")
            self.count -= 1
            if self.waiters:
                self.waiters.popleft().release()
        time.sleep(0.1)

    def locked(self):
        return self.count > 0
    
    def __enter__(self):
        self.acquire()
    
    def __exit__(self, type, val, traceback):
        self.release()

global_args = None
pipe = None
queue_lock = QLock()

@ngUI.page('/demo')
async def start_demo_instance():
    """
    Creates a new instance of the WebUI and runs it.
    This instance is private with the user and not shared.
    """
    global global_args
    global pipe
    ui = await WebUI.create(global_args, pipe, queue_lock)
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
        self.device = torch.device("cuda") if (global_args.device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")

        global pipe
        pipe = StableDiffusionPipeline.from_pretrained(
            global_args.hf_model_name,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=global_args.path.cache_dir,
            torch_dtype=torch.bfloat16,
        ).to(device=self.device)

        pipe.unet = torch.compile(pipe.unet, backend="cudagraphs")

        pipe.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device=pipe.device, dtype=pipe.dtype)
        pipe.vae = torch.compile(pipe.vae, backend="cudagraphs")
    
    def start(self):
        """
        Start the application.
        """
        global global_args
        ngUI.run(title='Image Generation System Demo', port=global_args.port, reconnect_timeout=global_args.reconnect_timeout)
        start()
    