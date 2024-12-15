from nicegui import ui as ngUI

from prototype.webuserinterface import WebUI

global_args = None

@ngUI.page('/demo')
async def start_demo_instance():
    """
    Creates a new instance of the WebUI and runs it.
    This instance is private with the user and not shared.
    """
    global global_args
    ui = await WebUI.create(global_args)
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
    
    def start(self):
        """
        Start the application.
        """
        global global_args
        ngUI.run(title='Image Generation System Demo', port=global_args.port)
        start()
    