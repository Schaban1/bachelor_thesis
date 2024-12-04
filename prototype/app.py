from nicegui import ui as ngUI

from prototype.webuserinterface import WebUI

global_args = None

@ngUI.page('/demo')
def start_demo_instance():
    global global_args
    ui = WebUI(global_args)
    ui.run()

@ngUI.page('/')
def start():
    ngUI.navigate.to('/demo')

class App:
    def __init__(self, args):
        global global_args
        global_args = args
    
    def start(self):
        global global_args
        ngUI.run(title='Image Generation System Demo', port=global_args.port)
        start()
    