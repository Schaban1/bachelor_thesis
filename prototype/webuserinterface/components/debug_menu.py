from nicegui import ui as ngUI


class DebugMenu(ngUI.element):
    """
    Contains the code for the debug menu.
    """

    def __init__(self, bg_color=(0,0,0,0.5)):
        super().__init__(tag='div')
        self.style('position: fixed; display: block; width: 100%; height: 100%;'
                   'top: 0; left: 0; right: 0; bottom: 0; z-index: 2; cursor: pointer;'
                   'background-color:' + f'rgba{bg_color};')
        with self:
            with ngUI.element('div').classes("h-screen flex items-center justify-center"):
                ngUI.label("Test").style("font-size: 50px; color: white;")
        self.toggle_visibility()
    
    def toggle_visibility(self):
        self.set_visibility(not self.visible)
