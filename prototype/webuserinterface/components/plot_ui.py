from nicegui import ui as ngUI
import plotly.graph_objects as go

from prototype.webuserinterface.components.ui_component import UIComponent
from prototype.constants import WebUIState


class PlotUI(UIComponent):
    """
    Contains the code for the interactive plot UI.
    """

    def build_userinterface(self):
        """
        Builds the UI for the interactive plot state.
        """
        with ngUI.column().classes('items-center w-full').bind_visibility_from(self.webUI, 'is_interactive_plot', value=True):
            ngUI.button('Back', on_click=self.on_back_to_main_loop_button_click)
            self.fig = go.Figure()
            self.fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            self.fig.update_layout(title_text='User Profile and Embeddings')
            self.plot = ngUI.plotly(self.fig)
            self.plot.update()


    def update_plot(self):
        """
        Updates the figure with the new embeddings.
        """
        self.fig.data = []
        user_profile, embeddings, _ = self.webUI.user_profile_host.plotting_utils()
        self.fig.add_trace(go.Scatter(x=embeddings[:, 0], y=embeddings[:, 1], mode='markers'))
        self.fig.add_trace(go.Scatter(x=[user_profile[0]], y=[user_profile[1]], mode='markers', marker=dict(size=10, color='red')))
        self.plot.update()


    def on_back_to_main_loop_button_click(self):
        """
        Returns to the 'Main Loop' screen.
        """
        self.webUI.change_state(WebUIState.MAIN_STATE)
        self.webUI.keyboard.active = True
