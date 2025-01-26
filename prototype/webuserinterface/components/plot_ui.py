from nicegui import ui as ngUI
import plotly.graph_objects as go
import torch

from prototype.webuserinterface.components.ui_component import UIComponent
from prototype.constants import WebUIState, RecommendationType

MAPPING = {0: '😢', 1: '🙁', 2: '😐', 3: '😄', 4: '😍'}


class PlotUI(UIComponent):
    """
    Contains the code for the interactive plot UI.
    """

    def __init__(self, webUI):
        super().__init__(webUI)

    def build_userinterface(self):
        """
        Builds the UI for the interactive plot state.
        """
        self.plot_ui = ngUI.column().classes('mx-auto items-center').bind_visibility_from(self.webUI, 'is_interactive_plot', value=True)
        self.unlabeled_images = []

        with self.plot_ui:
            with ngUI.row().classes('w-full justify-start mb-8'):
                ngUI.button('Back', icon='arrow_back', on_click=self.on_back_to_main_loop_button_click).style(
                    'font-weight: bold;').props('color=secondary unelevated rounded')

            switch = ngUI.switch('Show user profile history', on_change=self.on_switch_change).style('font-size: 150%;')

            with ngUI.row().classes('mx-auto items-center'):
                self.fig = go.Figure()
                self.fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), width=1000, height=600)
                self.plot = ngUI.plotly(self.fig)
                self.plot.on('plotly_click', self.on_plot_click)

                self.clicked_image = ngUI.interactive_image().style(
                    f'width: {self.webUI.image_display_width}px; height: {self.webUI.image_display_height}px; object-fit: scale-down; border-width: 3px; border-color: lightgray;')

            ngUI.separator()
            with ngUI.expansion('Your generation history', icon='history').classes('w-full').style('font-size: 150%; font-weight: bold;'):
                self.image_grid = ngUI.grid(columns=5)

    def on_switch_change(self, event):
        print(event.value)

    def build_image_grid(self, preferences):
        """
        Displays all previous generated images in a wall.
        """
        images = self.webUI.generator.get_latest_images()
        preferences.extend([None] * self.webUI.num_images_to_generate)  # Latest images not rated yet
        preferences = preferences[-(len(self.unlabeled_images) + len(images)):]

        for idx, display in enumerate(self.unlabeled_images):
            with display:
                if preferences[idx] is not None:
                    ngUI.label(MAPPING[preferences[idx]]).classes('absolute bottom-0 right-0 m-2')

        n = len(self.unlabeled_images)
        self.unlabeled_images = []

        with self.image_grid:
            for img, pref in zip(images, preferences[n:]):
                display = ngUI.image(img).style(
                        f'width: {self.webUI.image_display_width}px; height: {self.webUI.image_display_height}px; object-fit: scale-down; border-width: 3px; border-color: lightgray;')
                with display:
                    if pref is not None:
                        ngUI.label(MAPPING[pref]).classes('absolute bottom-0 right-0 m-2')
                    else:
                        self.unlabeled_images.append(display)

                display.move(target_index=0)

    def create_contour_plot(self, user_profile, embeddings):
        fig = go.Figure(data=go.Contour(x=user_profile[0], y=user_profile[1], z=user_profile[2], opacity=0.5))
        fig.add_trace(go.Scatter(x=embeddings[:, 0], y=embeddings[:, 1], mode='markers', name='embeddings'))
        return fig

    def create_scatter_plot(self, user_profile, embeddings):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=embeddings[:, 0], y=embeddings[:, 1], mode='markers', name='embeddings'))
        if user_profile is not None:
            fig.add_trace(
                go.Scatter(x=[user_profile[0]], y=[user_profile[1]], mode='markers', marker=dict(size=10, color='red'),
                           name='user profile'))
        return fig

    def update_plot(self):
        """
        Updates the figure with the new embeddings.
        """
        if self.webUI.user_profile_host is None:
            return

        user_profile, embeddings, _ = self.webUI.user_profile_host.plotting_utils()

        if user_profile is not None and len(user_profile) == 3:  # Heatmap for function-based recommender
            fig = self.create_contour_plot(user_profile, embeddings)
        else:
            fig = self.create_scatter_plot(user_profile, embeddings)

        self.plot.update_figure(fig)
        self.plot.update()

    def update_view(self):
        """
        Updates the view with the new embeddings.
        """
        self.update_plot()
        preferences = self.webUI.user_profile_host.preferences
        preferences = (preferences * 4).tolist()

        with self.plot_ui:
            self.build_image_grid(preferences)

    def clear_view(self):
        """
        Clears the view (for restarting generation process).
        """
        self.unlabeled_images = []
        self.image_grid.clear()

    def on_plot_click(self, data):
        idx = data.args['points'][0]['pointIndex']
        self.clicked_image.set_source(self.image_grid.default_slot.children[::-1][idx].source)

    def on_back_to_main_loop_button_click(self):
        """
        Returns to the 'Main Loop' screen.
        """
        self.webUI.change_state(WebUIState.MAIN_STATE)
