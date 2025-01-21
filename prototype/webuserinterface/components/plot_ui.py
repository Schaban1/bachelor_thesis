from nicegui import ui as ngUI
import plotly.graph_objects as go

from prototype.webuserinterface.components.ui_component import UIComponent
from prototype.constants import WebUIState, RecommendationType

MAPPING = {0: '😢', 1: '🙁', 2: '😐', 3: '😄', 4: '😍'}

class PlotUI(UIComponent):
    """
    Contains the code for the interactive plot UI.
    """

    def build_userinterface(self):
        """
        Builds the UI for the interactive plot state.
        """
        print('Building interactive plot UI...')
        with ((ngUI.column().classes('mx-auto items-center').bind_visibility_from(self.webUI, 'is_interactive_plot',
                                                                                  value=True))):
            with ngUI.row().classes('w-full justify-start mb-8'):
                ngUI.button('Back', icon='arrow_back', on_click=self.on_back_to_main_loop_button_click).style('font-weight: bold;').props('color=secondary unelevated rounded')
            with ngUI.row().classes('mx-auto items-center'):
                self.fig = go.Figure()
                self.fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), width=1000, height=600)
                self.plot = ngUI.plotly(self.fig)
                self.plot.on('plotly_click', self.on_plot_click)

                self.clicked_image = ngUI.image().style(f'width: {self.webUI.image_display_width}px; height: {self.webUI.image_display_height}px; object-fit: scale-down; border-width: 3px; border-color: lightgray;')

            if self.webUI.user_profile_host is not None and self.webUI.user_profile_host.user_profile is not None:
                self.user_profile, self.embeddings, self.preferences = self.webUI.user_profile_host.plotting_utils()
                self.preferences = self.preferences * 4
                self.preferences = self.preferences.tolist()
                ngUI.separator()
                self.build_image_grid()

    def build_image_grid(self):
        """
        Displays all previous generated images in a wall.
        """
        images = self.webUI.prev_images
        self.preferences.extend([None] * self.webUI.num_images_to_generate)

        ngUI.label('Your generation history:').style('font-size: 150%; font-weight: bold;')
        with ngUI.grid(columns=self.webUI.num_images_to_generate):
            for img, pref in zip(images[::-1], self.preferences[::-1]):
                with ngUI.row().classes('mx-auto items-center'):
                    with ngUI.image(img).style(f'width: {self.webUI.image_display_width}px; height: {self.webUI.image_display_height}px; object-fit: scale-down; border-width: 3px; border-color: lightgray;'):
                        if pref is not None:
                            ngUI.label(MAPPING[pref]).classes('absolute bottom-0 right-0 m-2')

    def create_contour_plot(self, user_profile, embeddings):
        fig = go.Figure(data=go.Contour(x=user_profile[0], y=user_profile[1], z=user_profile[2], opacity=0.4))
        fig.add_trace(go.Scatter(x=embeddings[:, 0], y=embeddings[:, 1], mode='markers', name='embeddings'))
        return fig

    def create_scatter_plot(self, user_profile, embeddings):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=embeddings[:, 0], y=embeddings[:, 1], mode='markers', name='embeddings'))
        if user_profile is not None:
            fig.add_trace(go.Scatter(x=[user_profile[0]], y=[user_profile[1]], mode='markers', marker=dict(size=10, color='red'), name='user profile'))
        return fig

    def update_plot(self):
        """
        Updates the figure with the new embeddings.
        """
        print('Updating plot...')
        self.fig.data = []
        self.clicked_image.set_source(None)

        if self.user_profile is not None and len(self.user_profile) == 3:  # Heatmap for function-based recommender
            fig = self.create_contour_plot(self.user_profile, self.embeddings)
        else:
            fig = self.create_scatter_plot(self.user_profile, self.embeddings)

        self.plot.update_figure(fig)

        self.plot.update()

    def on_plot_click(self, data):
        idx = data.args['points'][0]['pointIndex']
        image = self.webUI.prev_images[idx]
        self.clicked_image.set_source(image)

    def on_back_to_main_loop_button_click(self):
        """
        Returns to the 'Main Loop' screen.
        """
        self.webUI.change_state(WebUIState.MAIN_STATE)
