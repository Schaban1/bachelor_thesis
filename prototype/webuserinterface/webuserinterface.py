import os
from nicegui import ui as ngUI
from PIL import Image
from prototype.utils.constants import Constants

class WebUI:
    """
    This class implements a interactive web user interface for an image generation system.
    """
    def __init__(self):
        self.iteration = 1
        self.recommendation_type = Constants.POINT
        self.num_images_to_generate = 5
        self.user_prompt = ""
        self.generator = Generator() # Placeholder
        self.save_path = f"{os.getcwd()}/prototype/output"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def main_loop(self):
        """
        The main loop of the application.
        This method should be run when wanting to start the application.
        The main loop runs infinitely and contains the input of a textual prompt by the user, 
        the displaying of the generated images and the selection of preferreble images by the user.
        """
        self.build_userinterface()
        """while True:
            user_prompt = self.get_user_prompt()
            images = self.generate_images(user_prompt, user_preferences)
            self.display_images(images)
            user_preferences = self.select_best_images()
            self.iteration += 1"""
    
    def build_userinterface(self):
        webis_template_top, webis_template_bottom = self.get_webis_demo_template_html()
        with ngUI.column().classes('w-full'):
            ngUI.html(webis_template_top).classes('w-full')
            with ngUI.column().classes('mx-auto items-center'):
                ngUI.input(label='Your prompt:', on_change=self.on_user_prompt_input).props("size=100")
                ngUI.space().classes('w-full h-[2vh]')
                ngUI.select({t: t.value for t in Constants}, value=Constants.POINT, on_change=self.on_recommendation_type_select).props('popup-content-class="max-w-[200px]"')
                ngUI.space().classes('w-full h-[2vh]')
                ngUI.button('Generate images', on_click=self.on_generate_images_button_click)
            ngUI.space().classes('w-full h-[calc(80vh-2rem)]')
            ngUI.html(webis_template_bottom).classes('w-full')
        ngUI.run(title='Image Generation System Demo')
    
    def on_user_prompt_input(self, new_user_prompt):
        self.user_prompt = new_user_prompt
    
    def on_recommendation_type_select(self, new_recommendation_type):
        self.recommendation_type = new_recommendation_type
    
    def on_generate_images_button_click(self):
        ngUI.notify('Generating images...')

    
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

    def get_user_prompt(self):
        """
        Asks the user for a text prompt and returns their input.

        Returns:
            User prompt as a string.
        """
        print("Enter prompt:")
        user_prompt = input()
        return user_prompt
    
    def generate_images(self, user_prompt, user_preferences):
        """
        Generate images using the Generator with the provided user_prompt.

        Args:
            user_prompt: The user prompt as a string.
            user_preferences: A list of indices containing the preferred generated images of the previous iteration.
        
        Returns:
            A list of the generated images.
        """
        # Shouldn't the feedback of the user preferences be processed seperately?
        images = self.generator.generate_images(user_prompt, self.num_images_to_generate, self.recommendation_type, user_preferences)
        return images
    
    def display_images(self, images):
        """
        Display the provided images.
        Currently, the images will be saved into an output-folder that the user can access.

        Args:
            images: The images that should be displayed.
        """
        [images[i].save(f"{self.save_path}/image_{i}.png") for i in range(self.num_images_to_generate)]
        print(f"Images saved in {self.save_path}")
    
    def select_best_images(self):
        """
        Lets the user select their preferred images by inputting the corresponding indices.

        Returns:
            A list of indices containing the preferred generated images.
        """
        print("Enter your preferred images by entering the image numbers here seperated by whitespace:")
        user_preferences = [int(x) for x in input().split()]
        print("Thanks! Your selections will be used to generate better fitting images.")
        return user_preferences


# Placeholder
class Generator:
    def generate_images(self, user_prompt, num_images_to_generate, recommend_by, user_preferences):
        return [Image.new('RGB', (64, 64)) for _ in range(num_images_to_generate)]

if __name__ in {"__main__", "__mp_main__"}:
    ui = WebUI()
    ui.main_loop()
