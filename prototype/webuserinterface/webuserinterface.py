import os
from PIL import Image
import prototype.utils.constants as constants

class WebUI:
    def __init__(self):
        self.iteration = 0
        self.recommend_by = constants.RANDOM
        self.num_images_to_generate = 5
        self.generator = Generator() # Placeholder
        self.save_path = f"{os.getcwd()}/prototype/output"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def main_loop(self):
        print("Welcome to our image generation system prototype!")
        user_preferences = []
        while True:
            user_prompt = self.get_user_prompt()
            self.generate_images(user_prompt, user_preferences)
            user_preferences = self.select_best_images()

    def get_user_prompt(self):
        print("Enter prompt:")
        user_prompt = input()
        return user_prompt
    
    def generate_images(self, user_prompt, user_preferences):
        # Shouldn't the feedback of the user preferences be processed seperately?
        images = self.generator.generate_images(user_prompt, self.num_images_to_generate, self.recommend_by, user_preferences)
        self.display_images(images)
    
    def display_images(self, images):
        [images[i].save(f"{self.save_path}/image_{i}.png") for i in range(self.num_images_to_generate)]
        print(f"Images saved in {self.save_path}")
    
    def select_best_images(self):
        print("Enter your preferred images by entering the image numbers here seperated by whitespace:")
        user_preferences = [int(x) for x in input().split()]
        print("Thanks! Your selections will be used to generate better fitting images.")
        return user_preferences


# Placeholder
class Generator:
    def generate_images(self, user_prompt, num_images_to_generate, recommend_by, user_preferences):
        return [Image.new('RGB', (64, 64)) for _ in range(num_images_to_generate)]


if __name__ == '__main__':
    ui = WebUI()
    ui.main_loop()
