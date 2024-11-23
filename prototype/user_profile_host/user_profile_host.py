import torch
import torch.nn as nn

class UserProfileHost():

    def __init__(self, base_vectors):
        self.user_profile = None
        self.user_preferences = None
        self.embeddings = None
        self.base_vecotrs = base_vectors

    def transform(self):
        return None

    def inv_transform(self):
        return None

    def optimize_user_profile(self):
        return None
    
    def fit_user_profile(self):
        return None

    def generate_recommendations(self):
        return None