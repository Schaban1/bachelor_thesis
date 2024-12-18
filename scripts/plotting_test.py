from prototype.user_profile_host import UserProfileHost
from prototype.constants import RecommendationType
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    # Fit TSNE Representation Module from sklearn
    toy_uph = UserProfileHost(original_prompt='test', recommendation_type=RecommendationType.RANDOM)

    # Run a few iterations to gather some points
    for i in range(20):
        embeds, latents = toy_uph.generate_recommendations(num_recommendations=5)
        scores = torch.rand(size=(5,))
        toy_uph.fit_user_profile(preferences=scores)

    low_d_user_profile, low_d_embeddings, preferences = toy_uph.plotting_utils(algorithm='tsne')

    # PLot Embeddings with respective scores
    plt.figure(figsize=(10,10))
    plt.scatter(low_d_embeddings[:, 0], low_d_embeddings[:, 1], alpha=0.7)
    plt.show()