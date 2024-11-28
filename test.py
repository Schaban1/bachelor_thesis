from prototype.user_profile_host import UserProfileHost
import torch

# TODO: This is for debugging, can be removed later.

if __name__ == '__main__':
    # Create a UserProfileHost()
    user_profile_host = UserProfileHost(
        original_prompt='A cute cat',
        add_ons=None,
    )

    # Define some specifications
    num_recommendations = 5
    beta = 20

    # Play through an iteration loop
    embeddings = user_profile_host.generate_recommendations(num_recommendations=num_recommendations, beta=beta)
    preferences = torch.randint(0, 100, size=(num_recommendations,1)) / 10
    for i in range(20):
        # Reduce Beta
        beta -= 1
        # Update the user profile
        user_profile_host.fit_user_profile(preferences=preferences)
        # Generate new Recommendations
        embeddings = user_profile_host.generate_recommendations(num_recommendations=num_recommendations, beta=beta)
        # TODO: This would be where webui and generator generate the images and show them to the user to get preferences.
        preferences = torch.randint(0, 100, size=(num_recommendations,1)) / 10 # placeholder
    print("Test")