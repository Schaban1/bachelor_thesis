import torch
from prototype.user_profile_host.recommender import (SinglePointRecommender, SinglePointWeightedAxesRecommender,
                                                     BayesianRecommender)
from prototype.user_profile_host.utils import display_generated_points


if __name__ == '__main__':
    dummy_user_profile = torch.tensor([0.1, 0.2, 0.3])  # torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # test single point recommender
    # single_recommender = SinglePointRecommender()
    # single_point_recommendations = single_recommender.recommend_embeddings(user_profile=dummy_user_profile,
    #                                                                        n_recommendations=200)
    # if dummy_user_profile.shape[0] == 3:
    #     display_generated_points(single_point_recommendations, user_profile=dummy_user_profile)
    # print("single point", single_point_recommendations)

    # test single point + weighted axes recommender
    latent_bounds = (0., 1.)
    embedding_bounds = (0., 1.)
    single_weighted_recommender = SinglePointWeightedAxesRecommender(n_latent_axis=1, n_embedding_axis=2,
                                                                     latent_bounds=latent_bounds,
                                                                     embedding_bounds=embedding_bounds,
                                                                     exploration_factor=0.5)
    weighted_axes_recommendations = single_weighted_recommender.recommend_embeddings(user_profile=dummy_user_profile,
                                                                                     n_recommendations=100)
    if dummy_user_profile.shape[0] == 3:
        display_generated_points(weighted_axes_recommendations, user_profile=dummy_user_profile,
                                 x_bounds=embedding_bounds, y_bounds=embedding_bounds, z_bounds=embedding_bounds)
    print("single point + weighted axes", weighted_axes_recommendations)

    # TODO: test function-based recommender (Bayesian approach)
    # function_based_recommender = BayesianRecommender(n_steps=5, n_axis=3)
    # function_based_recommendations = function_based_recommender.recommend_embeddings(user_profile=dummy_user_profile,
    #                                                                                  n_recommendations=100)
    # print("function_based_recommender", function_based_recommendations)