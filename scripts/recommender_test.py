import torch
from prototype.user_profile_host.recommender import (SinglePointRecommender, SinglePointWeightedAxesRecommender,
                                                     BayesianRecommender)
from prototype.user_profile_host.utils import display_generated_points


if __name__ == '__main__':
    dummy_user_profile = torch.tensor([1, 2, 3])  # torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # test single point recommender
    single_recommender = SinglePointRecommender()
    single_point_recommendations = single_recommender.recommend_embeddings(user_profile=dummy_user_profile,
                                                                           n_recommendations=200)
    if dummy_user_profile.shape[0] == 3:
        display_generated_points(single_point_recommendations, user_profile=dummy_user_profile)
    print("single point", single_point_recommendations)

    # test single point + weighted axes recommender
    single_weighted_recommender = SinglePointWeightedAxesRecommender()
    weighted_axes_recommendations = single_weighted_recommender.recommend_embeddings(user_profile=dummy_user_profile,
                                                                                     n_recommendations=100)
    if dummy_user_profile.shape[0] == 3:
        display_generated_points(weighted_axes_recommendations, user_profile=dummy_user_profile)
    print("single point + weighted axes", weighted_axes_recommendations)

    # TODO: test function-based recommender (Bayesian approach)
    # function_based_recommender = BayesianRecommender(n_steps=5, n_axis=3)
    # function_based_recommendations = function_based_recommender.recommend_embeddings(user_profile=dummy_user_profile,
    #                                                                                  n_recommendations=100)
    # print("function_based_recommender", function_based_recommendations)