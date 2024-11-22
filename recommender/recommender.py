from abc import abstractmethod, ABC
import random
import numpy as np

# WIP

class Recommender(ABC):
    # ABC = Abstract Base Class

    @abstractmethod
    def recommend_embeddings(self, recommend_by: str, user_preferences: list, prompt_embedding: list,
                             user_profile: list, n: int = 5) -> list:
        pass


class RandomRecommender(Recommender):

    def recommend_embeddings(self, recommend_by: str, user_preferences: list, prompt_embedding: list,
                             user_profile: list, n: int = 5) -> list:

        # cf. "Manipulating Embeddings of Stable Diffusion Prompts", Deckers et al. 2024
        # random embedded prompt: concatenate random alphanumeric characters
        random_embeddings = []  # TODO: ask Generator to embed prompts

        # choose subset of embeddings with maximum pairwise cosine similarity -> diversity
        diverse_subset = [] # TODO: compute pairwise cosine similarities & select subset

        # choose individual interpolation parameters alpha_i s.t.
        # prompt_embedding * SLERP(prompt_embedding, random_embedding, alpha_i) is constant
        # TODO

        # compute recommendations: SLERP(prompt_embedding, random_embedding, alpha_i)
        # TODO


        recommendations = []
        for i in range(n):
            gaussian_noise = [random.gauss(mu=0.0, sigma=1.0) for _ in range(len(prompt_embedding))]
            recommendations.append(prompt_embedding + gaussian_noise)

        return recommendations  # TODO: Ensure recommendations exist (cf. paper Deckers et al. 2024)


class AdditionalRecommender(Recommender):

    def recommend_embeddings(self, recommend_by: str, user_preferences: list, prompt_embedding: list,
                             user_profile: list, n: int = 5) -> list:
        beta = 0.5 # Hyperparameter
        recommendations = []
        additional_embedding = prompt_embedding + [u * beta for u in user_profile]
        for i in range(n):
            gaussian_noise = [additional_embedding[j] + random.gauss(mu=0.0, sigma=1.0) for j in range(len(prompt_embedding))]
            recommendations.append(prompt_embedding + gaussian_noise)

        return recommendations   # TODO: Implement this method


class LinearCombinationRecommender(Recommender):

    def recommend_embeddings(self, recommend_by: str, user_preferences: list, prompt_embedding: list,
                             user_profile: list, n: int = 5) -> list:
        recommendations = []
        for alpha in np.linspace(0.1, 1.0, n):
            recommendations.append([alpha * prompt_embedding[j] + (1 - alpha) * user_profile[j]for j in
                                    range(len(prompt_embedding))])  # TODO: type

        return recommendations


if __name__ == '__main__':
    random_recommender = RandomRecommender()
    print(random_recommender.recommend_embeddings('random', [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]))
    additional_recommender = AdditionalRecommender()
    print(additional_recommender.recommend_embeddings('additional', [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]))
    linear_combination_recommender = LinearCombinationRecommender()
    print(linear_combination_recommender.recommend_embeddings('linear_combination', [1, 2, 3, 4, 5], [1, 2, 3, 4, 5],
                                                              [1, 2, 3, 4, 5]))