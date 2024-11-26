from abc import abstractmethod, ABC
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from prototype.utils.interpolation import slerp
import tensorflow as tf
import torch
import prototype.utils.constants as constants


# WIP
class Recommender(ABC):  # ABC = Abstract Base Class
    """
    A Recommender class instance derives recommended samples for the next iteration.
    In other words:
    Multiple alterations of the current CLIP embedding (first iteration: text prompt embedding) are returned.
    The method used for generation depends on the user choice.
    It is possible to generate embeddings via the following methods:
    1. Random generation:
        With reference to "Manipulating Embeddings of Stable Diffusion Prompts" (Deckers et al. 2024)
        a subset of multiple random
        embedded prompt (concatenated random alphanumeric characters)
        with maximum pairwise cosine similarity is chosen.
        Afterward, for each random embedding from the subset an
        individual interpolation parameter alpha_i are chosen s.t. the product of the current CLIP embedding
        and a SLERP interpolation of the current CLIP embedding and the random embedding is constant.
        In the end, the interpolations (one per embedding from the subset) are returned as recommendation.
    2. Additional generation:
        A scaled version of the user profile embedding and the current CLIP embedding are summed.
        Afterward, random versions of the new point in the CLIP space are returned as recommendations.
    3. Linear combination generation:
        Linear combinations of user profile embedding and the current CLIP embedding with different weightings
        are returned as recommendations.
    4. Convex Combination generation:
        The user profile consists of 10 weights associated with initial text embeddings.
        The recommendations returned are interpolations of the initial text embeddings.
        (A convex combination is a linear combination of vectors with non-negative weights that sum up to one.)
    """

    @abstractmethod
    def recommend_embeddings(self, user_preferences: list, prompt_embedding: list,
                             user_profile: list, n: int = 5) -> list:
        """
        :param user_preferences: List of length of number of images to generate. Contains ordinal information about the
            user's satisfaction with the images generated in the last iteration. Initially an empty list.
        :param prompt_embedding: Embedding of the current CLIP embedding. Initially the text prompt embedding.
        :param user_profile: Encodes the user profile in the CLIP space. Randomly initialized.
        :param n: Number of recommendations to return. By default, 5.
        :return: A list of recommendations, i.e. n many CLIP embeddings.
        """
        pass


class RandomRecommender(Recommender):

    def get_max_diverse_subset(self, embeddings: list, subset_size: int = 5) -> list:
        """
        :param embeddings: A list of embeddings
        :param subset_size: Number of embeddings to select
        :return: The indices of the subset of embeddings with maximum pairwise cosine dissimilarity
            obtained by a greedy selection
        """
        cos_sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(cos_sim_matrix, 0)  # Ignore self-similarity

        n_vectors = embeddings.shape[0]
        selected_indices = []

        # Greedy selection
        while len(selected_indices) < subset_size:
            if not selected_indices:
                # Start with vector having the highest total similarity
                next_index = np.argmax(cos_sim_matrix.sum(axis=1))
            else:
                # Compute marginal gain for adding each unselected vector
                unselected_indices = list(set(range(n_vectors)) - set(selected_indices))
                gains = [cos_sim_matrix[i, selected_indices].sum() for i in unselected_indices]
                next_index = unselected_indices[np.argmax(gains)]

            selected_indices.append(next_index)

        return selected_indices

    def get_set_of_similar_distance_embeddings(self, original_prompt: list, embeddings: dict,
                                               subset_size: int = 5) -> list:
        """
        :param original_prompt: Original prompt embedding
        :param embeddings: A list of embeddings
        :param subset_size: Number of embeddings to select
        :return: Subset of embeddings which are similar in distance to original embedding (i.e. product of embedding
            with original embedding is similar)
        """
        # greedy selection
        costs = {j: 0 for j in range(len(embeddings[0]))}
        selection = {j: [] for j in range(len(embeddings[0]))}
        for i in range(len(embeddings[0])):
            product = np.dot(embeddings[i], original_prompt)
            for j in list(embeddings.keys())[1:]:   # omit first key, bc it is used to as initial starting point
                # sum of dot products of each embedding with original prompt
                cost_per_embedding = [np.abs((np.dot(embeddings[j][k], original_prompt) - product).sum()) for k in
                                      range(len(embeddings[j]))]
                lowest_cost_idx = np.argmin(cost_per_embedding)
                costs[i] += cost_per_embedding[lowest_cost_idx]
                selection[i].append(embeddings[j][lowest_cost_idx])

        sorted_costs = [(k, v) for k, v in costs.items()]
        sorted_costs.sort(key=lambda s: s[1])
        keys = [i[0] for i in sorted_costs[:min(subset_size, len(sorted_costs))]]
        return [selection[k] for k in keys]


    def recommend_embeddings(self, user_preferences: list, prompt_embedding: list,
                             user_profile, n: int = 5) -> list:

        # cf. "Manipulating Embeddings of Stable Diffusion Prompts", Deckers et al. 2024
        # random embedded prompt: concatenate random alphanumeric characters
        # TODO: ask Generator to embed prompts
        g1 = tf.random.Generator.from_seed(1, alg='philox')
        random_embeddings = g1.normal(shape=[200, len(prompt_embedding)])

        # choose subset of embeddings with maximum pairwise cosine similarity -> diversity
        diverse_subset = [random_embeddings[i] for i in
                          self.get_max_diverse_subset(embeddings=random_embeddings, subset_size=n)]

        if isinstance(prompt_embedding, list):  # necessary during mocking
            prompt_embedding = tf.convert_to_tensor(prompt_embedding)

        # compute recommendations: SLERP(prompt_embedding, random_embedding, alpha_i), different here
        recommendations = {l: slerp(prompt_embedding, diverse_subset[l], n) for l in
                           range(len(diverse_subset))}

        # compute matrix of products of prompt_embedding and recommendations, choose s.t. all products are similar
        subset_recommendations = self.get_set_of_similar_distance_embeddings(original_prompt=prompt_embedding,
                                                                      embeddings=recommendations,
                                                                      subset_size=n)

        return subset_recommendations


class AdditionalRecommender(Recommender):

    def recommend_embeddings(self, user_preferences: list, prompt_embedding: list,
                             user_profile, n: int = 5) -> list:
        beta = 0.5  # Hyperparameter

        additional_embedding = [p + u * beta for p, u in zip(prompt_embedding, user_profile)]

        # use random generator to ensure existing embeddings in the latent space
        random_recommender = RandomRecommender()
        recommendations = random_recommender.recommend_embeddings(user_preferences=user_preferences,
                                                                  user_profile=user_profile,
                                                                  prompt_embedding=additional_embedding, n=n)
        return recommendations


class LinearCombinationRecommender(Recommender):

    def recommend_embeddings(self, user_preferences: list, prompt_embedding: list,
                             user_profile, n: int = 5) -> list:
        recommendations = []
        for alpha in np.linspace(0.1, 1.0, n):
            recommendations.append([alpha * prompt_embedding[j] + (1 - alpha) * user_profile[j] for j in
                                    range(len(prompt_embedding))])  # TODO: type

        return recommendations


class ConvexCombinationRecommender(Recommender):

    def recommend_embeddings(self, user_preferences: list, prompt_embedding: list,
                             user_profile, n: int = 5) -> list:
        recommendations = []
        # TODO: user profile has to contain axes of initial text embeddings in the CLIP space for this approach
        user_profile = np.random.rand(10, len(prompt_embedding))  # TODO: dummy
        for num_embs_to_combine in range(2, n + 1):
            embs_to_combine = user_profile[random.sample(range(len(user_profile)), num_embs_to_combine)]
            # Dirichlet's distribution is a distribution over vectors x that are positive and sum to 1
            weights = np.random.dirichlet(np.ones(num_embs_to_combine), size=1)[0]
            recommendations.append([sum([w * emb for w, emb in zip(weights, embs_to_combine[:, i])]) for i in
                                    range(len(prompt_embedding))])

        return recommendations


if __name__ == '__main__':
    random_recommender = RandomRecommender()
    print("random",
          random_recommender.recommend_embeddings([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]))
    additional_recommender = AdditionalRecommender()
    print("additional",
          additional_recommender.recommend_embeddings( [1, 2, 3, 4, 5], [1, 2, 3, 4, 5],
                                                      [1, 2, 3, 4, 5]))
    linear_combination_recommender = LinearCombinationRecommender()
    print("linear-combi",
          linear_combination_recommender.recommend_embeddings([1, 2, 3, 4, 5],
                                                              [1, 2, 3, 4, 5],
                                                              [1, 2, 3, 4, 5]))
    convex_combination_recommender = ConvexCombinationRecommender()
    print("convex-combi",
          convex_combination_recommender.recommend_embeddings([1, 2, 3, 4, 5],
                                                              [1, 2, 3, 4, 5],
                                                              [1, 2, 3, 4, 5]))
