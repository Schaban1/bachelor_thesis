from abc import abstractmethod, ABC
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#from splines.quaternion import slerp as SLERP
import tensorflow as tf
import torch
import constants

# WIP

class Recommender(ABC):
    # ABC = Abstract Base Class

    @abstractmethod
    def recommend_embeddings(self, recommend_by: str, user_preferences: list, prompt_embedding: list,
                             user_profile: list, n: int = 5) -> list:
        pass


class RandomRecommender(Recommender):

    def get_max_diverse_subset(self, embeddings: list, subset_size: int = 5) -> list:

        # Step 1: Compute the cosine similarity matrix
        cos_sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(cos_sim_matrix, 0)  # Ignore self-similarity

        # Step 2: Initialize
        n_vectors = embeddings.shape[0]
        selected_indices = []

        # Step 3: Greedy selection
        while len(selected_indices) < subset_size:
            if not selected_indices:
                # Start with the vector having the highest total similarity
                next_index = np.argmax(cos_sim_matrix.sum(axis=1))
            else:
                # Compute the marginal gain for adding each unselected vector
                unselected_indices = list(set(range(n_vectors)) - set(selected_indices))
                gains = [
                    cos_sim_matrix[i, selected_indices].sum()
                    for i in unselected_indices
                ]
                next_index = unselected_indices[np.argmax(gains)]

            selected_indices.append(next_index)

        return selected_indices

    def slerp(self, v0, v1, num, t0=0, t1=1):
        """Spherical linear interpolation between two vectors.
        :param v0: start vector
        :param v1: end vector
        :param num: number of interpolation steps
        :param t0: start interpolation value
        :param t1: end interpolation value
        :return: interpolated vectors
        """
        v0 = v0.numpy()#v0.detach().cpu().numpy()
        v1 = v1.numpy()#v1.detach().cpu().numpy()

        def interpolation(t, v0, v1, DOT_THRESHOLD=0.9995):
            """helper function to spherically interpolate two arrays v1 v2"""
            dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
            if np.abs(dot) > DOT_THRESHOLD:
                v2 = (1 - t) * v0 + t * v1
            else:
                theta_0 = np.arccos(dot)
                sin_theta_0 = np.sin(theta_0)
                theta_t = theta_0 * t
                sin_theta_t = np.sin(theta_t)
                s0 = np.sin(theta_0 - theta_t) / sin_theta_0
                s1 = sin_theta_t / sin_theta_0
                v2 = s0 * v0 + s1 * v1
            return v2

        t = np.linspace(t0, t1, num)

        v3 = torch.tensor(np.array([interpolation(t[i], v0, v1) for i in range(num)]))

        return v3

    def recommend_embeddings(self, recommend_by: str, user_preferences: list, prompt_embedding: list,
                             user_profile: list, n: int = 5) -> list:

        # cf. "Manipulating Embeddings of Stable Diffusion Prompts", Deckers et al. 2024
        # random embedded prompt: concatenate random alphanumeric characters
        g1 = tf.random.Generator.from_seed(1, alg='philox')
        random_embeddings = g1.normal(shape=[200, len(prompt_embedding)])
        #np.random.rand(200, len(prompt_embedding))  # TODO: ask Generator to embed prompts


        # choose subset of embeddings with maximum pairwise cosine similarity -> diversity
        diverse_subset = [random_embeddings[i] for i in
                          self.get_max_diverse_subset(embeddings=random_embeddings, subset_size=n)]

        # choose individual interpolation parameters alpha_i s.t.
        # prompt_embedding * SLERP(prompt_embedding, random_embedding, alpha_i) is constant
        # TODO: How to choose alpha_i effectively? cf. below: alpha corresponds to indices which correspond to similar products
        alphas = np.linspace(0.1, 1.0, 10)
        #print("alphas", alphas)

        if type(prompt_embedding) == list:  # necessary during mocking
            prompt_embedding = tf.convert_to_tensor(prompt_embedding)

        # compute recommendations: SLERP(prompt_embedding, random_embedding, alpha_i), different here
        recommendations = {l: self.slerp(prompt_embedding, diverse_subset[l], n) for l in
                           range(len(diverse_subset))}

        # TODO: compute matrix of products of prompt_embedding and recommendations, choose s.t. all products are similar
        # product_matr = np.array([tf.math.multiply(prompt_embedding, recommendations[random_embedding]) for random_embedding in diverse_subset])
        recommendations = [recommendations[l][2] for l in range(len(diverse_subset))] # TODO: change to best choice

        # recommendations = []
        # for i in range(n):
        #     gaussian_noise = [random.gauss(mu=0.0, sigma=1.0) for _ in range(len(prompt_embedding))]
        #     recommendations.append(prompt_embedding + gaussian_noise)

        return recommendations


class AdditionalRecommender(Recommender):

    def recommend_embeddings(self, recommend_by: str, user_preferences: list, prompt_embedding: list,
                             user_profile: list, n: int = 5) -> list:
        beta = 0.5 # Hyperparameter
        recommendations = []
        additional_embedding = [p + u * beta for p, u in zip(prompt_embedding, user_profile)]
        for i in range(n):
            gaussian_noise = [additional_embedding[j] + random.gauss(mu=0.0, sigma=1.0) for j in range(len(prompt_embedding))]
            recommendations.append([p + g for p, g in zip(prompt_embedding, gaussian_noise)])

        return recommendations   # TODO: Implement this method


class LinearCombinationRecommender(Recommender):

    def recommend_embeddings(self, recommend_by: str, user_preferences: list, prompt_embedding: list,
                             user_profile: list, n: int = 5) -> list:
        recommendations = []
        for alpha in np.linspace(0.1, 1.0, n):
            recommendations.append([alpha * prompt_embedding[j] + (1 - alpha) * user_profile[j] for j in
                                    range(len(prompt_embedding))])  # TODO: type

        return recommendations


if __name__ == '__main__':
    random_recommender = RandomRecommender()
    print("random", random_recommender.recommend_embeddings('random', [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]))
    additional_recommender = AdditionalRecommender()
    print("additional", additional_recommender.recommend_embeddings('additional', [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]))
    linear_combination_recommender = LinearCombinationRecommender()
    print("linear-combi", linear_combination_recommender.recommend_embeddings('linear_combination', [1, 2, 3, 4, 5], [1, 2, 3, 4, 5],
                                                              [1, 2, 3, 4, 5]))