import numpy as np
from scipy.spatial.distance import pdist, squareform


def sample_dpp(similarity_matrix, k):
    L = np.linalg.cholesky(similarity_matrix)
    idx = np.arange(0, L.shape[0])

    # Initialize empty set and probabilities
    selected_indices = []
    # Loop for k iterations
    for _ in range(k):
        # Compute probabilities based on the diagonal of LL^T
        probabilities = np.array(np.diag(L @ L.T))
        probabilities[0] = 0.0

        # Normalize probabilities
        probabilities /= np.sum(probabilities)

        # Sample an item
        selected_idx = np.random.choice(np.arange(0, L.shape[0]), p=probabilities)
        selected_item = idx[selected_idx]

        # Add the selected item to the set
        selected_indices.append(selected_item)

        # Update Cholesky decomposition
        L = np.delete(L, selected_idx, axis=0)
        L = np.delete(L, selected_idx, axis=1)
        idx = np.delete(idx, selected_idx, axis=0)

    return np.array(selected_indices)


def dpp(sorted_scores, sorted_location, sorted_idx, num_results) -> np.ndarray:
    similarity_matrix = np.exp(-squareform(pdist(sorted_location)) / 5000)
    relevance_matrix = sorted_scores.reshape(-1, 1) * sorted_scores.reshape(1, -1)
    dpp_matrix = relevance_matrix * similarity_matrix
    index = sample_dpp(dpp_matrix, num_results)
    return sorted_idx[index]


def identify_diverse_results(
        embeds: np.ndarray,
        ids: np.ndarray,
        location: np.ndarray,
        qid: int,
        num_results: int = 5,
        expansion_factor: float = 10.0
) -> np.ndarray:
    k_max = int(expansion_factor * num_results)
    idx = np.where(ids == qid)[0]
    query = embeds[idx]
    scores = (query @ embeds.T).squeeze()
    # ignore the query
    order = np.argsort(-scores)[:(k_max + 1)]
    return dpp(np.sqrt(np.clip(scores[order], 0, 1)), location[order], ids[order], num_results)

