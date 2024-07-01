from trajectory import Trajectory
import numpy as np
from numpy.linalg import norm
from itertools import zip_longest


def cosine_similarity_measure(main_trajectory: Trajectory, trajectory_dict: dict, embedding_dict: dict, top_k: int, method: str, pagerank: dict) -> list:
    """Calculate cosine similarity based on the specified method."""
    match method:
        case 'node_embedding':
            return top_k_cosine_similarity(main_trajectory, trajectory_dict, embedding_dict, pagerank, top_k)
        case 'trajectory_embedding':
            return top_k_trajectory_cosine_similarity(main_trajectory, trajectory_dict, embedding_dict, top_k)
        case _:
            raise Exception('Invalid method specified')


def cosine_similarity(vector1: list, vector2: list) -> np.ndarray:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vector1, vector2) / (norm(vector1) * norm(vector2))

######################## DISTANCE SIMILARITY ##########################


def calculate_distance_similarity(trajectory1: Trajectory, trajectory2: Trajectory, distance_dict: dict) -> float:
    """Calculate distance-based similarity between two trajectories."""
    distance_1_to_2 = calculate_node_distance(trajectory1, trajectory2, distance_dict)
    distance_2_to_1 = calculate_node_distance(trajectory2, trajectory1, distance_dict)
    return (distance_1_to_2 + distance_2_to_1) / 2


def top_k_distance_similarity(main_trajectory: Trajectory, trajectory_dict: dict, distance_dict: dict, top_k: int) -> list:
    """Calculate the top k distance-based similarities."""
    similarity_list = []
    for trajectory in trajectory_dict.values():
        if main_trajectory == trajectory:
            continue
        similarity = calculate_distance_similarity(main_trajectory, trajectory, distance_dict)
        similarity_list.append(((main_trajectory.id, trajectory.id), similarity))
    return sorted(similarity_list, key=lambda x: x[1])[:top_k]


def calculate_node_distance(trajectory1: Trajectory, trajectory2: Trajectory, distance_dict: dict) -> float:
    """Calculate average node distance between two trajectories."""
    distances = []
    for node1 in trajectory1.trajectory_path:
        try:
            path_distances = min([distance_dict[node1.node_id][node2.node_id]
                                  for node2 in trajectory2.trajectory_path
                                  if node2.node_id in distance_dict[node1.node_id]])
        except ValueError:
            path_distances = None

        if path_distances is not None:
            distances.append(path_distances)
    if not distances:
        return float('inf')  # Large number if no paths found
    return sum(distances) / len(trajectory1.trajectory_path)

#######################################################################

####################### TSGE1, TSGE2 SIMILARITY #######################


def get_trajectory_vector(trajectory: Trajectory, embedding_dict: dict, pagerank: dict | None) -> list:
    """Convert a trajectory to its vector representation using embeddings."""
    trajectory_vector = []
    for point in trajectory.trajectory_path:
        point_vector = embedding_dict[str(point.node_id)]
        if pagerank is None:
            trajectory_vector = [a + b for a, b in zip_longest(trajectory_vector, point_vector, fillvalue=0)]
        else:
            pagerank_value = pagerank[point.node_id]
            trajectory_vector = [a + b * pagerank_value for a, b in zip_longest(trajectory_vector, point_vector, fillvalue=0)]
    return trajectory_vector


def get_cosine_similarity(trajectory1: Trajectory, trajectory2: Trajectory, embedding_dict: dict, pagerank: dict | None) -> np.ndarray:
    """Calculate cosine similarity between two trajectory vectors."""
    vector1 = get_trajectory_vector(trajectory1, embedding_dict, pagerank)
    vector2 = get_trajectory_vector(trajectory2, embedding_dict, pagerank)
    return cosine_similarity(vector1, vector2)


def top_k_cosine_similarity(main_trajectory: Trajectory, trajectory_dict: dict, embedding_dict: dict, pagerank: dict, top_k: int) -> list:
    """Find the top k most similar trajectories based on cosine similarity."""
    similarity_list = []
    for trajectory in trajectory_dict.values():
        if main_trajectory == trajectory:
            continue
        similarity = get_cosine_similarity(main_trajectory, trajectory, embedding_dict, pagerank)
        similarity_list.append(((main_trajectory.id, trajectory.id), similarity))
    return sorted(similarity_list, key=lambda x: x[1], reverse=True)[:top_k]


#######################################################################

########################## TSGE3 SIMILARITY ###########################

def get_trajectory_cosine_similarity(traj_id_1: str, traj_id_2: str, embedding_dict: dict) -> np.ndarray:
    """Calculate cosine similarity between two trajectories using precomputed embeddings."""
    vector1 = embedding_dict[traj_id_1]
    vector2 = embedding_dict[traj_id_2]
    return cosine_similarity(vector1, vector2)


def top_k_trajectory_cosine_similarity(main_trajectory: Trajectory, trajectory_dict: dict, embedding_dict: dict, top_k: int) -> list:
    """Find the top k most similar trajectories based on precomputed trajectory embeddings."""
    similarity_list = []
    for trajectory in trajectory_dict.values():
        if main_trajectory == trajectory:
            continue
        similarity = get_trajectory_cosine_similarity(str(main_trajectory.id), str(trajectory.id), embedding_dict)
        similarity_list.append(((main_trajectory.id, trajectory.id), similarity))
    return sorted(similarity_list, key=lambda x: x[1], reverse=True)[:top_k]
