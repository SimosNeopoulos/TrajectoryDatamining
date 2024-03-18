from trajectory import Trajectory
from numpy import dot
from numpy.linalg import norm
from itertools import zip_longest


def get_trajectory_vector(trajectory, embedding_dict, node_index):
    trajectory_vector = []
    for point in trajectory.trajectory_path:
        point_vector = embedding_dict[str(node_index[point.node_id])]
        trajectory_vector = [a + b for a, b in
                             zip_longest(trajectory_vector, point_vector,
                                         fillvalue=0)]

    return trajectory_vector


def get_cos_similarity(trajectory1, trajectory2, embedding_dict, node_index):
    vector1 = get_trajectory_vector(trajectory1, embedding_dict, node_index)
    vector2 = get_trajectory_vector(trajectory2, embedding_dict, node_index)

    return cosine_similarity(vector1, vector2)


def cosine_similarity(vector1, vector2):
    return dot(vector1, vector2) / (norm(vector1) * norm(vector2))


def k_cosine_similarity(main_trajectory, trajectories, embedding_dict, node_index, k):
    similarity_list = []
    for trajectory in trajectories.values():
        similarity = get_cos_similarity(main_trajectory, trajectory, embedding_dict, node_index)
        similarity_list.append(((main_trajectory.id, trajectory.id), similarity))
    return sorted(similarity_list, key=lambda x: x[1], reverse=True)[:k]


def get_distance_similarity(trajectory1, trajectory2, node_dict, pair_dist):
    distance12 = calc_node_dist(trajectory1, trajectory2, node_dict, pair_dist)
    distance21 = calc_node_dist(trajectory2, trajectory1, node_dict, pair_dist)
    return (distance12 + distance21) / 2


def k_distance_similarity(main_trajectory, trajectories, node_dict, pair_dist,
                          k):
    similarity_list = []
    for trajectory in trajectories.values():
        similarity = get_distance_similarity(main_trajectory, trajectory, node_dict, pair_dist)
        similarity_list.append(((main_trajectory.id, trajectory.id), similarity))
    return sorted(similarity_list, key=lambda x: x[1])[:k]


def calc_node_dist(trajectory1, trajectory2, node_dict, pair_dist):
    distance_list = []
    for node1 in trajectory1.trajectory_path:
        distance_list.append(min([pair_dist[node_dict[node1.node_id]][node_dict[node2.node_id]]
                                  for node2 in trajectory2.trajectory_path]))

    return sum(distance_list) / len(trajectory1.trajectory_path)


def get_jaccard_similarity(trajectory1, trajectory2):
    edge_list1 = trajectory1.get_edge_list()
    edge_list2 = trajectory2.get_edge_list()
    intersection = len(list(set(edge_list1).intersection(edge_list2)))
    union = (len(edge_list1) + len(edge_list2)) - intersection
    similarity = float(intersection) / union
    return similarity, trajectory2.id


def k_similar_trajectories(trajectory: Trajectory,
                           trajectories_to_compare: dict, k: int):
    weighted_list = []

    for key in trajectories_to_compare.keys():
        trajectory2 = trajectories_to_compare[key]
        if trajectory2 == trajectory:
            continue
        weighted_list.append(
            get_jaccard_similarity(trajectory, trajectory2))

    return sorted(weighted_list, key=lambda x: x[0], reverse=True)[:k]
