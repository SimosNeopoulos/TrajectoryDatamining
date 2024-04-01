from trajectory import Trajectory
from numpy import dot
from numpy.linalg import norm
from itertools import zip_longest


def cos_sim_mesure(main_traj: Trajectory, traj_dict: dict, emb_dict: dict, k: int, method: str):
    match method:
        case 'node_emb':
            return k_cos_sim(main_traj, traj_dict, emb_dict, k)
        case 'traj_emb':
            return traj_k_cos_sim(main_traj, traj_dict, emb_dict, k)
        case _:
            raise Exception('Method was not valid')


# def get_traj_vec(trajectory, embedding_dict, node_index):
#     trajectory_vector = []
#     for point in trajectory.trajectory_path:
#         point_vector = embedding_dict[str(node_index[point.node_id])]
#         trajectory_vector = [a + b for a, b in
#                              zip_longest(trajectory_vector, point_vector,
#                                          fillvalue=0)]
#
#     return trajectory_vector


# def get_cos_sim(trajectory1, trajectory2, embedding_dict, node_index):
#     vector1 = get_traj_vec(trajectory1, embedding_dict, node_index)
#     vector2 = get_traj_vec(trajectory2, embedding_dict, node_index)
#
#     return cos_sim(vector1, vector2)


def cos_sim(vector1, vector2):
    return dot(vector1, vector2) / (norm(vector1) * norm(vector2))


# def k_cos_sim(main_traj, trajectories, emb_dict, node_index, k):
#     sim_list = []
#     for trajectory in trajectories.values():
#         if main_traj == trajectory:
#             continue
#         similarity = get_cos_sim(main_traj, trajectory, emb_dict, node_index)
#         sim_list.append(((main_traj.id, trajectory.id), similarity))
#     return sorted(sim_list, key=lambda x: x[1], reverse=True)[:k]


def get_dist_sim(trajectory1, trajectory2, pair_dist):
    distance12 = calc_node_dist(trajectory1, trajectory2, pair_dist)
    distance21 = calc_node_dist(trajectory2, trajectory1, pair_dist)
    return (distance12 + distance21) / 2


def k_dist_sim(main_trajectory, trajectories, pair_dist, k):
    similarity_list = []
    for trajectory in trajectories.values():
        if main_trajectory == trajectory:
            continue
        similarity = get_dist_sim(main_trajectory, trajectory, pair_dist)
        similarity_list.append(((main_trajectory.id, trajectory.id), similarity))
    return sorted(similarity_list, key=lambda x: x[1])[:k]


def calc_node_dist(trajectory1, trajectory2, pair_dist):
    distance_list = []
    for node1 in trajectory1.trajectory_path:
        for node2 in trajectory2.trajectory_path:
            if node1.node_id not in pair_dist or node2.node_id not in pair_dist[node1.node_id]:
                continue
            distance_list.append(min([pair_dist[node1.node_id][node2.node_id]]))
    if len(distance_list) == 0:
        return 100000000
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


############################ BIASED ##################################################################

def get_biased_traj_vec(traj, embedding_dict):
    traj_vec = []
    for point in traj.trajectory_path:
        point_vector = embedding_dict[str(point.node_id)]
        traj_vec = [a + b for a, b in zip_longest(traj_vec, point_vector, fillvalue=0)]

    return traj_vec


def get_cos_sim(trajectory1, trajectory2, embedding_dict):
    vector1 = get_biased_traj_vec(trajectory1, embedding_dict)
    vector2 = get_biased_traj_vec(trajectory2, embedding_dict)

    return cos_sim(vector1, vector2)


def k_cos_sim(main_trajectory, trajectories, embedding_dict, k):
    similarity_list = []
    for trajectory in trajectories.values():
        if main_trajectory == trajectory:
            continue
        similarity = get_cos_sim(main_trajectory, trajectory, embedding_dict)
        similarity_list.append(((main_trajectory.id, trajectory.id), similarity))
    return sorted(similarity_list, key=lambda x: x[1], reverse=True)[:k]


def calc_biased_node_dist(trajectory1, trajectory2, pair_dist):
    distance_list = []
    for node1 in trajectory1.trajectory_path:
        distance_list.append(min([pair_dist[node1.node_id][node2.node_id]
                                  for node2 in trajectory2.trajectory_path]))

    return sum(distance_list) / len(trajectory1.trajectory_path)


def get_biased_dist_sim(trajectory1, trajectory2, pair_dist):
    distance12 = calc_biased_node_dist(trajectory1, trajectory2, pair_dist)
    distance21 = calc_biased_node_dist(trajectory2, trajectory1, pair_dist)
    return (distance12 + distance21) / 2


def biased_k_dist_sim(main_trajectory, trajectories, pair_dist, k):
    similarity_list = []
    for trajectory in trajectories.values():
        if main_trajectory == trajectory:
            continue
        similarity = get_biased_dist_sim(main_trajectory, trajectory, pair_dist)
        similarity_list.append(((main_trajectory.id, trajectory.id), similarity))
    return sorted(similarity_list, key=lambda x: x[1])[:k]


######################################################################################################

################################### Traj2Vec #########################################################

def get_traj_cos_sim(traj_id_1: str, traj_id_2: str, emb_dict: dict):
    vector1 = emb_dict[traj_id_1]
    vector2 = emb_dict[traj_id_2]

    return cos_sim(vector1, vector2)


def traj_k_cos_sim(main_traj, trajectories, emb_dict, k):
    similarity_list = []
    for trajectory in trajectories.values():
        if main_traj == trajectory:
            continue
        similarity = get_traj_cos_sim(str(main_traj.id), str(trajectory.id), emb_dict)
        similarity_list.append(((main_traj.id, trajectory.id), similarity))
    return sorted(similarity_list, key=lambda x: x[1], reverse=True)[:k]
