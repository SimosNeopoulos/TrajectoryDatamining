from trajectory import Trajectory
from point import Point
import networkx as nx
import json
import struct
import zipfile as zp

# Information on how to read a line from the trajectory file
NEW_TRAJECTORY_VALUE = '0'
END_OF_TRAJECTORY = '2'
NEW_TRAJECTORY_VALUE_LOCATION = 0
TRAJECTORY_ID_LOCATION = 1
OBJECT_CLASS_ID_LOCATION = 2
TIME_LOCATION = 3
CURRENT_X_LOCATION = 4
CURRENT_Y_LOCATION = 5
CURRENT_SPEED_LOCATION = 6
NEXT_X_LOCATION = 7
NEXT_Y_LOCATION = 8
CURRENT_NODE_ID = 9
NEXT_NODE_ID = 10


# old code, before started using brute force with generator #####################

def read_node(in_file):
    try:
        name_len = struct.unpack('b', in_file.read(1))[0]
        name = in_file.read(name_len).decode('utf-8')
        _id = struct.unpack('q', in_file.read(8))[0]
        x = struct.unpack('i', in_file.read(4))[0]
        y = struct.unpack('i', in_file.read(4))[0]
        return name, _id, x, y
    except (IOError, struct.error):
        return None


def read_edge(in_file):
    try:
        node_id1, node_id2, name_len = struct.unpack('qqb', in_file.read(17))
        name = in_file.read(name_len).decode()
        _id, edge_class = struct.unpack('qi', in_file.read(12))
        return node_id1, node_id2, name, _id, edge_class
    except (IOError, struct.error):
        return None


def get_node_data(zip_node_file, node_file):
    node_file_data = []
    with zp.ZipFile(zip_node_file, 'r') as zip_f:
        with zip_f.open(node_file, 'r') as file:
            while True:
                node_data = read_node(file)

                if node_data is None:
                    break

                node_file_data.append(node_data)

    return node_file_data


def get_edge_data(zip_edge_file, edge_file):
    edge_file_data = []
    with zp.ZipFile(zip_edge_file, 'r') as zip_f:
        with zip_f.open(edge_file, 'r') as file:
            while True:
                edge_data = read_edge(file)

                if edge_data is None:
                    break

                edge_file_data.append(edge_data)

    return edge_file_data


def create_graph_from_files(zip_node_file, node_file, zip_edge_file, edge_file):
    graph = nx.DiGraph()
    node_data = get_node_data(zip_node_file, node_file)
    edge_data = get_edge_data(zip_edge_file, edge_file)

    for node in node_data:
        name, node_id, x, y = node
        graph.add_node(node_id, x=x, y=y, name=name)

    for edge in edge_data:
        node_id1, node_id2, name, edge_id, edge_class = edge
        graph.add_edge(node_id1, node_id2)

    return graph


########################################################################


def create_graph_and_index(file_path):
    graph = nx.DiGraph()
    node_index = {}
    node_set, edge_set = get_sets(file_path)

    for index, node_info in enumerate(node_set):
        node_id, x, y = node_info
        graph.add_node(index, x=x, y=y)
        node_index[node_id] = index

    for edge_info in edge_set:
        node_id1, node_id2 = edge_info
        graph.add_edge(node_index[node_id1], node_index[node_id2])

    return graph, node_index


def save_graph_data(graph, node_index, json_path):
    dict_to_store = {
        'graph': nx.to_dict_of_dicts(graph),
        'index': node_index
    }
    with open(json_path, 'w') as file:
        json.dump(dict_to_store, file)


def load_graph_data(json_path):
    with open(json_path, 'r') as file:
        graph_data = json.load(file)

    graph = nx.DiGraph(graph_data['graph'])
    node_index = graph_data['index']

    return graph, node_index


def get_sets(file_path):
    node_set = set()
    edge_set = set()
    with open(file_path, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break

            line_values = line.replace('\n', '').split('\t')

            if line_values[NEW_TRAJECTORY_VALUE_LOCATION] == END_OF_TRAJECTORY:
                continue

            x = line_values[CURRENT_X_LOCATION]
            y = line_values[CURRENT_Y_LOCATION]
            next_node_x = line_values[NEXT_X_LOCATION]
            next_node_y = line_values[NEXT_Y_LOCATION]
            node_id = line_values[CURRENT_NODE_ID]
            next_node_id = line_values[NEXT_NODE_ID]

            node_set.add((node_id, x, y))
            node_set.add((next_node_id, next_node_x, next_node_y))

            edge_set.add((node_id, next_node_id))

    return node_set, edge_set


# Read the id and the object id from the given trajectory
# Create the trajectory from those values and return it
def get_trajectory_from_line(line_values):
    object_id = line_values[TRAJECTORY_ID_LOCATION]
    object_class_id = line_values[OBJECT_CLASS_ID_LOCATION]
    return Trajectory(object_id, object_class_id)


# Read the values for the point from the line_values list
# Create and return a Point object
def get_point_from_line(line_values):
    time = line_values[TIME_LOCATION]
    x = line_values[CURRENT_X_LOCATION]
    y = line_values[CURRENT_Y_LOCATION]
    speed = line_values[CURRENT_SPEED_LOCATION]
    next_node_x = line_values[NEXT_X_LOCATION]
    next_node_y = line_values[NEXT_Y_LOCATION]
    node_id = line_values[CURRENT_NODE_ID]
    next_node_id = line_values[NEXT_NODE_ID]

    return Point(time, x, y, node_id, speed, next_node_x, next_node_y,
                 next_node_id)


def get_trajectory_catalog(path_to_trajectory):
    trajectory_catalog = {}

    with open(path_to_trajectory, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break

            line_values = line.replace('\n', '').split('\t')
            current_trajectory = get_trajectory_from_line(line_values)

            if line_values[NEW_TRAJECTORY_VALUE_LOCATION] == END_OF_TRAJECTORY:
                continue

            # If the trajectory is being created add it to the catalog
            if line_values[
                NEW_TRAJECTORY_VALUE_LOCATION] == NEW_TRAJECTORY_VALUE:
                trajectory_catalog[current_trajectory.id] = current_trajectory

            # Add a point to an existing trajectory
            trajectory_catalog[current_trajectory.id] \
                .add_point(
                get_point_from_line(line_values)
            )
    return trajectory_catalog


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
        dict_trajectory = trajectories_to_compare[key]
        if dict_trajectory == trajectory:
            continue
        weighted_list.append(
            get_jaccard_similarity(trajectory, dict_trajectory))

    return sorted(weighted_list, key=lambda x: x[0], reverse=True)[:k]
