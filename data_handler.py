from trajectory import Trajectory
from karateclub.node_embedding.neighbourhood.node2vec import Node2Vec
from timeit import default_timer as timer
from point import Point
import pickle
import networkx as nx
import json
import struct
import zipfile as zp

# Information on how to read a line from the trajectory file
NEW_TRAJ_VALUE = '0'
END_OF_TRAJ = '2'
NEW_TRAJ_VALUE_LOC = 0
TRAJ_ID_LOC = 1
OBJECT_CLASS_ID_LOC = 2
TIME_LOC = 3
CURRENT_X_LOC = 4
CURRENT_Y_LOC = 5
CURRENT_SPEED_LOC = 6
NEXT_X_LOC = 7
NEXT_Y_LOC = 8
CURRENT_NODE_ID = 9
NEXT_NODE_ID = 10


# old code, before started using brute force with generator ###################

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


def print_to_file(text, file_path=f'./data_files/similarity_data.txt'):
    with open(file_path, 'a') as file:
        file.write(text + '\n')


def get_time_from_secs(secs):
    hrs = secs // 3600
    mins = (secs % 3600) // 60
    rem_secs = secs % 60

    return int(hrs), int(mins), rem_secs


def create_graph_and_index(file_path):
    start = timer()

    graph = nx.DiGraph()
    node_index = {}
    node_set, edge_set = get_sets(file_path)

    for index, node_info in enumerate(node_set):
        node_id, x, y = node_info
        graph.add_node(index, x=x, y=y)
        node_index[int(node_id)] = index

    for edge_info in edge_set:
        node_id1, node_id2 = edge_info
        graph.add_edge(node_index[int(node_id1)], node_index[int(node_id2)])

    end = timer()

    print(graph.__str__())
    print_to_file(graph.__str__() + '\n')

    hrs, mins, secs = get_time_from_secs(end - start)
    text_to_print = f'Time to create graph: {hrs}hr {mins}min {secs:.5f}sec\n'
    print(text_to_print)
    print_to_file(text_to_print)

    start = timer()
    node2vec = Node2Vec()
    node2vec.fit(graph)
    vec_graph = node2vec.get_embedding()
    end = timer()

    hrs, mins, secs = get_time_from_secs(end - start)
    text_to_print = f'Time for Node2Vec embedding: {hrs}hr {mins}min {secs:.5f}sec\n'

    print(text_to_print)
    print_to_file(text_to_print)

    # The keys of embedding_dict are type 'str' after being saved in JSON
    start = timer()
    emb_dict = {int(index): vec_graph[int(index)].tolist() for index in sorted(graph.nodes)}
    end = timer()

    hrs, mins, secs = get_time_from_secs(end - start)
    text_to_print = f'Time for embedding_dict: {hrs}hr {mins}min {secs:.5f}sec\n'

    print(text_to_print)
    print_to_file(text_to_print)

    return graph, node_index, emb_dict


def save_graph_data(graph, node_index, emb_dict, json_path):
    dict_to_store = {
        'edges': list(graph.edges),
        'attributes': [
            (node_id, int(graph.nodes[node_id]['x']),
             int(graph.nodes[node_id]['y'])) for
            node_id in graph.nodes],
        'node_index': node_index,
        'embedding_dict': emb_dict
    }
    with open(json_path, 'w') as file:
        json.dump(dict_to_store, file)


def save_graph_pairs(graph, file_path):
    start = timer()
    pair_dict = dict(nx.all_pairs_shortest_path_length(graph))
    end = timer()

    hrs, mins, secs = get_time_from_secs(end - start)
    text_to_print = f'Time for pairs: {hrs}hr {mins}min {secs:.5f}sec\n'

    print(text_to_print)
    print_to_file(text_to_print)

    with open(file_path, 'wb') as file:
        pickle.dump(pair_dict, file)


def load_graph_pairs(file_path):
    with open(file_path, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict


def load_graph_data(json_path):
    with open(json_path, 'r') as file:
        graph_data = json.load(file)

    graph = nx.DiGraph()

    # Add nodes with integer indices and attributes
    for node_data in graph_data['attributes']:
        index, x, y = node_data
        graph.add_node(index, x=int(x), y=int(y))

    for edge_data in graph_data['edges']:
        index1, index2 = edge_data
        graph.add_edge(index1, index2)

    node_index = {int(key): value for key, value in
                  graph_data['node_index'].items()}

    embedding_dict = graph_data['embedding_dict']
    # for index, (node_id, x, y) in enumerate(graph_data['attributes']):
    #     graph.add_node(index, x=int(x), y=int(y), original_node_id=node_id)
    #
    # # Add edges with integer indices
    # edges = [(index1, index2) for index1, index2 in graph_data['edges']]
    # graph.add_edges_from(edges)

    return graph, node_index, embedding_dict


def load_city_paths(json_path=r'.\data_files\city_paths.json'):
    with open(json_path, 'r') as file:
        cities_paths = json.load(file)
    return cities_paths


def get_sets(file_path):
    node_set = set()
    edge_set = set()
    with open(file_path, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break

            line_values = line.replace('\n', '').split('\t')

            if line_values[NEW_TRAJ_VALUE_LOC] == END_OF_TRAJ:
                continue

            x = line_values[CURRENT_X_LOC]
            y = line_values[CURRENT_Y_LOC]
            next_node_x = line_values[NEXT_X_LOC]
            next_node_y = line_values[NEXT_Y_LOC]
            node_id = line_values[CURRENT_NODE_ID]
            next_node_id = line_values[NEXT_NODE_ID]

            node_set.add((node_id, x, y))
            node_set.add((next_node_id, next_node_x, next_node_y))

            edge_set.add((node_id, next_node_id))

    return node_set, edge_set


# Read the id and the object id from the given trajectory
# Create the trajectory from those values and return it
def get_traj_from_line(line_values):
    object_id = line_values[TRAJ_ID_LOC]
    object_class_id = line_values[OBJECT_CLASS_ID_LOC]
    return Trajectory(object_id, object_class_id)


# Read the values for the point from the line_values list
# Create and return a Point object
def get_point_from_line(line_values):
    time = line_values[TIME_LOC]
    x = line_values[CURRENT_X_LOC]
    y = line_values[CURRENT_Y_LOC]
    speed = line_values[CURRENT_SPEED_LOC]
    next_node_x = line_values[NEXT_X_LOC]
    next_node_y = line_values[NEXT_Y_LOC]
    node_id = line_values[CURRENT_NODE_ID]
    next_node_id = line_values[NEXT_NODE_ID]

    return Point(time, x, y, node_id, speed, next_node_x, next_node_y,
                 next_node_id)


def get_trajectory_catalog(file_path):
    traj_catalog = {}

    with open(file_path, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break

            line_values = line.replace('\n', '').split('\t')
            current_traj = get_traj_from_line(line_values)

            if line_values[NEW_TRAJ_VALUE_LOC] == END_OF_TRAJ:
                continue

            # If the trajectory is being created add it to the catalog
            if line_values[NEW_TRAJ_VALUE_LOC] == NEW_TRAJ_VALUE:
                traj_catalog[current_traj.id] = current_traj

            # Add a point to an existing trajectory
            traj_catalog[current_traj.id].add_point(get_point_from_line(line_values))

    return traj_catalog
