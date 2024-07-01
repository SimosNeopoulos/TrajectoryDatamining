import json
import pickle
import log_printer as log_pr
from timeit import default_timer as timer

import networkx as nx
from node2vec import Node2Vec as N2V
from tsge2 import TSGE2
from tsge3 import TSGE3

from point import Point
from trajectory import Trajectory

# from main import main_out, testing_out, testing2_out

# Information on how to read a line from the trajectory file
NEW_TRAJECTORY = '0'
END_OF_TRAJECTORY = '2'
NEW_TRAJECTORY_INDEX = 0
TRAJECTORY_ID_INDEX = 1
OBJECT_CLASS_ID_INDEX = 2
TIME_INDEX = 3
CURRENT_X_INDEX = 4
CURRENT_Y_INDEX = 5
CURRENT_SPEED_INDEX = 6
NEXT_X_INDEX = 7
NEXT_Y_INDEX = 8
CURRENT_NODE_ID_INDEX = 9
NEXT_NODE_ID_INDEX = 10


# old code, before started using brute force with generator ###################

# def read_node(in_file):
#     try:
#         name_len = struct.unpack('b', in_file.read(1))[0]
#         name = in_file.read(name_len).decode('utf-8')
#         _id = struct.unpack('q', in_file.read(8))[0]
#         x = struct.unpack('i', in_file.read(4))[0]
#         y = struct.unpack('i', in_file.read(4))[0]
#         return name, _id, x, y
#     except (IOError, struct.error):
#         return None
#
#
# def read_edge(in_file):
#     try:
#         node_id1, node_id2, name_len = struct.unpack('qqb', in_file.read(17))
#         name = in_file.read(name_len).decode()
#         _id, edge_class = struct.unpack('qi', in_file.read(12))
#         return node_id1, node_id2, name, _id, edge_class
#     except (IOError, struct.error):
#         return None
#
#
# def get_node_data(zip_node_file, node_file):
#     node_file_data = []
#     with zp.ZipFile(zip_node_file, 'r') as zip_f:
#         with zip_f.open(node_file, 'r') as file:
#             while True:
#                 node_data = read_node(file)
#
#                 if node_data is None:
#                     break
#
#                 node_file_data.append(node_data)
#
#     return node_file_data
#
#
# def get_edge_data(zip_edge_file, edge_file):
#     edge_file_data = []
#     with zp.ZipFile(zip_edge_file, 'r') as zip_f:
#         with zip_f.open(edge_file, 'r') as file:
#             while True:
#                 edge_data = read_edge(file)
#
#                 if edge_data is None:
#                     break
#
#                 edge_file_data.append(edge_data)
#
#     return edge_file_data
#
#
# def create_graph_from_files(zip_node_file, node_file, zip_edge_file, edge_file):
#     graph = nx.DiGraph()
#     node_data = get_node_data(zip_node_file, node_file)
#     edge_data = get_edge_data(zip_edge_file, edge_file)
#
#     for node in node_data:
#         name, node_id, x, y = node
#         graph.add_node(node_id, x=x, y=y, name=name)
#
#     for edge in edge_data:
#         node_id1, node_id2, name, edge_id, edge_class = edge
#         graph.add_edge(node_id1, node_id2)
#
#     return graph


########################################################################


def format_time(secs):
    """Convert seconds into hours, minutes, and seconds."""
    hrs = secs // 3600
    mins = (secs % 3600) // 60
    rem_secs = secs % 60

    return int(hrs), int(mins), rem_secs


def load_city_paths(json_path=r'city_paths.json'):
    """Load city paths from a JSON file."""
    with open(json_path, 'r') as file:
        cities_paths = json.load(file)
    return cities_paths


############################ PAGERANK ################################


def compute_pagerank(graph: nx.DiGraph, log_path: str) -> dict:
    """Compute PageRank for a given graph and log the time taken."""
    start = timer()
    pagerank = nx.pagerank(graph)
    end = timer()

    hrs, mins, secs = format_time(end - start)
    text_to_print = f'Time for pagerank: {hrs}hr {mins}min {secs:.5f}sec\n'

    print(text_to_print)
    log_pr.print_to_file(text_to_print, log_path)

    return pagerank


def save_pagerank(pagerank_dict: dict, json_path: str):
    """Save the PageRank dictionary to a file."""
    with open(json_path, 'wb') as file:
        pickle.dump(pagerank_dict, file)


def load_pagerank(file_path: str) -> dict:
    """Load the PageRank dictionary from a file."""
    with open(file_path, 'rb') as file:
        pagerank_dict = pickle.load(file)
    return pagerank_dict


#######################################################################

################### CREATE, SAVE AND LOAD GRAPH #######################


def create_graph(traj_path: str, log_path: str) -> nx.DiGraph:
    """Create a directed graph from trajectory data and log the time taken."""
    start = timer()

    graph = nx.DiGraph()
    node_set, edge_set = extract_nodes_edges(traj_path)

    for node_id, x, y in node_set:
        graph.add_node(int(node_id), x=int(x), y=int(y))

    for node_id1, node_id2 in edge_set:
        graph.add_edge(int(node_id1), int(node_id2))

    end = timer()

    print(graph.__str__())
    log_pr.print_to_file(graph.__str__() + '\n', log_path)

    hrs, mins, secs = format_time(end - start)
    text_to_print = f'Time to create graph: {hrs}hr {mins}min {secs:.5f}sec\n'
    print(text_to_print)
    log_pr.print_to_file(text_to_print, log_path)
    return graph


def save_graph(graph: nx.DiGraph, json_path: str):
    """Save graph data to a JSON file."""
    graph_data = {
        'edges': list(graph.edges),
        'attributes': [
            (node_id,
             int(graph.nodes[node_id]['x']),
             int(graph.nodes[node_id]['y'])
             ) for node_id in graph.nodes
        ],
    }
    with open(json_path, 'w') as file:
        json.dump(graph_data, file)


def load_graph(graph_path: str) -> nx.DiGraph:
    """Load a graph from a JSON file."""
    with open(graph_path, 'r') as file:
        graph_data = json.load(file)

    graph = nx.DiGraph()

    # Add nodes with integer indices and attributes
    for node_id, x, y in graph_data['attributes']:
        graph.add_node(node_id, x=int(x), y=int(y))

    for index1, index2 in graph_data['edges']:
        graph.add_edge(index1, index2)

    return graph


#######################################################################


################## SAVE AND LOAD GRAPH EMBEDDING ######################


def save_embedding(embedding: dict, file_path: str):
    """Save embedding dictionary to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(embedding, file)


def load_embedding(file_path: str) -> dict:
    """Load embedding dictionary from a JSON file."""
    with open(file_path, 'r') as file:
        embedding = json.load(file)
    return embedding


#######################################################################

####################### TSGE1, TSGE2, TSGE3 ###########################

def create_tsge1_embedding(graph: nx.DiGraph, dimensions: int, log_path: str) -> dict:
    """Create TSGE1 embedding using Node2Vec and log the time taken."""
    start = timer()
    node2vec = N2V(graph, dimensions=dimensions)
    model = node2vec.fit()
    end = timer()

    hrs, mins, secs = format_time(end - start)
    text_to_print = f'Time for Node2Vec embedding: {hrs}hr {mins}min {secs:.5f}sec\n'

    print(text_to_print)
    log_pr.print_to_file(text_to_print, log_path)

    start = timer()
    embedding = {str(index): model.wv[str(index)].tolist() for index in sorted(graph.nodes) if str(index) in model.wv}
    end = timer()

    hrs, mins, secs = format_time(end - start)
    text_to_print = f'Time for TSGE1 embedding: {hrs}hr {mins}min {secs:.5f}sec\n'

    print(text_to_print)
    log_pr.print_to_file(text_to_print, log_path)
    return embedding


def create_tsge2_embedding(graph: nx.DiGraph, dimensions: int, traj_data: dict, log_path: str) -> dict:
    """Create TSGE2 embedding using trajectory data and log the time taken."""
    start = timer()
    tsge2_model = TSGE2(graph, dimensions)
    embedding = tsge2_model.fit(traj_data)
    end = timer()

    hrs, mins, secs = format_time(end - start)
    log_message = f'Time for TSGE2 embedding: {hrs}hr {mins}min {secs:.5f}sec\n'
    print(log_message)
    log_pr.print_to_file(log_message, log_path)

    return embedding


def create_tsge3_embedding(graph: nx.DiGraph, dimensions: int, traj_data: dict, log_path: str) -> dict:
    """Create TSGE3 embedding using doc2vec and log the time taken."""
    start = timer()
    node2vec = N2V(graph)
    tsge3_model = TSGE3(node2vec.walks, vector_size=dimensions)
    tsge3_model.train_model()
    trajectory_embedding = tsge3_model.get_traj_emb(traj_data)
    end = timer()

    hrs, mins, secs = format_time(end - start)
    log_message = f'Time for TSGE3 embedding: {hrs}hr {mins}min {secs:.5f}sec\n'
    print(log_message)
    log_pr.print_to_file(log_message, log_path)

    return trajectory_embedding


#################################################################################


######################### GRAPH SHORTEST PAIRS ##################################


def save_shortest_paths(graph: nx.DiGraph, file_path: str, log_path: str):
    """Compute and save all-pairs shortest paths lengths for the graph."""
    start = timer()
    shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))
    end = timer()

    hrs, mins, secs = format_time(end - start)
    log_message = f'Time for computing shortest paths: {hrs}hr {mins}min {secs:.5f}sec\n'
    print(log_message)
    log_pr.print_to_file(log_message, log_path)

    with open(file_path, 'wb') as file:
        pickle.dump(shortest_paths, file)


def load_shortest_paths(file_path: str) -> dict:
    """Load all-pairs shortest paths lengths from a file."""
    with open(file_path, 'rb') as file:
        shortest_paths = pickle.load(file)
    return shortest_paths


############################################################################


def extract_nodes_edges(file_path: str) -> tuple[set, set]:
    """Extract nodes and edges from trajectory file."""
    nodes = set()
    edges = set()

    with open(file_path, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break

            line_values = line.strip().split('\t')

            if line_values[NEW_TRAJECTORY_INDEX] == END_OF_TRAJECTORY:
                continue

            current_x = line_values[CURRENT_X_INDEX]
            current_y = line_values[CURRENT_Y_INDEX]
            next_x = line_values[NEXT_X_INDEX]
            next_y = line_values[NEXT_Y_INDEX]
            node_id = line_values[CURRENT_NODE_ID_INDEX]
            next_node_id = line_values[NEXT_NODE_ID_INDEX]

            nodes.add((node_id, current_x, current_y))
            nodes.add((next_node_id, next_x, next_y))

            edges.add((node_id, next_node_id))

    return nodes, edges


def get_trajectory_from_line(line_values: list) -> Trajectory:
    """Create a Trajectory object from line values."""
    traj_id = line_values[TRAJECTORY_ID_INDEX]
    object_class_id = line_values[OBJECT_CLASS_ID_INDEX]
    return Trajectory(traj_id, object_class_id)


def get_point_from_line(line_values: list) -> Point:
    """Create a Point object from line values."""
    timestamp = line_values[TIME_INDEX]
    x = line_values[CURRENT_X_INDEX]
    y = line_values[CURRENT_Y_INDEX]
    speed = line_values[CURRENT_SPEED_INDEX]
    next_x = line_values[NEXT_X_INDEX]
    next_y = line_values[NEXT_Y_INDEX]
    node_id = line_values[CURRENT_NODE_ID_INDEX]
    next_node_id = line_values[NEXT_NODE_ID_INDEX]

    return Point(timestamp, x, y, node_id, speed, next_x, next_y, next_node_id)


def build_trajectory_catalog(file_path: str) -> dict:
    """Build a catalog of trajectories from a file."""
    trajectory_catalog = {}

    with open(file_path, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break

            line_values = line.strip().split('\t')
            current_trajectory = get_trajectory_from_line(line_values)

            if line_values[NEW_TRAJECTORY_INDEX] == END_OF_TRAJECTORY:
                continue

            if line_values[NEW_TRAJECTORY_INDEX] == NEW_TRAJECTORY:
                trajectory_catalog[current_trajectory.id] = current_trajectory

            trajectory_catalog[current_trajectory.id].add_point(get_point_from_line(line_values))

    return trajectory_catalog

#######################################################################################


############################# TSGE4 ###################################
# def create_biased_traj_emb(traj_dict, log_file):
#     start = timer()
#     biased_traj2vec = BT2V(traj_dict)
#     biased_traj2vec.train_model()
#     traj_emb = biased_traj2vec.get_traj_emb(traj_dict)
#     end = timer()
#
#     hrs, mins, secs = get_time_from_secs(end - start)
#     text_to_print = f'Time for Biased Traj2Vec embedding: {hrs}hr {mins}min {secs:.5f}sec\n'
#
#     print(text_to_print)
#     log_pr.print_to_file(text_to_print, log_file)
#
#     return traj_emb
