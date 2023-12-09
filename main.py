from functions import *
from karateclub.node_embedding.neighbourhood.deepwalk import DeepWalk
import networkx as nx

path_to_trajectory = r'C:\Trajectory_Generator\generator_v2\oldenberg.dat'
# path_to_trajectory = r'C:\Trajectory_Generator\generator_v2\tl_2017_06075_edges.dat'
oldenburg_json_path = r'.\data_files\oldenburg_data.json'
tl_json_path = r'.\data_files\tl_data.json'


# zip_node_file = r'C:\Trajectory_Generator\generator_v2\oldenburgGen.node.zip'
# node_file = r'oldenburgGen.node'
# zip_edge_file = r'C:\Trajectory_Generator\generator_v2\oldenburgGen.edge.zip'
# edge_file = r'oldenburgGen.edge'

# zip_node_file = r'C:\Trajectory_Generator\generator_v2\tl_2017_06075_edges.node.zip'
# node_file = r'tl_2017_06075_edges.node'
# zip_edge_file = r'C:\Trajectory_Generator\generator_v2\tl_2017_06075_edges.edge.zip'
# edge_file = r'tl_2017_06075_edges.edge'


def main():
    graph, node_index = create_graph_and_index(path_to_trajectory)
    save_graph_data(graph, node_index, oldenburg_json_path)
    graph, node_index = load_graph_data(oldenburg_json_path)
    print(graph)
    # print(node_index)

    deep_walk = DeepWalk()
    deep_walk.fit(graph)
    vec_graph = deep_walk.get_embedding()

    print(vec_graph)


    # graph_file = create_graph_from_files(zip_node_file, node_file, zip_edge_file, edge_file)
    # print(graph_file)
    # trajectory_catalog = get_trajectory_catalog(path_to_trajectory)
    #
    # print(k_similar_trajectories(trajectory_catalog[0], trajectory_catalog, 5))


if __name__ == '__main__':
    main()

