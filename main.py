import data_handler as dh
import similarity_mesures as sm

path_to_trajectory = r'C:\Trajectory_Generator\generator_v2\oldenberg.dat'
# path_to_trajectory = r'C:\Trajectory_Generator\generator_v2\tl_2017_06075_edges.dat'
oldenburg_json_path = r'.\data_files\oldenburg_data.json'
tl_json_path = r'.\data_files\tl_data.json'
oldenburg_pairs_path = r'.\data_files\oldenburg_data.pkl'
tl_pairs_path = r'.\data_files\tl_data.pkl'


# zip_node_file = r'C:\Trajectory_Generator\generator_v2\oldenburgGen.node.zip'
# node_file = r'oldenburgGen.node'
# zip_edge_file = r'C:\Trajectory_Generator\generator_v2\oldenburgGen.edge.zip'
# edge_file = r'oldenburgGen.edge'

# zip_node_file = r'C:\Trajectory_Generator\generator_v2\tl_2017_06075_edges.node.zip'
# node_file = r'tl_2017_06075_edges.node'
# zip_edge_file = r'C:\Trajectory_Generator\generator_v2\tl_2017_06075_edges.edge.zip'
# edge_file = r'tl_2017_06075_edges.edge'


def main():
    print(':)')

    # graph, node_index, embedding_dict = dh.create_graph_and_index(
    #     path_to_trajectory)
    # dh.save_graph_data(graph, node_index, embedding_dict,
    #                 oldenburg_json_path)
    # graph, node_index, embedding_dict = dh.load_graph_data(
    #     tl_json_path)
    # dh.save_graph_pairs(graph, tl_pairs_path)
    # pair_dict = dh.load_graph_pairs(oldenburg_pairs_path)


    # trajectory_catalog = get_trajectory_catalog(path_to_trajectory)
    #
    # print(k_similar_trajectories(trajectory_catalog[0], trajectory_catalog, 5))


if __name__ == '__main__':
    main()
