import data_handler as dh
import similarity_mesures as sm
import random as rand


oldenburg_dat = r'C:\Trajectory_Generator\generator_v2\oldenburg.dat'
tl_2017_06075_edges_dat = r'C:\Trajectory_Generator\generator_v2\tl_2017_06075_edges.dat'
Tgr06077_dat = r'C:\Trajectory_Generator\generator_v2\Tgr06077-1.dat'
oldenburg_json_path = r'.\data_files\oldenburg_data.json'
tl_json_path = r'.\data_files\tl_data.json'
tg_json_path = r'.\data_files\tg_data.json'
oldenburg_pairs_path = r'.\data_files\oldenburg_data.pkl'
tl_pairs_path = r'.\data_files\tl_data.pkl'
tg_pairs_path = r'.\data_files\tg_data.pkl'

# zip_node_file = r'C:\Trajectory_Generator\generator_v2\oldenburgGen.node.zip'
# node_file = r'oldenburgGen.node'
# zip_edge_file = r'C:\Trajectory_Generator\generator_v2\oldenburgGen.edge.zip'
# edge_file = r'oldenburgGen.edge'

# zip_node_file = r'C:\Trajectory_Generator\generator_v2\tl_2017_06075_edges.node.zip'
# node_file = r'tl_2017_06075_edges.node'
# zip_edge_file = r'C:\Trajectory_Generator\generator_v2\tl_2017_06075_edges.edge.zip'
# edge_file = r'tl_2017_06075_edges.edge'

# zip_node_file = r'C:\Trajectory_Generator\generator_v2\Tgr06077-1.node.zip'
# node_file = r'Tgr06077-1.node'
# zip_edge_file = r'C:\Trajectory_Generator\generator_v2\Tgr06077-1.edge.zip'
# edge_file = r'Tgr06077-1.edge'


def main():
    graph, node_index, embedding_dict = dh.load_graph_data(oldenburg_json_path)
    pair_dict = dh.load_graph_pairs(oldenburg_pairs_path)
    trajectory_catalog = dh.get_trajectory_catalog(oldenburg_dat)
    catalog_num = len(trajectory_catalog)

    print(f'Number of trajectories: {catalog_num}')

    k_10_list = []
    k_100_list = []
    k_1000_list = []

    k = 1000
    for i in range(30):
        print(f'Iteration: {i}')
        rand_trajectory = rand.randint(0, catalog_num)

        distance_similarity = sm.k_distance_similarity(
            trajectory_catalog[rand_trajectory], trajectory_catalog, node_index,
            pair_dict, k
        )

        cos_similarity = sm.k_cosine_similarity(
            trajectory_catalog[rand_trajectory], trajectory_catalog,
            embedding_dict, node_index, k
        )

        dist10 = set()
        cos10 = set()
        dist100 = set()
        cos100 = set()
        dist1000 = set()
        cos1000 = set()
        for index, (dist_sim, cos_sim) in enumerate(
                zip(distance_similarity, cos_similarity)):
            (dist_id1, dist_id2), dist_s = dist_sim
            (cos_id1, cos_id2), cos_s = cos_sim

            if index < 10:
                dist10.add((dist_id1, dist_id2))
                cos10.add((cos_id1, cos_id2))

            if index < 100:
                dist100.add((dist_id1, dist_id2))
                cos100.add((cos_id1, cos_id2))

            dist1000.add((dist_id1, dist_id2))
            cos1000.add((cos_id1, cos_id2))

        k_10_list.append(len(list(dist10.intersection(cos10))))
        k_100_list.append(len(list(dist100.intersection(cos100))))
        k_1000_list.append(len(list(dist1000.intersection(cos1000))))

    print(f'For k=10:')
    print(f'Min sim: {min(k_10_list)}/10')
    print(f'Avg sim: {sum(k_10_list)/len(k_10_list)}/10')
    print(f'Max sim: {max(k_10_list)}/10')

    print('=================================================')

    print(f'For k=100:')
    print(f'Min sim: {min(k_100_list)}/100')
    print(f'Avg sim: {sum(k_100_list) / len(k_10_list)}/100')
    print(f'Max sim: {max(k_100_list)}/100')

    print('=================================================')

    print(f'For k=1000:')
    print(f'Min sim: {min(k_1000_list)}/1000')
    print(f'Avg sim: {sum(k_1000_list) / len(k_1000_list)}/1000')
    print(f'Max sim: {max(k_1000_list)}/1000')


if __name__ == '__main__':
    main()





    # print('Oldenburg:')
    # graph, node_index, embedding_dict = dh.create_graph_and_index(
    #     oldenburg_dat)
    # dh.save_graph_data(graph, node_index, embedding_dict, oldenburg_json_path)
    # # graph, node_index, embedding_dict = dh.load_graph_data(oldenburg_json_path)
    # dh.save_graph_pairs(graph, oldenburg_pairs_path)
    # print(f'Graph: {graph}')
    # # pair_dict = dh.load_graph_pairs(oldenburg_pairs_path)
    #
    # print('===================================================================')
    #
    # print('tl_2017:')
    #
    # graph, node_index, embedding_dict = dh.create_graph_and_index(
    #     tl_2017_06075_edges_dat)
    # dh.save_graph_data(graph, node_index, embedding_dict,
    #                    tl_json_path)
    # # graph, node_index, embedding_dict = dh.load_graph_data(tl_json_path)
    # dh.save_graph_pairs(graph, tl_pairs_path)
    # print(f'Graph: {graph}')
    # # pair_dict = dh.load_graph_pairs(tl_pairs_path)
    #
    # print('===================================================================')
    #
    # print('Tg:')
    #
    # graph, node_index, embedding_dict = dh.create_graph_and_index(
    #     Tgr06077_dat)
    # dh.save_graph_data(graph, node_index, embedding_dict,
    #                    tg_json_path)
    # # graph, node_index, embedding_dict = dh.load_graph_data(tl_json_path)
    # dh.save_graph_pairs(graph, tg_pairs_path)
    # print(f'Graph: {graph}')
    # # pair_dict = dh.load_graph_pairs(tl_pairs_path)
    # k = 100
    #
    # distance_similarity = sm.k_distance_similarity(
    #     trajectory_catalog[0], trajectory_catalog, node_index, pair_dict, k
    # )
    #
    # cos_similarity = sm.k_cosine_similarity(
    #     trajectory_catalog[0], trajectory_catalog, embedding_dict,
    #     node_index, k
    # )
    #
    # set1 = set()
    # set2 = set()
    # for index, (dist_sim, cos_sim) in enumerate(
    #         zip(distance_similarity, cos_similarity)):
    #     (dist_id1, dist_id2), dist_s = dist_sim
    #     (cos_id1, cos_id2), cos_s = cos_sim
    #     print(f'{index + 1}) Dist similarity: nodes {dist_id1}-{dist_id2} with'
    #           f' {dist_s:.4f} | Cos similarity: nodes {cos_id1}-{cos_id2} with'
    #           f' {cos_s:.4f}')
    #     set1.add((dist_id1, dist_id2))
    #     set2.add((cos_id1, cos_id2))
    #
    # print(f'{len(list(set1.intersection(set2)))}/{len(distance_similarity)}')