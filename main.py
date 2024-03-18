import data_handler as dh
import similarity_mesures as sm
import random as rand
from timeit import default_timer as timer

cities = ['oldenburg', 'san_francisco', 'anchorage', 'el_dorado', 'knox_county', 'california']


# oldenburg has name problem on pairs
# anchorage crushes on dist similarity somewhere
# el_dorado line 197, in load_graph_pairs loaded_dict = pickle.load(file): MemoryError

def get_city_name(name):
    return str.title(name.replace('_', ' '))


def print_trajectory_info(trajectory_catalog):
    catalog_num = len(trajectory_catalog)
    text_to_print = f'Number of trajectories: {catalog_num}'
    print(text_to_print)
    dh.print_to_file(text_to_print)

    trajectory_length_list = []
    for trajectory in trajectory_catalog.values():
        trajectory_length_list.append(len(trajectory.trajectory_path))

    text_to_print = f'Min length of trajectories: {min(trajectory_length_list)}'
    print(text_to_print)
    dh.print_to_file(text_to_print)
    text_to_print = f'Avg length of trajectories: {sum(trajectory_length_list) / catalog_num}'
    print(text_to_print)
    dh.print_to_file(text_to_print)
    text_to_print = f'Max length of trajectories: {max(trajectory_length_list)}\n'
    print(text_to_print)
    dh.print_to_file(text_to_print)


def print_similarity(node_index, emb_dict, pair_dict, traj_catalog, catalog_num):
    k_10_list = []
    k_100_list = []
    k_1000_list = []

    dist_time = []
    cos_time = []

    k = 1000
    all_start = timer()
    for i in range(20):
        start = timer()
        rand_trajectory = rand.randint(0, catalog_num)

        start_dist = timer()
        distance_similarity = sm.k_distance_similarity(
            traj_catalog[rand_trajectory], traj_catalog, node_index,
            pair_dict, k
        )
        end_dist = timer()
        dist_time.append(end_dist - start_dist)

        start_cos = timer()
        cos_similarity = sm.k_cosine_similarity(
            traj_catalog[rand_trajectory], traj_catalog,
            emb_dict, node_index, k
        )
        end_cos = timer()
        cos_time.append(end_cos - start_cos)

        dist10 = set()
        cos10 = set()
        dist100 = set()
        cos100 = set()
        dist1000 = set()
        cos1000 = set()
        for index, (dist_sim, cos_sim) in enumerate(zip(distance_similarity, cos_similarity)):

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

        k_10_sim = len(list(dist10.intersection(cos10)))
        k_100_sim = len(list(dist100.intersection(cos100)))
        k_1000_sim = len(list(dist1000.intersection(cos1000)))

        k_10_list.append(k_10_sim)
        k_100_list.append(k_100_sim)
        k_1000_list.append(k_1000_sim)

        end = timer()
        hrs, mins, secs = dh.get_time_from_secs(end - start)
        dist_hrs, dist_mins, dist_secs = dh.get_time_from_secs(end_dist - start_dist)
        cos_hrs, cos_mins, cos_secs = dh.get_time_from_secs(end_cos - start_cos)
        text_to_print = f'Iteration: {i + 1}. Time of iteration: {hrs}hr {mins}min {secs:.5f}secs.' \
                        f' Distance Similarity time: {dist_hrs}hr {dist_mins}min {dist_secs:.5f}secs.' \
                        f' Cosine Similarity time: {cos_hrs}hr {cos_mins}min {cos_secs:.5f}secs.' \
                        f' Trajectory of iteration: {rand_trajectory}.' \
                        f' k=10 {k_10_sim}/10, k=100 {k_100_sim}/100, k=1000 {k_1000_sim}/1000.'

        print(text_to_print)
        dh.print_to_file(text_to_print)

    all_end = timer()

    text_to_print = f'For k=10:\n' \
                    f'Min similarity: {min(k_10_list)}/10\n' \
                    f'Avg similarity: {sum(k_10_list) / len(k_10_list)}/10\n' \
                    f'Max similarity: {max(k_10_list)}/10\n'
    print(text_to_print)
    dh.print_to_file(text_to_print)

    print('=================================================')

    text_to_print = f'For k=100:\n' \
                    f'Min similarity: {min(k_100_list)}/100\n' \
                    f'Avg similarity: {sum(k_100_list) / len(k_100_list)}/100\n' \
                    f'Max similarity: {max(k_100_list)}/100\n'
    print(text_to_print)
    dh.print_to_file(text_to_print)

    print('=================================================')

    text_to_print = f'For k=1000:\n' \
                    f'Min similarity: {min(k_1000_list)}/1000\n' \
                    f'Avg similarity: {sum(k_1000_list) / len(k_1000_list)}/1000\n' \
                    f'Max similarity: {max(k_1000_list)}/1000\n'
    print(text_to_print)
    dh.print_to_file(text_to_print)

    min_hr, min_mins, min_secs = dh.get_time_from_secs(min(dist_time))
    avg_hr, avg_mins, avg_secs = dh.get_time_from_secs(sum(dist_time) / len(dist_time))
    max_hr, max_mins, max_secs = dh.get_time_from_secs(max(dist_time))
    sum_hr, sum_mins, sum_secs = dh.get_time_from_secs(sum(dist_time))
    text_to_print = f'Distance Similarity time:\n' \
                    f'Min time: {min_hr}hr {min_mins}min {min_secs:.5f}secs.\n' \
                    f'Avg time: {avg_hr}hr {avg_mins}min {avg_secs:.5f}secs.\n' \
                    f'Max time: {max_hr}hr {max_mins}min {max_secs:.5f}secs.\n' \
                    f'Total time: {sum_hr}hr {sum_mins}min {sum_secs:.5f}secs.\n'
    print(text_to_print)
    dh.print_to_file(text_to_print)

    min_hr, min_mins, min_secs = dh.get_time_from_secs(min(cos_time))
    avg_hr, avg_mins, avg_secs = dh.get_time_from_secs(sum(cos_time) / len(cos_time))
    max_hr, max_mins, max_secs = dh.get_time_from_secs(max(cos_time))
    sum_hr, sum_mins, sum_secs = dh.get_time_from_secs(sum(cos_time))
    text_to_print = f'Distance Similarity time:\n' \
                    f'Min time: {min_hr}hr {min_mins}min {min_secs:.5f}secs.\n' \
                    f'Avg time: {avg_hr}hr {avg_mins}min {avg_secs:.5f}secs.\n' \
                    f'Max time: {max_hr}hr {max_mins}min {max_secs:.5f}secs.\n' \
                    f'Total time: {sum_hr}hr {sum_mins}min {sum_secs:.5f}secs.\n'
    print(text_to_print)
    dh.print_to_file(text_to_print)

    hrs, mins, secs = dh.get_time_from_secs(all_end - all_start)

    text_to_print = f'Time for all iterations: {hrs}hr {mins}min {secs:.5f}sec\n'
    print(text_to_print)
    dh.print_to_file(text_to_print)
    dh.print_to_file('*******************************************************************\n')
    print()


def print_similarity_test(node_index, emb_dict, traj_catalog, catalog_num):
    k_10_list = []
    k_100_list = []
    k_1000_list = []

    cos_time = []

    k = 1000
    all_start = timer()
    for i in range(20):
        start = timer()
        rand_trajectory = rand.randint(0, catalog_num)

        start_cos = timer()
        cos_similarity = sm.k_cosine_similarity(
            traj_catalog[rand_trajectory], traj_catalog,
            emb_dict, node_index, k
        )
        end_cos = timer()
        cos_time.append(end_cos - start_cos)

        cos10 = set()
        cos100 = set()
        cos1000 = set()
        for index, cos_sim in enumerate(cos_similarity):

            (cos_id1, cos_id2), cos_s = cos_sim

            if index < 10:
                cos10.add((cos_id1, cos_id2))

            if index < 100:
                cos100.add((cos_id1, cos_id2))

            cos1000.add((cos_id1, cos_id2))

        k_10_sim = len(cos10)
        k_100_sim = len(cos100)
        k_1000_sim = len(cos1000)

        k_10_list.append(k_10_sim)
        k_100_list.append(k_100_sim)
        k_1000_list.append(k_1000_sim)

        end = timer()
        hrs, mins, secs = dh.get_time_from_secs(end - start)
        cos_hrs, cos_mins, cos_secs = dh.get_time_from_secs(end_cos - start_cos)
        text_to_print = f'Iteration: {i + 1}. Time of iteration: {hrs}hr {mins}min {secs:.5f}secs.' \
                        f' Cosine Similarity time: {cos_hrs}hr {cos_mins}min {cos_secs:.5f}secs.' \
                        f' Trajectory of iteration: {rand_trajectory}.' \
                        f' k=10 {k_10_sim}/10, k=100 {k_100_sim}/100, k=1000 {k_1000_sim}/1000.'

        print(text_to_print)
        dh.print_to_file(text_to_print)

    all_end = timer()

    text_to_print = f'For k=10:\n' \
                    f'Min similarity: {min(k_10_list)}/10\n' \
                    f'Avg similarity: {sum(k_10_list) / len(k_10_list)}/10\n' \
                    f'Max similarity: {max(k_10_list)}/10\n'
    print(text_to_print)
    dh.print_to_file(text_to_print)

    print('=================================================')

    text_to_print = f'For k=100:\n' \
                    f'Min similarity: {min(k_100_list)}/100\n' \
                    f'Avg similarity: {sum(k_100_list) / len(k_100_list)}/100\n' \
                    f'Max similarity: {max(k_100_list)}/100\n'
    print(text_to_print)
    dh.print_to_file(text_to_print)

    print('=================================================')

    text_to_print = f'For k=1000:\n' \
                    f'Min similarity: {min(k_1000_list)}/1000\n' \
                    f'Avg similarity: {sum(k_1000_list) / len(k_1000_list)}/1000\n' \
                    f'Max similarity: {max(k_1000_list)}/1000\n'
    print(text_to_print)
    dh.print_to_file(text_to_print)

    min_hr, min_mins, min_secs = dh.get_time_from_secs(min(cos_time))
    avg_hr, avg_mins, avg_secs = dh.get_time_from_secs(sum(cos_time) / len(cos_time))
    max_hr, max_mins, max_secs = dh.get_time_from_secs(max(cos_time))
    sum_hr, sum_mins, sum_secs = dh.get_time_from_secs(sum(cos_time))
    text_to_print = f'Distance Similarity time:\n' \
                    f'Min time: {min_hr}hr {min_mins}min {min_secs:.5f}secs.\n' \
                    f'Avg time: {avg_hr}hr {avg_mins}min {avg_secs:.5f}secs.\n' \
                    f'Max time: {max_hr}hr {max_mins}min {max_secs:.5f}secs.\n' \
                    f'Total time: {sum_hr}hr {sum_mins}min {sum_secs:.5f}secs.\n'
    print(text_to_print)
    dh.print_to_file(text_to_print)

    hrs, mins, secs = dh.get_time_from_secs(all_end - all_start)

    text_to_print = f'Time for all iterations: {hrs}hr {mins}min {secs:.5f}sec\n'
    print(text_to_print)
    dh.print_to_file(text_to_print)
    dh.print_to_file('*******************************************************************\n')
    print()


# zip_node_file = r'C:\Trajectory_Generator\generator_v2\El_Dorado_tl_2023_06017_edges.node.zip'
# node_file = r'El_Dorado_tl_2023_06017_edges.node.node'
# zip_edge_file = r'C:\Trajectory_Generator\generator_v2\El_Dorado_tl_2023_06017_edges.edge.zip'
# edge_file = r'El_Dorado_tl_2023_06017_edges.edge.edge'

# zip_node_file = r'C:\Trajectory_Generator\generator_v2\Knox_County_tl_2023_39083_edges.node.zip'
# node_file = r'Knox_County_tl_2023_39083_edges.node.node'
# zip_edge_file = r'C:\Trajectory_Generator\generator_v2\Knox_County_tl_2023_39083_edges.edge.zip'
# edge_file = r'Knox_County_tl_2023_39083_edges.edge.edge'

# zip_node_file = r'C:\Trajectory_Generator\generator_v2\California_tl_2023_06037_edges.node.zip'
# node_file = r'California_tl_2023_06037_edges.node.node'
# zip_edge_file = r'C:\Trajectory_Generator\generator_v2\California_tl_2023_06037_edges.edge.zip'
# edge_file = r'California_tl_2023_06037_edges.edge.edge'


def main():
    city_paths = dh.load_city_paths()

    for city in cities[:2]:
        dh.print_to_file(get_city_name(city) + ':\n')

        graph, node_index, emb_dict = dh.load_graph_data(city_paths[city]['data'])
        pair_dict = dh.load_graph_pairs(city_paths[city]['pairs'])
        traj_catalog = dh.get_trajectory_catalog(city_paths[city]['dat_file'])
        catalog_num = len(traj_catalog)
        print_trajectory_info(traj_catalog)

        print_similarity(node_index=node_index,
                         emb_dict=emb_dict,
                         pair_dict=pair_dict,
                         traj_catalog=traj_catalog,
                         catalog_num=catalog_num
                         )
        del node_index, emb_dict, traj_catalog


if __name__ == '__main__':
    main()

    # graph = dh.create_graph_from_files(zip_node_file, node_file, zip_edge_file,
    #                                    edge_file)
    #
    # print('California:')
    # print('Graph from files: ' + graph.__str__())
    #
    # city_paths = dh.load_city_paths()
    #
    # dh.print_to_file('California:\n')
    # graph, node_index, embedding_dict = dh.create_graph_and_index(
    #     city_paths[cities[5]]['dat_file'])
    # dh.save_graph_data(graph, node_index, embedding_dict,
    #                    city_paths[cities[5]]['data'])
    # dh.save_graph_pairs(graph, city_paths[cities[5]]['pairs'])
