import data_handler as dh
import similarity_mesures as sm
import random as rand
from timeit import default_timer as timer


def print_to_file(text, file_path=f'./log_files/similarity_data.txt'):
    with open(file_path, 'a') as file:
        file.write(text + '\n')


def print_trajectory_info(trajectory_catalog, file=f'./log_files/cos_sim.txt'):
    catalog_num = len(trajectory_catalog)
    text_to_print = f'Number of trajectories: {catalog_num}'
    print(text_to_print)
    print_to_file(text_to_print, file)

    trajectory_length_list = []
    for trajectory in trajectory_catalog.values():
        trajectory_length_list.append(len(trajectory.trajectory_path))

    text_to_print = f'Min length of trajectories: {min(trajectory_length_list)}'
    print(text_to_print)
    print_to_file(text_to_print, file)
    text_to_print = f'Avg length of trajectories: {sum(trajectory_length_list) / catalog_num}'
    print(text_to_print)
    print_to_file(text_to_print, file)
    text_to_print = f'Max length of trajectories: {max(trajectory_length_list)}\n'
    print(text_to_print)
    print_to_file(text_to_print, file)


def print_biased_similarity(emb_dict, pair_dict, traj_catalog, catalog_num, log_file):
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
        biased_distance_similarity = sm.biased_k_dist_sim(
            traj_catalog[rand_trajectory], traj_catalog,
            pair_dict, k
        )
        end_dist = timer()
        dist_time.append(end_dist - start_dist)

        start_cos = timer()
        biased_cos_similarity = sm.biased_k_cos_sim(
            traj_catalog[rand_trajectory], traj_catalog,
            emb_dict, k
        )
        end_cos = timer()
        cos_time.append(end_cos - start_cos)

        dist10 = set()
        cos10 = set()
        dist100 = set()
        cos100 = set()
        dist1000 = set()
        cos1000 = set()
        for index, (dist_sim, cos_sim) in enumerate(zip(biased_distance_similarity, biased_cos_similarity)):

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
        print_to_file(text_to_print, log_file)

    all_end = timer()

    text_to_print = f'\nFor k=10:\n' \
                    f'Min similarity: {min(k_10_list)}/10\n' \
                    f'Avg similarity: {sum(k_10_list) / len(k_10_list)}/10\n' \
                    f'Max similarity: {max(k_10_list)}/10\n'
    print(text_to_print)
    print_to_file(text_to_print, log_file)

    print('=================================================')

    text_to_print = f'For k=100:\n' \
                    f'Min similarity: {min(k_100_list)}/100\n' \
                    f'Avg similarity: {sum(k_100_list) / len(k_100_list)}/100\n' \
                    f'Max similarity: {max(k_100_list)}/100\n'
    print(text_to_print)
    print_to_file(text_to_print, log_file)

    print('=================================================')

    text_to_print = f'For k=1000:\n' \
                    f'Min similarity: {min(k_1000_list)}/1000\n' \
                    f'Avg similarity: {sum(k_1000_list) / len(k_1000_list)}/1000\n' \
                    f'Max similarity: {max(k_1000_list)}/1000\n'
    print(text_to_print)
    print_to_file(text_to_print, log_file)

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
    print_to_file(text_to_print, log_file)

    min_hr, min_mins, min_secs = dh.get_time_from_secs(min(cos_time))
    avg_hr, avg_mins, avg_secs = dh.get_time_from_secs(sum(cos_time) / len(cos_time))
    max_hr, max_mins, max_secs = dh.get_time_from_secs(max(cos_time))
    sum_hr, sum_mins, sum_secs = dh.get_time_from_secs(sum(cos_time))
    text_to_print = f'Cosine Similarity time:\n' \
                    f'Min time: {min_hr}hr {min_mins}min {min_secs:.5f}secs.\n' \
                    f'Avg time: {avg_hr}hr {avg_mins}min {avg_secs:.5f}secs.\n' \
                    f'Max time: {max_hr}hr {max_mins}min {max_secs:.5f}secs.\n' \
                    f'Total time: {sum_hr}hr {sum_mins}min {sum_secs:.5f}secs.\n'
    print(text_to_print)
    print_to_file(text_to_print, log_file)

    hrs, mins, secs = dh.get_time_from_secs(all_end - all_start)

    text_to_print = f'Time for all iterations: {hrs}hr {mins}min {secs:.5f}sec\n'
    print(text_to_print)
    print_to_file(text_to_print, log_file)
    print_to_file('*******************************************************************\n', log_file)
    print()


def print_similarity(node_index, emb_dict, pair_dict, traj_catalog, catalog_num, log_file):
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
        distance_similarity = sm.k_dist_sim(
            traj_catalog[rand_trajectory], traj_catalog, node_index,
            pair_dict, k
        )
        end_dist = timer()
        dist_time.append(end_dist - start_dist)

        start_cos = timer()
        cos_similarity = sm.k_cos_sim(
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
        print_to_file(text_to_print, log_file)

    all_end = timer()

    text_to_print = f'\nFor k=10:\n' \
                    f'Min similarity: {min(k_10_list)}/10\n' \
                    f'Avg similarity: {sum(k_10_list) / len(k_10_list)}/10\n' \
                    f'Max similarity: {max(k_10_list)}/10\n'
    print(text_to_print)
    print_to_file(text_to_print, log_file)

    print('=================================================')

    text_to_print = f'For k=100:\n' \
                    f'Min similarity: {min(k_100_list)}/100\n' \
                    f'Avg similarity: {sum(k_100_list) / len(k_100_list)}/100\n' \
                    f'Max similarity: {max(k_100_list)}/100\n'
    print(text_to_print)
    print_to_file(text_to_print, log_file)

    print('=================================================')

    text_to_print = f'For k=1000:\n' \
                    f'Min similarity: {min(k_1000_list)}/1000\n' \
                    f'Avg similarity: {sum(k_1000_list) / len(k_1000_list)}/1000\n' \
                    f'Max similarity: {max(k_1000_list)}/1000\n'
    print(text_to_print)
    print_to_file(text_to_print, log_file)

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
    print_to_file(text_to_print, log_file)

    min_hr, min_mins, min_secs = dh.get_time_from_secs(min(cos_time))
    avg_hr, avg_mins, avg_secs = dh.get_time_from_secs(sum(cos_time) / len(cos_time))
    max_hr, max_mins, max_secs = dh.get_time_from_secs(max(cos_time))
    sum_hr, sum_mins, sum_secs = dh.get_time_from_secs(sum(cos_time))
    text_to_print = f'Cosine Similarity time:\n' \
                    f'Min time: {min_hr}hr {min_mins}min {min_secs:.5f}secs.\n' \
                    f'Avg time: {avg_hr}hr {avg_mins}min {avg_secs:.5f}secs.\n' \
                    f'Max time: {max_hr}hr {max_mins}min {max_secs:.5f}secs.\n' \
                    f'Total time: {sum_hr}hr {sum_mins}min {sum_secs:.5f}secs.\n'
    print(text_to_print)
    print_to_file(text_to_print, log_file)

    hrs, mins, secs = dh.get_time_from_secs(all_end - all_start)

    text_to_print = f'Time for all iterations: {hrs}hr {mins}min {secs:.5f}sec\n'
    print(text_to_print)
    print_to_file(text_to_print, log_file)
    print_to_file('*******************************************************************\n', log_file)
    print()


def print_similarity_test(node_index, emb_dict, traj_catalog, catalog_num, log_file=None):
    log_file = f'./log_files/cos_sim.txt'
    cos_time = []

    k = 1000
    all_start = timer()
    for i in range(20):
        start = timer()
        rand_trajectory = rand.randint(0, catalog_num)

        start_cos = timer()
        cos_similarity = sm.k_cos_sim(
            traj_catalog[rand_trajectory], traj_catalog,
            emb_dict, node_index, k
        )
        end_cos = timer()
        cos_time.append(end_cos - start_cos)

        end = timer()
        hrs, mins, secs = dh.get_time_from_secs(end - start)
        cos_hrs, cos_mins, cos_secs = dh.get_time_from_secs(end_cos - start_cos)
        text_to_print = f'Iteration: {i + 1}. Time of iteration: {hrs}hr {mins}min {secs:.5f}secs.' \
                        f' Cosine Similarity time: {cos_hrs}hr {cos_mins}min {cos_secs:.5f}secs.' \
                        f' Trajectory of iteration: {rand_trajectory}.'

        print(text_to_print)
        dh.print_to_file(text_to_print, log_file)

    all_end = timer()

    min_hr, min_mins, min_secs = dh.get_time_from_secs(min(cos_time))
    avg_hr, avg_mins, avg_secs = dh.get_time_from_secs(sum(cos_time) / len(cos_time))
    max_hr, max_mins, max_secs = dh.get_time_from_secs(max(cos_time))
    sum_hr, sum_mins, sum_secs = dh.get_time_from_secs(sum(cos_time))
    text_to_print = f'Cosine Similarity time:\n' \
                    f'Min time: {min_hr}hr {min_mins}min {min_secs:.5f}secs.\n' \
                    f'Avg time: {avg_hr}hr {avg_mins}min {avg_secs:.5f}secs.\n' \
                    f'Max time: {max_hr}hr {max_mins}min {max_secs:.5f}secs.\n' \
                    f'Total time: {sum_hr}hr {sum_mins}min {sum_secs:.5f}secs.\n'
    print(text_to_print)
    dh.print_to_file(text_to_print, log_file)

    hrs, mins, secs = dh.get_time_from_secs(all_end - all_start)

    text_to_print = f'Time for all iterations: {hrs}hr {mins}min {secs:.5f}sec\n'
    print(text_to_print)
    dh.print_to_file(text_to_print, log_file)
    dh.print_to_file('*******************************************************************\n', log_file)
    print()
