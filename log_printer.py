import data_handler as dh
import similarity_measures as sm
from timeit import default_timer as timer
from trajectory import Trajectory
import multiprocessing

file_lock = multiprocessing.Lock()


def get_city_name(name: str) -> str:
    return name.replace('_', ' ').title()


def print_to_file(text: str, file_path: str):
    with file_lock:
        with open(file_path, 'a') as file:
            file.write(text + '\n')


def print_graph_name(graph_str: str, city: str, log_file: str):
    city_name = f'{get_city_name(city)}:\n'

    print_to_file(city_name, log_file)

    print_to_file(graph_str + '\n', log_file)


def log_trajectory_information(traj_dict: dict, log_file: str):
    catalog_num = len(traj_dict)
    text_to_print = f'Number of trajectories: {catalog_num}'
    print_to_file(text_to_print, log_file)

    trajectory_length_list = []
    for trajectory in traj_dict.values():
        trajectory_length_list.append(len(trajectory.trajectory_path))

    text_to_print = f'Min length of trajectories: {min(trajectory_length_list)}'
    print_to_file(text_to_print, log_file)
    text_to_print = f'Avg length of trajectories: {sum(trajectory_length_list) / catalog_num}'
    print_to_file(text_to_print, log_file)
    text_to_print = f'Max length of trajectories: {max(trajectory_length_list)}\n'
    print_to_file(text_to_print, log_file)


def compute_cosine_similarity_and_time(main_traj: Trajectory, traj_dict: dict, emb_dict: dict, k: int, sim_method: str, pagerank: dict | None) -> tuple[list, float]:
    start = timer()
    sim = sm.cosine_similarity_measure(main_traj, traj_dict, emb_dict, k, sim_method, pagerank)
    end = timer()

    return sim, end - start


def log_similarity_results(emb_dict: dict, dist_data: list, traj_dict: dict, rand_traj: list, sim_method: str, log_file: str, pagerank: dict | None):
    k_10_list = []
    k_100_list = []
    k_1000_list = []

    dist_time = []
    cos_time = []

    k = 1000
    all_start = timer()
    for i in range(20):
        start = timer()

        dist_sim, dist_thread_time = dist_data[i]
        cos_sim, cos_thread_time = compute_cosine_similarity_and_time(traj_dict[rand_traj[i]], traj_dict, emb_dict, k, sim_method, pagerank)

        dist_time.append(dist_thread_time)

        cos_time.append(cos_thread_time)

        dist10 = set()
        cos10 = set()
        dist100 = set()
        cos100 = set()
        dist1000 = set()
        cos1000 = set()
        for index, (dist_sim, cos_sim) in enumerate(zip(dist_sim, cos_sim)):

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
        hrs, mins, secs = dh.format_time(end - start)
        dist_hrs, dist_mins, dist_secs = dh.format_time(dist_thread_time)
        cos_hrs, cos_mins, cos_secs = dh.format_time(cos_thread_time)
        text_to_print = f'Iteration: {i + 1}. Time of iteration: {hrs}hr {mins}min {secs:.5f}secs.' \
                        f' Distance Similarity time: {dist_hrs}hr {dist_mins}min {dist_secs:.5f}secs.' \
                        f' Cosine Similarity time: {cos_hrs}hr {cos_mins}min {cos_secs:.5f}secs.' \
                        f' Trajectory of iteration: {rand_traj[i]}.' \
                        f' k=10 {k_10_sim}/10, k=100 {k_100_sim}/100, k=1000 {k_1000_sim}/1000.'

        print_to_file(text_to_print, log_file)

    all_end = timer()

    text_to_print = f'\nFor k=10:\n' \
                    f'Min similarity: {min(k_10_list)}/10\n' \
                    f'Avg similarity: {sum(k_10_list) / len(k_10_list)}/10\n' \
                    f'Max similarity: {max(k_10_list)}/10\n'
    print_to_file(text_to_print, log_file)

    text_to_print = f'For k=100:\n' \
                    f'Min similarity: {min(k_100_list)}/100\n' \
                    f'Avg similarity: {sum(k_100_list) / len(k_100_list)}/100\n' \
                    f'Max similarity: {max(k_100_list)}/100\n'
    print_to_file(text_to_print, log_file)

    text_to_print = f'For k=1000:\n' \
                    f'Min similarity: {min(k_1000_list)}/1000\n' \
                    f'Avg similarity: {sum(k_1000_list) / len(k_1000_list)}/1000\n' \
                    f'Max similarity: {max(k_1000_list)}/1000\n'
    print_to_file(text_to_print, log_file)

    min_hr, min_mins, min_secs = dh.format_time(min(dist_time))
    avg_hr, avg_mins, avg_secs = dh.format_time(sum(dist_time) / len(dist_time))
    max_hr, max_mins, max_secs = dh.format_time(max(dist_time))
    sum_hr, sum_mins, sum_secs = dh.format_time(sum(dist_time))
    text_to_print = f'Distance Similarity time:\n' \
                    f'Min time: {min_hr}hr {min_mins}min {min_secs:.5f}secs.\n' \
                    f'Avg time: {avg_hr}hr {avg_mins}min {avg_secs:.5f}secs.\n' \
                    f'Max time: {max_hr}hr {max_mins}min {max_secs:.5f}secs.\n' \
                    f'Total time: {sum_hr}hr {sum_mins}min {sum_secs:.5f}secs.\n'
    print_to_file(text_to_print, log_file)

    min_hr, min_mins, min_secs = dh.format_time(min(cos_time))
    avg_hr, avg_mins, avg_secs = dh.format_time(sum(cos_time) / len(cos_time))
    max_hr, max_mins, max_secs = dh.format_time(max(cos_time))
    sum_hr, sum_mins, sum_secs = dh.format_time(sum(cos_time))
    text_to_print = f'Cosine Similarity time:\n' \
                    f'Min time: {min_hr}hr {min_mins}min {min_secs:.5f}secs.\n' \
                    f'Avg time: {avg_hr}hr {avg_mins}min {avg_secs:.5f}secs.\n' \
                    f'Max time: {max_hr}hr {max_mins}min {max_secs:.5f}secs.\n' \
                    f'Total time: {sum_hr}hr {sum_mins}min {sum_secs:.5f}secs.\n'
    print_to_file(text_to_print, log_file)

    hrs, mins, secs = dh.format_time(all_end - all_start)

    text_to_print = f'Time for all iterations: {hrs}hr {mins}min {secs:.5f}sec\n'
    print_to_file(text_to_print, log_file)
    print_to_file('*******************************************************************\n', log_file)
    print()


def get_dist_data(traj_dict: dict, rand_traj: int, pair_dict: dict, k: int = 1000) -> tuple[list, float]:
    print(f'Start of {rand_traj}')
    start = timer()
    sim = sm.top_k_distance_similarity(traj_dict[rand_traj], traj_dict, pair_dict, k)
    end = timer()
    print(f'End of {rand_traj} time: {dh.format_time(end - start)}')

    return sim, end - start
