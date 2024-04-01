import data_handler as dh
import log_printer as log_pr
import random as rand
from timeit import default_timer as timer
import similarity_mesures as sm
import multiprocessing as mp
import concurrent.futures
from itertools import repeat

# Assuming traj_dict, pair_dict, and k are defined



cities = ['oldenburg', 'san_francisco', 'anchorage', 'knox_county', 'el_dorado', 'california']

dims = ['64', '128', '256']


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

def get_rand_trajectories(catalog_num):
    rand_trajectories = []
    for _ in range(20):
        while True:
            rand_traj = rand.randint(0, catalog_num)
            if rand_traj not in rand_trajectories:
                break
        rand_trajectories.append(rand_traj)
    return rand_trajectories


def process_dist_sim(traj_dict, pair_dict, half, k, results_queue):
    dist_data_half = []
    for traj_id in half:
        print(f'Start of {traj_id}')
        start = timer()
        sim = sm.k_dist_sim(traj_dict[traj_id], traj_dict, pair_dict, k)
        end = timer()
        print(f'End of {traj_id} time: {dh.get_time_from_secs(end - start)}')
        dist_data_half.append((sim, end - start))
    results_queue.put(dist_data_half)


# def get_dist_data(traj_dict: dict, rand_trajectories: list, pair_dict: dict, k: int):
#     mid = len(rand_trajectories) // 2
#     first_half = rand_trajectories[:mid]
#     second_half = rand_trajectories[mid:]
#
#     results_queue1 = mp.Queue()
#     results_queue2 = mp.Queue()
#
#     process1 = mp.Process(target=process_dist_sim, args=(traj_dict, pair_dict, first_half, k, results_queue1))
#     process2 = mp.Process(target=process_dist_sim, args=(traj_dict, pair_dict, second_half, k, results_queue2))
#
#     process1.start()
#     process2.start()
#
#     process1.join()
#     process2.join()
#
#     dist_data_first_half = results_queue1.get()
#     dist_data_second_half = results_queue2.get()
#
#     dist_data = dist_data_first_half + dist_data_second_half
#
#     return dist_data


# def get_dist_data(traj_id, traj_dict, pair_dict, k):
#     print(f'Start of {traj_id}')
#     start = timer()
#     sim = sm.k_dist_sim(traj_dict[traj_id], traj_dict, pair_dict, k)
#     end = timer()
#     print(f'End of {traj_id} time: {dh.get_time_from_secs(end-start)}')
#     return sim, end - start


def get_dist_data(traj_dict: dict, rand_trajectories: list, pair_dict: dict, k: int):
    dist_data = []
    for traj_id in rand_trajectories:
        print(f'Start of {traj_id}')
        start = timer()
        sim = sm.k_dist_sim(traj_dict[traj_id], traj_dict, pair_dict, k)
        end = timer()
        print(f'End of {traj_id} time: {dh.get_time_from_secs(end - start)}')
        dist_data.append((sim, end - start))

    return dist_data


def run_print_sim(emb_dict, dist_data, traj_dict, rand_trajectories, sim_method, log_file):
    log_pr.print_sim(emb_dict=emb_dict,
                     dist_data=dist_data,
                     traj_dict=traj_dict,
                     rand_traj=rand_trajectories,
                     sim_method=sim_method,
                     log_file=log_file)


def main():
    city_paths = dh.load_city_paths()
    # manager = mp.Manager()

    for city in cities[2:-1]:
        graph = dh.load_graph(city_paths[city]['graph'])
        graph_str = graph.__str__()
        del graph

        pair_dict = dh.load_graph_pairs(city_paths[city]['pairs'])
        traj_dict = dh.get_trajectory_catalog(city_paths[city]['dat_file'])
        traj_num = len(traj_dict)
        rand_trajectories = get_rand_trajectories(traj_num)

        print(f'{city} inside dist_data')

        start = timer()
        dist_data = get_dist_data(traj_dict, rand_trajectories, pair_dict, 1000)
        # with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        #     dist_data = list(executor.map(get_dist_data, rand_trajectories, repeat(traj_dict), repeat(pair_dict), repeat(1000)))
        end = timer()
        print(f'Time for dist data: {dh.get_time_from_secs(end - start)}')
        print(dist_data)

        del pair_dict

        for dim in dims:
            processes = []
            ########################################## Normal n2v ####################################################
            node_emb = dh.load_emb(city_paths[city][dim]['emb'])
            log_pr.print_graph_name(graph_str, city, city_paths[city][dim]['log_sim'])
            n2v_process = mp.Process(target=run_print_sim, args=(node_emb, dist_data, traj_dict, rand_trajectories, 'node_emb', city_paths[city][dim]['log_sim']))
            processes.append(n2v_process)
            del node_emb
            ##########################################################################################################

            ########################################## Biased n2v ####################################################
            biased_node_emb = dh.load_emb(city_paths[city][dim]['biased_emb'])
            log_pr.print_graph_name(graph_str, city, city_paths[city][dim]['log_biased_sim'])
            bn2v_process = mp.Process(target=run_print_sim, args=(biased_node_emb, dist_data, traj_dict, rand_trajectories, 'node_emb', city_paths[city][dim]['log_biased_sim']))
            processes.append(bn2v_process)
            del biased_node_emb
            ########################################################################################################

            ########################################## Traj2Vec ####################################################
            traj_emb = dh.load_emb(city_paths[city][dim]['traj_emb'])
            log_pr.print_graph_name(graph_str, city, city_paths[city][dim]['log_traj_sim'])
            bn2v_process = mp.Process(target=run_print_sim, args=(traj_emb, dist_data, traj_dict, rand_trajectories, 'traj_emb', city_paths[city][dim]['log_traj_sim']))
            processes.append(bn2v_process)
            del traj_emb
            #######################################################################################################

            for process in processes:
                process.start()

            for process in processes:
                process.join()


if __name__ == '__main__':
    main()
