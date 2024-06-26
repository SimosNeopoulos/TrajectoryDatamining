import data_handler as dh
import log_printer as log_pr
import random as rand
from timeit import default_timer as timer
import similarity_mesures as sm
import multiprocessing as mp

main_out = r'main_out.txt'
testing_out = r'testing_out.txt'
testing2_out = r'testing2_out.txt'
cities = ['oldenburg', 'el_dorado']  # , 'san_francisco', 'anchorage', 'knox_county'

dims = ['64', '128', '256']


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
        log_pr.print_to_file(f'Start of {traj_id}', main_out)
        start = timer()
        sim = sm.k_dist_sim(traj_dict[traj_id], traj_dict, pair_dict, k)
        end = timer()
        log_pr.print_to_file(f'End of {traj_id} time: {dh.get_time_from_secs(end - start)}', main_out)
        dist_data_half.append((sim, end - start))
    results_queue.put(dist_data_half)


def get_dist_data(traj_dict: dict, rand_trajectories: list, pair_dict: dict, k: int) -> list:
    dist_data = []
    for traj_id in rand_trajectories:
        print(f'Start of {traj_id}')
        start = timer()
        sim = sm.k_dist_sim(traj_dict[traj_id], traj_dict, pair_dict, k)
        end = timer()
        print(f'End of {traj_id} time: {dh.get_time_from_secs(end - start)}')
        dist_data.append((sim, end - start))

    return dist_data


def run_print_sim(emb_dict, dist_data, traj_dict, rand_trajectories, sim_method, pagerank, log_file):
    log_pr.print_to_file('Inside run_print_sim', main_out)
    log_pr.print_sim(emb_dict=emb_dict,
                     dist_data=dist_data,
                     traj_dict=traj_dict,
                     rand_traj=rand_trajectories,
                     sim_method=sim_method,
                     log_file=log_file,
                     error_log_file=main_out,
                     pagerank=pagerank)


def get_mock_dist_data():
    return [(((i, i+1), i / 10), i) for i in range(20)]


def main():
    log_pr.print_to_file('Inside main', main_out)
    city_paths = dh.load_city_paths()

    for city in cities:
        log_pr.print_to_file(f'City {city} iteration started', main_out)
        print(f'City {city} iteration started')
        graph = dh.load_graph(city_paths[city]['graph2'])
        pagerank = dh.load_pagerank(city_paths[city]['pagerank2'])
        log_pr.print_to_file(f'Graph {city} loaded', main_out)
        print(f'Graph {city} loaded')
        graph_str = graph.__str__()
        del graph

        pair_dict = dh.load_graph_pairs(city_paths[city]['pairs2'])
        log_pr.print_to_file(f'Pairs {city} loaded', main_out)
        print(f'Pairs {city} loaded')
        traj_dict = dh.get_trajectory_catalog(city_paths[city]['dat_file2'])
        log_pr.print_to_file(f'Trajectories {city} loaded', main_out)
        print(f'Trajectories {city} loaded')
        traj_num = len(traj_dict)
        rand_trajectories = get_rand_trajectories(traj_num)

        log_pr.print_to_file(f'{city} inside dist_data', main_out)
        start = timer()
        dist_data = get_dist_data(traj_dict, rand_trajectories, pair_dict, 1000)
        # dist_data = get_mock_dist_data()
        end = timer()
        log_pr.print_to_file(f'Time for dist data: {dh.get_time_from_secs(end - start)}', main_out)

        del pair_dict
        dim = dims[1]
        # for dim in dims:
        # log_pr.print_to_file(f'Dimension {dim} iteration started', main_out)
        processes = []
        ########################################## Normal n2v ####################################################
        node_emb = dh.load_emb(city_paths[city][dim]['emb2'])
        log_pr.print_graph_name(graph_str, city, city_paths[city][dim]['pagerank_log_sim2'], main_out)
        n2v_pagerank_process = mp.Process(target=run_print_sim, args=(node_emb, dist_data, traj_dict, rand_trajectories, 'node_emb', pagerank, city_paths[city][dim]['pagerank_log_sim2']))
        processes.append(n2v_pagerank_process)
        log_pr.print_to_file(f'Process node2vec pagerank initialised', main_out)

        log_pr.print_graph_name(graph_str, city, city_paths[city][dim]['log_sim2'], main_out)
        n2v_process = mp.Process(target=run_print_sim, args=(node_emb, dist_data, traj_dict, rand_trajectories, 'node_emb', None, city_paths[city][dim]['log_sim2']))
        processes.append(n2v_process)
        del node_emb
        log_pr.print_to_file(f'Process node2vec initialised', main_out)
        ##########################################################################################################

        ######################################### Biased n2v ####################################################
        biased_node_emb = dh.load_emb(city_paths[city][dim]['biased_emb2'])
        log_pr.print_graph_name(graph_str, city, city_paths[city][dim]['pagerank_log_biased_sim2'], main_out)
        bn2v_pagerank_process = mp.Process(target=run_print_sim, args=(biased_node_emb, dist_data, traj_dict, rand_trajectories, 'node_emb', pagerank, city_paths[city][dim]['pagerank_log_biased_sim2']))
        processes.append(bn2v_pagerank_process)
        log_pr.print_to_file(f'Process biased_node2vec pagerank initialised', main_out)

        log_pr.print_graph_name(graph_str, city, city_paths[city][dim]['log_biased_sim2'], main_out)
        bn2v_process = mp.Process(target=run_print_sim, args=(biased_node_emb, dist_data, traj_dict, rand_trajectories, 'node_emb', None, city_paths[city][dim]['log_biased_sim2']))
        processes.append(bn2v_process)
        del biased_node_emb
        log_pr.print_to_file(f'Process biased_node2vec initialised', main_out)
        ########################################################################################################

        ########################################## Traj2Vec ####################################################
        # traj_emb = dh.load_emb(city_paths[city][dim]['traj_emb'])
        # log_pr.print_graph_name(graph_str, city, city_paths[city][dim]['log_traj_sim'], main_out)
        # bn2v_process = mp.Process(target=run_print_sim, args=(traj_emb, dist_data, traj_dict, rand_trajectories, 'traj_emb', city_paths[city][dim]['log_traj_sim']))
        # processes.append(bn2v_process)
        # del traj_emb
        # log_pr.print_to_file(f'Process traj2vec initialised', main_out)
        ######################################################################################################

        for process in processes:
            process.start()

        for process in processes:
            process.join()


if __name__ == '__main__':
    main()
