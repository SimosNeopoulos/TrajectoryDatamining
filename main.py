import data_handler as dh
import log_printer as log_pr
import random as rand
from timeit import default_timer as timer
import similarity_measures as sm
import multiprocessing as mp

cities = ['oldenburg', 'san_francisco', 'anchorage', 'knox_county', 'el_dorado']
dimensions = ['64', '128', '256']


def get_random_trajectories(catalog_size: int) -> list:
    """Select 20 unique random trajectory indices from the catalog."""
    random_trajectories = []
    for _ in range(20):
        while True:
            random_traj = rand.randint(0, catalog_size - 1)
            if random_traj not in random_trajectories:
                break
        random_trajectories.append(random_traj)
    return random_trajectories


def calculate_distance_data(traj_dict: dict, random_trajectories: list, shortest_paths_dict: dict, k: int) -> list:
    """Calculate top-k distance similarities for random trajectories."""
    distance_data = []
    for traj_id in random_trajectories:
        print(f'Starting calculations for trajectory {traj_id}')
        start = timer()
        similarities = sm.top_k_distance_similarity(traj_dict[traj_id], traj_dict, shortest_paths_dict, k)
        end = timer()
        print(f'Finished trajectory {traj_id} in {dh.format_time(end - start)}')
        distance_data.append((similarities, end - start))
    return distance_data


def run_similarity_printing(embedding_dict, distance_data, traj_dict, random_trajectories, sim_method, pagerank, log_file):
    """Print similarity results to a log file."""
    log_pr.log_similarity_results(
        emb_dict=embedding_dict,
        dist_data=distance_data,
        traj_dict=traj_dict,
        rand_traj=random_trajectories,
        sim_method=sim_method,
        log_file=log_file,
        pagerank=pagerank
    )


def create_graph_and_embeddings(city, traj_dict, city_paths):
    graph = dh.create_graph(city_paths[city]['dat_file'], city_paths[city]['log_construct'])
    dh.save_graph(graph, city_paths[city]['graph'])

    dh.save_shortest_paths(graph, city_paths[city]['pairs'], city_paths[city]['log_construct'])

    for dim in dimensions:
        embedding = dh.create_tsge1_embedding(graph, int(dim), city_paths[city][dim]['log_sim'])
        dh.save_embedding(embedding, city_paths[city][dim]['emb'])
        del embedding

        embedding = dh.create_tsge2_embedding(graph, int(dim), traj_dict, city_paths[city][dim]['log_biased_sim'])
        dh.save_embedding(embedding, city_paths[city][dim]['biased_emb'])
        del embedding

        embedding = dh.create_tsge3_embedding(graph, int(dim), traj_dict, city_paths[city][dim]['log_traj_sim'])
        dh.save_embedding(embedding, city_paths[city][dim]['traj_emb'])
        del embedding


def main():
    city_paths = dh.load_city_paths()

    for city in cities:
        shortest_paths_dict = dh.load_shortest_paths(city_paths[city]['pairs'])
        traj_dict = dh.build_trajectory_catalog(city_paths[city]['dat_file'])
        random_trajectories = get_random_trajectories(len(traj_dict))

        distance_data = calculate_distance_data(traj_dict, random_trajectories, shortest_paths_dict, 1000)
        del shortest_paths_dict  # Release memory after use

        for dim in dimensions:
            processes = []

            ############################################ TSGE1 #####################################################
            tsge1_node_embeddings = dh.load_embedding(city_paths[city][dim]['emb'])
            tsge1_process = mp.Process(
                target=run_similarity_printing,
                args=(tsge1_node_embeddings, distance_data, traj_dict, random_trajectories, 'node_emb', None, city_paths[city][dim]['log_sim'])
            )
            processes.append(tsge1_process)
            ########################################################################################################

            ############################################## TSGE2 ###################################################
            tsge2_node_embeddings = dh.load_embedding(city_paths[city][dim]['biased_emb'])
            tsge2_process = mp.Process(
                target=run_similarity_printing,
                args=(tsge2_node_embeddings, distance_data, traj_dict, random_trajectories, 'node_emb', None, city_paths[city][dim]['log_biased_sim'])
            )
            processes.append(tsge2_process)
            ########################################################################################################

            ############################################ TSGE3 #####################################################
            tsge3_embeddings = dh.load_embedding(city_paths[city][dim]['traj_emb'])
            tsge3_embeddings_process = mp.Process(
                target=run_similarity_printing,
                args=(tsge3_embeddings, distance_data, traj_dict, random_trajectories, 'traj_emb', None, city_paths[city][dim]['log_traj_sim'])
            )
            processes.append(tsge3_embeddings_process)
            ########################################################################################################

            # Start all processes
            for process in processes:
                process.start()

            # Ensure all processes complete
            for process in processes:
                process.join()


if __name__ == '__main__':
    main()
