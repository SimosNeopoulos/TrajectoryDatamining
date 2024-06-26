import data_handler as dh
import log_printer as log_pr
import random as rand
from timeit import default_timer as timer
import multiprocessing as mp
from main import get_dist_data, run_print_sim
from main import testing2_out

cities = ['oldenburg', 'el_dorado']  # 'san_francisco', 'anchorage', 'knox_county',
dims = [64, 128, 256]


def get_rand_trajectories(catalog_num):
    rand_trajectories = []
    for _ in range(20):
        while True:
            rand_traj = rand.randint(0, catalog_num)
            if rand_traj not in rand_trajectories:
                break
        rand_trajectories.append(rand_traj)
    return rand_trajectories


def main():
    log_pr.print_to_file('Inside main', testing2_out)
    city_paths = dh.load_city_paths()
    # dim = dims[1]

    for city in cities:
        city_name = log_pr.get_city_name(city) + ':\n'
        log_pr.print_to_file(city_name, city_paths[city]['log_construct'])

        graph = dh.load_graph(city_paths[city]['graph2'])
        pagerank = dh.create_pagerank(graph, city_paths[city]['log_construct'])
        dh.save_pagerank(pagerank, city_paths[city]['pagerank2'])


if '__main__' == __name__:
    main()
