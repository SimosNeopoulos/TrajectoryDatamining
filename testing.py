from node2vec import Node2Vec as n2v
from karateclub.node_embedding.neighbourhood.node2vec import Node2Vec
import data_handler as dh
import similarity_mesures as sm
import random as rand
from timeit import default_timer as timer
from main import get_city_name
import networkx as nx


def n2vec(graph, city):
    start = timer()
    node2vec = n2v(graph)
    model = node2vec.fit()
    end = timer()

    hrs, mins, secs = dh.get_time_from_secs(end - start)
    dh.print_to_file(get_city_name(city) + ':\n')
    text_to_print = f'Time for Node2Vec embedding: {hrs}hr {mins}min {secs:.5f}sec\n'
    dh.print_to_file(text_to_print, f'./data_files/new_construct_data.txt')
    print(text_to_print)


def main():
    city_paths = dh.load_city_paths()
    cities = ['oldenburg', 'san_francisco', 'anchorage', 'el_dorado', 'knox_county']
    for city in cities:
        graph, _, _ = dh.load_graph_data(city_paths[city]['data'])
        n2vec(graph, city)

    # print(cities[:2])
    # graph, node_index, emb_dict = dh.load_graph_data(city_paths[cities[4]]['data'])
    # pair_dict = dh.load_graph_pairs(city_paths[cities[4]]['pairs'])
    # print(18691 in node_index.values())
    # print(18691 in node_index.keys())
    # print(18691 in pair_dict.keys())
    # print(18691 in emb_dict.keys())

    # graph = nx.Graph()
    #
    # for i in range(1, 10):
    #     graph.add_edge(i - 1, i)
    #
    # node2vec = Node2Vec(walk_length=5)
    # node2vec.fit(graph)
    # node_vectors = node2vec.get_embedding()
    # node_arr = []
    # for i in range(10):
    #     print(i, sm.cosine_similarity(list(node_vectors[0]), list(node_vectors[i])))
    #     node_arr.append((i, sm.cosine_similarity(list(node_vectors[0]), list(node_vectors[i]))))
    #
    # node2vec = n2v(graph, walk_length=5)
    # model = node2vec.fit()
    # node_vectors = model.wv
    # print()
    #
    # for i in range(10):
    #     print(i, sm.cosine_similarity(list(node_vectors[0]), list(node_vectors[i])))


if __name__ == '__main__':
    main()

#
# node2vec = Node2Vec(graph)
#
# # Εκπαίδευση του Node2Vec
# model = node2vec.fit()
#
# print(model.wv.most_similar('5'))
#
# # Αποκτούμε τα διανύσματα των κόμβων από το εκπαιδευμένο μοντέλο
# node_vectors = model.wv
#
# # Τώρα μπορούμε να αναζητήσουμε το διάνυσμα ενός κόμβου, για παράδειγμα του κόμβου '1'
# vector_of_node1 = node_vectors[0]
# vector_of_node2 = node_vectors[0]
#
# # Εκτύπωση του διανύσματος του κόμβου '1'
# print(vector_of_node1 == vector_of_node2)
# print(vector_of_node1)
# print(vector_of_node2)

# cities_paths = dh.load_city_paths()
#
# print(cities_paths['california']['dat_file'])
