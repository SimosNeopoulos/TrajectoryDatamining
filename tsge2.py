import networkx as nx
from gensim.models import Word2Vec


def _get_walks(traj_dict):
    return [traj.get_path_list() for traj in traj_dict.values()]


class TSGE2:

    def __init__(self, graph: nx.Graph, dimensions: int = 128):
        self.graph = graph
        self.dimensions = dimensions

    def fit(self, traj_dict: dict) -> dict:
        walks = _get_walks(traj_dict)
        model = Word2Vec(sentences=walks, vector_size=self.dimensions, window=5, min_count=1, sg=1, workers=4)
        node_vectors = model.wv
        graph_nodes = self.graph.nodes
        return {str(node): node_vectors[str(node)].tolist() for node in graph_nodes if str(node) in node_vectors}
