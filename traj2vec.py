from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class Traj2Vec:

    def __init__(self, vec_dict: list, vector_size: int = 128, window: int = 5, min_count: int = 1, workers: int = 5, epochs: int = 1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.tagged_walks = [TaggedDocument(words=walk, tags=[str(i)]) for i, walk in enumerate(vec_dict)]

        self.model = Doc2Vec(vector_size=self.vector_size,
                             window=self.window,
                             min_count=self.min_count,
                             workers=self.workers,
                             epochs=self.epochs
                             )

    def train_model(self):
        self.model.build_vocab(self.tagged_walks)
        self.model.train(self.tagged_walks, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def get_traj_emb(self, traj_dict):
        if self.model.corpus_count == 0:
            raise ValueError("Model has not been trained. Call train_model() first.")

        return {str(traj_id): self.model.infer_vector(traj_dict[traj_id].get_path_list()).tolist() for traj_id in traj_dict.keys()}
