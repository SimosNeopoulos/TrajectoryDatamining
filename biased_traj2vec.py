from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# This method isn't used since the results it yielded were the the worst by far
class BiasedTraj2Vec:

    def __init__(self, traj_dict: dict, vector_size: int = 128, window: int = 5, min_count: int = 1, workers: int = 5, epochs: int = 20):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.tagged_walks = [TaggedDocument(words=traj.get_path_list(), tags=[str(traj_id)]) for traj_id, traj in traj_dict.items()]

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

        return {int(traj_id): self.model.infer_vector(traj_dict[traj_id].get_path_list()).tolist() for traj_id in traj_dict.keys()}
