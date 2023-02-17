from collections import defaultdict

import numpy as np


class Member:
    def __init__(self, r_d, label, doc_id):
        self._r_d = r_d
        self._label = label
        self._doc_id = doc_id


class Cluster:
    def __init__(self):
        self._centroid = None
        self._members = []

    def reset(self):
        self._members = []

    def add_member(self, member):
        self._members.append(member)

    @property
    def centroid(self):
        return self._centroid


class Kmeans:
    def __init__(self, num_clusters):
        self._num_clusters = num_clusters
        self._clusters = [Cluster() for _ in range(num_clusters)]
        self._E = []  # list of centroids
        self._S = 0  # overall similarity

        self._iteration = None # iterations until convergence

    def load_data(self, data_path):
        def sparse_to_dense(sparse_r_d, vocab_size):  # from sparse data of tfidf to array
            _r_d = [0.0 for _ in range(vocab_size)]
            indices_tfidfs = sparse_r_d.split()
            for index_tfidf in indices_tfidfs:
                index = int(index_tfidf.split(":")[0])
                tfidf = float(index_tfidf.split(":")[1])
                _r_d[index] = tfidf
            _r_d = np.array(_r_d)
            return _r_d

        print("start loading data")
        with open(data_path) as f:
            d_lines = f.read().splitlines()
        with open("../datasets/20news-bydate/word_idfs.txt") as f:
            vocab_size = len(f.read().splitlines())
        self._data: list[Member] = []
        self._label_count = defaultdict(int)
        for d in d_lines:
            features = d.split("<fff>")
            label = int(features[0])
            doc_id = int(features[1])
            self._label_count[label] += 1
            r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
            self._data.append(Member(r_d, label, doc_id))
        print("data loaded")

    def random_init(self, seed_value):
        np.random.seed(seed_value)
        total_docs = (len(self._data))
        for i in range(self._num_clusters):
            centroid_idx = np.random.randint(0, total_docs)
            self._E.append(self._data[centroid_idx]._r_d)
            self._clusters[i]._centroid = self._data[centroid_idx]._r_d
            print(self._clusters[i]._centroid)

    def compute_similarity(self, member, centroid):
        sqr_dist = np.sum((member._r_d - centroid) ** 2)
        if sqr_dist == 0:
            return np.inf
        return -np.log10(sqr_dist)

    def select_cluster_for(self, member):
        best_fit_cluster = None
        max_similarity = -1
        for cluster in self._clusters:
            similarity = self.compute_similarity(member, cluster.centroid)
            if similarity > max_similarity:
                best_fit_cluster = cluster
                max_similarity = similarity
        best_fit_cluster.add_member(member)
        return max_similarity

    def update_centroid_of(self, cluster):
        if len(cluster._members) == 0: return
        if len(cluster._members) == 1: 
            cluster._centroid = cluster._members[0]._r_d
            return
        member_r_ds = [member._r_d for member in cluster._members]
        aver_r_d = np.mean(member_r_ds, axis=0)  # midpoint of cluster

        cluster._centroid = aver_r_d
        
    def stopping_condition(self, criterion, threshold):
        criteria = ["centroid", "similarity", "max_iters"]
        assert criterion in criteria
        if criterion == "max_iters":
            return self._iteration >= threshold
        elif criterion == "centroid":
            _new_E = [list(cluster.centroid) for cluster in self._clusters]
            _new_E_minus_E = [centroid for centroid in _new_E
                              if centroid not in self._E]
            self._E = _new_E
            return len(_new_E_minus_E) <= threshold  # too few centroid is updated
        else:
            _new_S_minus_S = abs(self._new_S - self._S)
            self._S = self._new_S
            return _new_S_minus_S <= threshold  # overall similarity have not changed

    def run(self, seed_value, criterion, threshold):
        self.random_init(seed_value)
        # update clusters until convergence
        self._iteration = 0
        while True:
            print("loop {}".format(self._iteration))
            # reset clusters, retain only centroids
            for cluster in self._clusters:
                cluster.reset()
            self._new_S = 0
            for member in self._data:
                max_similarity = self.select_cluster_for(member)
                self._new_S += max_similarity
            for cluster in self._clusters:
                self.update_centroid_of(cluster)
            self._iteration += 1
            if self.stopping_condition(criterion, threshold): break

    def compute_purity(self):
        majority_sum = 0
        N = len(self._data)
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members]
            max_count = max([member_labels.count(label) for label in range(20)])  # number of the most label in cluster
            majority_sum += max_count
        return majority_sum * 1. / N

    def compute_NMI(self):
        I_value, H_omega, H_C = 0., 0., 0.
        N = len(self._data)
        for cluster in self._clusters:
            wk = len(cluster._members) * 1.
            H_omega += -wk * np.log10(wk/N) / N
            member_labels = [member._label for member in cluster._members]
            for label in range(20):
                wk_cj = member_labels.count(label) * 1.
                cj = self._label_count[label]
                I_value += wk_cj/N * np.log10(N*wk_cj / (wk*cj) + 1e-12)
        for label in range(20):
            cj = self._label_count[label] * 1.
            H_C += -cj/N * np.log10(cj/N)
        return I_value * 2. / (H_C + H_omega)


if __name__ == '__main__':
    kmeans = Kmeans(num_clusters=20)
    kmeans.load_data("../datasets/20news-bydate/tf-idf.txt")
    # kmeans.run(seed_value=128, criterion="max_iters", threshold=3)
    kmeans.random_init(1512)
    print(kmeans._clusters[0]._centroid)
    for i in range(1, 20):
        print(np.array_equal(kmeans._clusters[0]._centroid, kmeans._clusters[i]._centroid))