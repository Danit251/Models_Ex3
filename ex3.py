# Students	Refael  Greenfeld    Danit   Yshaayahu  305030868   312434269
import sys
import os
from collections import Counter
import numpy as np
from math import e
from datetime import datetime
import logging
from time import time


now = datetime.now().strftime("%d_%H_%M_%S")
if not os.path.isdir('logs'):
    os.mkdir('logs')
file_handler = logging.FileHandler(filename=os.path.join('logs', f'log_{now}'))
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
)


class Doc:
    def __init__(self, doc, topics, common_words):
        self.doc = doc
        self.filtered_doc = self.filtered_doc(common_words)
        self.topics = topics
        self.words_stat = self.count_words()
        self.len = self.get_doc_len()

    def count_words(self):
        return Counter(self.filtered_doc)

    def filtered_doc(self, common_words):
        return [w for w in self.doc.split() if w in common_words]

    def get_doc_len(self):
        return sum(self.words_stat.values())


class Em:
    K = 10
    LAMBDA = 0.06
    EPSILON = 10**(-6)

    def __init__(self, path_data, path_topics):
        self.logger = logging.getLogger()
        self.data, self.words_stat = self.load_data(path_data)
        self.logger.info("data loaded")
        self.i2topic, self.topic2i = self.load_topics(path_topics)
        self.logger.info("topics loaded")
        self.num_topics = len(self.topic2i)
        self.num_docs = len(self.data)
        self.voc_len = len(self.words_stat)
        self.word2index = {word: i for i, word in enumerate(self.words_stat)}
        self.logger.info("init finished")

    def load_data(self, path):
        with open(path) as f:
            lines = f.readlines()
            all_topics_lines = lines[0::4]
            docs = lines[2::4]
            words_stat = {x: count for x, count in Counter(" ".join(docs).split()).items() if count > 3}
            data = []
            for i, (topics, doc) in enumerate(zip(all_topics_lines, docs)):
                clean_topics = topics[6:-2].split()[1:]
                data.append(Doc(doc.strip(), clean_topics, words_stat.keys()))
            return data, words_stat

    def load_topics(self, path):
        with open(path) as f:
            lines = f.readlines()
            topics = lines[0::2]
            topics_dict = {i: top.strip() for i, top in enumerate(topics)}
            return topics_dict, {v: k for k, v in topics_dict.items()}

    def calculate_w(self, z: np.ndarray):
        w_matrix = np.ndarray((self.num_topics, self.num_docs), dtype='float64')

        print(f"The vector of the topic of the 0 doc: {len(z[:,0])}. The result is OK? {len(z[:,0]) == self.num_topics}")

        for t, doc in enumerate(self.data):
            z_doc = z[:, t]
            m = z_doc.max()

            all_not_zero = 0
            for i in range(self.num_topics):
                if z_doc[i] - m < - self.K:
                    w_matrix[i][t] = 0
                else:
                    numerator = e ** (z_doc[i] - m)
                    all_not_zero += numerator
                    w_matrix[i][t] = numerator
            w_matrix[:, t] = w_matrix[:, t] / all_not_zero
        return w_matrix

    def initialize_alpha(self, num_topics):
        return np.array([1/num_topics]*num_topics)

    def calculate_p(self, num_topics, w):
        start = time()
        p_matrix = np.zeros((num_topics, len(self.words_stat)), dtype='float64')
        self.logger.info(f"the len of word {len(self.words_stat)}")
        self.logger.info(f"the len of data {len(self.data)}")

        for t, doc in enumerate(self.data):
            for word in doc.words_stat:
                word_index = self.word2index[word]
                n_t_k = doc.words_stat[word]
                p_matrix[:, word_index] += (w[:, t] * n_t_k)
        p_matrix += self.LAMBDA
        dominator = np.full(self.num_topics, self.LAMBDA * self.voc_len)

        for t, doc in enumerate(self.data):
            dominator += (w[:, t] * doc.len)
        for k in range(self.voc_len):
            try:
                p_matrix[:, k] /= dominator
            except Exception as warn:
                self.logger.error(f"the dominator is {dominator}")
        self.logger.info(f"time calculate p: {time() - start}")
        return p_matrix

    def calculate_alpha(self, w):
        alpha = np.zeros(len(self.topic2i))
        for i in range(len(self.topic2i)):
            alpha_value = np.sum(w[i, :])
            if alpha_value == 0:
                alpha_value = self.EPSILON

            alpha[i] = alpha_value

        alpha = self.normalize_alpha(alpha)

        return alpha

    def normalize_alpha(self, alpha):
        all_alpha_sum = np.sum(alpha)
        for j in range(len(alpha)):
            alpha[j] = alpha[j] / all_alpha_sum
        return alpha

    def initialize_w(self, num_topics, num_docs):
        w_matrix = np.zeros((num_topics, num_docs), dtype='float64')

        for i in range(num_topics):
            for j in range(num_docs):
                if j % num_topics == i:
                    w_matrix[i][j] = 1
        return w_matrix

    def calculate_z(self, alpha: np.array, p: np.ndarray) -> np.ndarray:
        z_matrix = np.ndarray((self.num_topics, self.num_docs), dtype='float64')
        for t, doc in enumerate(self.data):
            for i in range(self.num_topics):
                for word in doc.words_stat:
                    word_index = self.word2index[word]
                    n_t_k = doc.words_stat[word]
                    z_matrix[i][t] += n_t_k * np.log(p[i][word_index])
                z_matrix[i][t] += np.log(alpha[i])
        return z_matrix

    def get_n_t_k(self, doc, word):
        n_t_k = 0
        if word in doc.words_stat:
            n_t_k = doc.words_stat[word]
        return n_t_k

    def run_em(self):
        self.logger.info("start em")
        w = self.initialize_w(self.num_topics, self.num_docs)
        self.logger.info("finish initialize w")
        alpha = self.initialize_alpha(self.num_topics)
        self.logger.info("finish initialize alph")
        p = self.calculate_p(self.num_topics, w)
        self.logger.info("finish initialize p")

        for i in range(3):
            self.logger.info(f"start iteration number {str(i)}")
            # E
            z = self.calculate_z(alpha, p)
            self.logger.info("finish calculate z")
            w = self.calculate_w(z)
            self.logger.info("finish calculate w")

            # M
            alpha = self.calculate_alpha(w)
            self.logger.info("finish calculate alpha")
            p = self.calculate_p(self.num_topics, w)
            self.logger.info("finish calculate p")

    def get_likelihood(self, z):
        sum_l = 0
        for t in range(len(self.data)):

            m_t = z[:, t].max()
            sum_ln = 0
            for i in range(self.num_topics):
                if z[i][t] - m_t >= - self.K:
                    sum_ln += e**(z[i][t] - m_t)

            sum_l += m_t + np.log(sum_ln)

        return sum_l

    def get_clusters(self):
        pass


def main():
    data_path = sys.argv[1]
    topic_path = sys.argv[2]
    algo = Em(data_path, topic_path)
    algo.run_em()


if __name__ == '__main__':
    main()
