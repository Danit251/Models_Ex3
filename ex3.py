# Students	Refael  Greenfeld    Danit   Yshaayahu  305030868   312434269
import sys
from collections import Counter
import numpy as np
from math import e
from datetime import datetime
import logging
from time import time

now = datetime.now().strftime("%H_%M_%S")
logging.basicConfig(filename=f"log_{now}", level=logging.INFO)
logger = logging.getLogger()


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
        logger.info("data loaded")
        self.data, self.words_stat = self.load_data(path_data)
        logger.info("data loaded")
        self.i2topic, self.topic2i = self.load_topics(path_topics)
        logger.info("topics loaded")
        self.num_topics = len(self.topic2i)
        self.num_docs = len(self.data)
        self.splited_data = self.initial_split(self.num_topics)
        logger.info("init finished")

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

    def initial_split(self, num_topics):
        splited_data = [[] for _ in range(num_topics)]
        for i in range(len(self.data)):
            mod = i % num_topics
            splited_data[mod].append(self.data[i])
        return splited_data

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
        p_matrix = np.ndarray((num_topics, len(self.words_stat)), dtype='float64')

        # for each topic
        for i in range(num_topics):

            sum_numerator = 0
            sum_denominator = 0

            # for each word
            for k, word in enumerate(self.words_stat):

                # for each document
                for t, doc in enumerate(self.data):
                    n_t_k = self.get_n_t_k(doc, word)
                    sum_numerator += w[i][t]*n_t_k
                    sum_denominator += w[i][t]*doc.len

                sum_numerator += self.LAMBDA
                sum_denominator += self.LAMBDA*len(self.words_stat)

                p_matrix[i][k] = sum_numerator/sum_denominator
        logger.info(f"time calculate p: {time() - start}")
        return p_matrix

    def calculate_alpha(self, w):
        alpha = np.zeros(len(self.topic2i))
        for i in range(len(self.topic2i)):
            print(f"topic line length: {len(w[i, :])}")

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
        w_matrix = np.ndarray((num_topics, num_docs), dtype='float64')

        for i in range(num_topics):
            for j in range(num_docs):
                if j % num_topics == i:
                    w_matrix[i][j] = 1
        return w_matrix

    def calculate_z(self, alpha: np.array, p: np.ndarray) -> np.ndarray:
        z_matrix = np.ndarray((self.num_topics, self.num_docs), dtype='float64')
        for t, doc in enumerate(self.data):
            for i in range(self.num_topics):
                for k, word in enumerate(self.words_stat):
                    n_t_k = self.get_n_t_k(doc, word)
                    z_matrix[i][t] += n_t_k * np.log(p[i][k])
                z_matrix[i][t] += np.log(alpha[i])
        return z_matrix

    def get_n_t_k(self, doc, word):
        n_t_k = 0
        if word in doc.words_stat:
            n_t_k = doc.words_stat[word]
        return n_t_k

    def run_em(self):
        logger.info("start em")
        w = self.initialize_w(self.num_topics, self.num_docs)
        logger.info("finish initialize w")
        alpha = self.initialize_alpha(self.num_topics)
        logger.info("finish initialize alph")
        p = self.calculate_p(self.num_topics, w)
        logger.info("finish initialize p")

        for i in range(3):
            logger.info(f"start iteration number {str(i)}")
            # E
            z = self.calculate_z(alpha, p)
            logger.info("finish calculate z")
            w = self.calculate_w(z)
            logger.info("finish calculate w")

            # M
            alpha = self.calculate_alpha(w)
            logger.info("finish calculate alpha")
            p = self.calculate_p(self.num_topics, w)
            logger.info("finish calculate p")

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
