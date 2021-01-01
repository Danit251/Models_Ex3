# Students	Refael  Greenfeld    Danit   Yshaayahu  305030868   312434269
import sys
import os
from collections import Counter
import numpy as np
from math import e
from datetime import datetime
import logging
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import operator

pd.set_option('display.max_columns', 15)
now = datetime.now().strftime("%d_%H_%M_%S")
if not os.path.isdir('logs'):
    os.mkdir('logs')
file_handler = logging.FileHandler(filename=os.path.join('logs', f'log_{now}'))
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO,
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
        self.pred = None

    def count_words(self):
        return Counter(self.filtered_doc)

    def filtered_doc(self, common_words):
        return [w for w in self.doc.split() if w in common_words]

    def get_doc_len(self):
        return sum(self.words_stat.values())


class Em:
    K = 10
    LAMBDA = 0.15
    EPSILON = 10**(-6)

    def __init__(self, path_data, path_topics):
        self.logger = logging.getLogger()
        self.logger.info(f'The lambda is: {self.LAMBDA}')
        self.docs, self.words_stat = self.load_data(path_data)
        self.i2topic, self.topic2i = self.load_topics(path_topics)
        self.num_topics = len(self.topic2i)
        self.num_docs = len(self.docs)
        self.voc_len = len(self.words_stat)
        self.word2index = {word: i for i, word in enumerate(self.words_stat)}

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

        for t, doc in enumerate(self.docs):
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
        p_matrix = np.zeros((num_topics, len(self.words_stat)), dtype='float64')

        for t, doc in enumerate(self.docs):
            for word in doc.words_stat:
                word_index = self.word2index[word]
                n_t_k = doc.words_stat[word]
                p_matrix[:, word_index] += (w[:, t] * n_t_k)
        p_matrix += self.LAMBDA
        dominator = np.full(self.num_topics, self.LAMBDA * self.voc_len)

        for t, doc in enumerate(self.docs):
            dominator += (w[:, t] * doc.len)
        for k in range(self.voc_len):
            p_matrix[:, k] /= dominator
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
        z_matrix = np.zeros((self.num_topics, self.num_docs), dtype='float64')
        for t, doc in enumerate(self.docs):
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
        w = self.initialize_w(self.num_topics, self.num_docs)
        alpha = self.initialize_alpha(self.num_topics)
        p = self.calculate_p(self.num_topics, w)

        last_liklihood = 0
        likelihood = 0
        hist_likelihood = []
        i = 0
        while likelihood - last_liklihood > 2 or likelihood == 0 or i == 1:
            last_liklihood = likelihood
            self.logger.info(f"start iteration number {str(i+1)}")
            # E
            z = self.calculate_z(alpha, p)
            w = self.calculate_w(z)

            # M
            alpha = self.calculate_alpha(w)
            p = self.calculate_p(self.num_topics, w)
            likelihood = self.get_likelihood(z)
            self.logger.info(f"Current likelihood {likelihood}")
            cluster_res = w.argmax(axis=0)

            perplexity = self.get_perplexity(likelihood)
            accuracy = self.get_accuracy(cluster_res)

            hist_likelihood.append({'likelihood': likelihood, 'accuracy': accuracy, "perplexity": perplexity})
            self.logger.info(f'the log likelihood diff is {likelihood - last_liklihood} (should be positive)')
            self.logger.info(f"Cur accuracy is: {accuracy}")
            i += 1

        self.plot_confusion(cluster_res)
        likelihood = pd.DataFrame(hist_likelihood)
        likelihood['likelihood'].plot()
        plt.savefig(f"likelihood_{self.LAMBDA}_{now}.jpg")
        plt.show()
        plt.close()
        likelihood['accuracy'].plot()
        plt.savefig(f"loss_{self.LAMBDA}_{now}.jpg")
        plt.show()
        plt.close()
        likelihood['perplexity'].plot()
        plt.savefig(f"perplexity_{self.LAMBDA}_{now}.jpg")
        plt.show()

    def get_accuracy(self, cluster_res):
        results = defaultdict(list)
        correct = 0
        for doc, res in zip(self.docs, cluster_res):
            results[res].append(doc)
        c2topic = {}
        for cluster_i, cluster_docs in results.items():
            cluster_topics = []
            for doc in cluster_docs:
                cluster_topics.extend(doc.topics)
            counter = Counter(cluster_topics)
            cluster_topic = counter.most_common(1)[0][0]
            c2topic[cluster_i] = cluster_topic
            for doc in cluster_docs:
                if cluster_topic in doc.topics:
                    correct +=1
        self.logger.info(f"the topics cluster are: \n{c2topic}")
        return correct / len(self.docs)

    def get_likelihood(self, z):
        sum_l = 0
        for t in range(len(self.docs)):

            m_t = z[:, t].max()
            sum_ln = 0
            for i in range(self.num_topics):
                if z[i][t] - m_t >= - self.K:
                    sum_ln += e**(z[i][t] - m_t)

            sum_l += m_t + np.log(sum_ln)

        return sum_l

    def get_perplexity(self, likelihood):
        return round(e**(-likelihood/len(self.words_stat)), 2)

    def plot_confusion(self, cluster_res):
        results = defaultdict(list)
        for doc, res in zip(self.docs, cluster_res):
            results[res].append(doc)

        sorted_res = sorted(results.values(), key=lambda x: len(x), reverse=True)

        confusion_matrix = []
        for i, cluster_docs in enumerate(sorted_res):
            cluster_topics = {topic: 0 for topic in self.topic2i}
            for doc in cluster_docs:
                for topic in doc.topics:
                    cluster_topics[topic] += 1
            df = pd.DataFrame.from_dict(cluster_topics, orient='index')
            max_topic = max(cluster_topics.items(), key=operator.itemgetter(1))[0]
            df.plot(kind='bar', title=f"cluster {i} - {max_topic}")
            plt.show()
            plt.savefig(f'cluster_{i}.jpg')
            plt.close()
            cluster_topics['cluster_size'] = len(cluster_docs)
            confusion_matrix.append(cluster_topics)

        print(pd.DataFrame(confusion_matrix))


def main():
    data_path = sys.argv[1]
    topic_path = sys.argv[2]
    algo = Em(data_path, topic_path)
    algo.run_em()


if __name__ == '__main__':
    main()
