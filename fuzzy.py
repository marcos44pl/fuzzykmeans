from bs4 import BeautifulSoup
import glob
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import sys
from FuzzyKMeans import FuzzyKMeans
import matplotlib.pyplot as plt
import collections

files_pattern = "./RM/*.html"
METRIC_METHOD = "euc"
K = 3
M = 1.3


def create_bag_of_words():
    vectorizer = CountVectorizer()
    texts = []
    docs_dic = {}
    i = 0
    for file_path in glob.glob(files_pattern):
        with open(file_path, mode="r", encoding="UTF-8") as f:
            try:

                html_text = f.read()
                soup = BeautifulSoup(html_text, "html5lib")
                all_p = [tag.text for tag in soup.findAll('p')]
                text = ' '.join(all_p)
                texts.append(text)
                docs_dic[i] = file_path
                i += 1
            except UnicodeDecodeError:
                pass

    return vectorizer.fit_transform(texts), docs_dic


def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def euclidean_sim(a, b):
    r = np.subtract(a, b)
    rp = np.power(r, 2)
    sum_rp = np.sum(rp)
    return np.sqrt(sum_rp)


sim_func = \
    {
        "cos": cos_sim,
        "euc": euclidean_sim
    }


def sim(a, b, metric):
    return sim_func[metric](a, b)


def norm(a):
    sum_d = np.sum(a)
    return np.divide(a, sum_d)


if __name__ == '__main__':
    bow, docs_dic = create_bag_of_words()
    vec_0 = np.array([sys.float_info.epsilon for i in range(bow.shape[1])])
    dist_from_0 = []
    fuzzy = FuzzyKMeans(K, M)
    X = bow.toarray()
    fuzzy.fit(X)
    dbg = 0
    for vec in bow:
        vec_1d = vec.toarray().reshape(-1)
        dist_from_0.append(sim(vec_1d, vec_0, METRIC_METHOD))
    centers = []
    for c in fuzzy.cluster_centers_:
        centers.append(sim(c, vec_0, METRIC_METHOD))

    # SHOW FUZZY K-MEANDs GROUPS AS PLOT
    plot_colors = 'rygcmkw'
    plt.plot(dist_from_0, [0 for i in range(len(dist_from_0))], 'b.')
    for ik in range(K):
        od = collections.OrderedDict(sorted(zip(dist_from_0, fuzzy.fuzzy_labels_[:, ik])))
        plt.plot(od.keys(), od.values(), plot_colors[ik] + '.')
    plt.plot(centers, [0.1 * i for i in range(len(centers))], 'k+')
    plt.show()

    # SORTING DOCS AND ITS GROUPS
    docs_list = []
    for i in range(len(dist_from_0)):
        docs_list.append((dist_from_0[i], fuzzy.fuzzy_labels_[i, :], docs_dic[i]))
    sorted_docs = sorted(docs_list, key=lambda x: x[0])

    for d in sorted_docs: print(d)
