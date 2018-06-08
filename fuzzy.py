from bs4 import BeautifulSoup
import glob
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import sys
from FuzzyKMeans import FuzzyKMeans
import matplotlib.pyplot as plt

files_pattern = "./RM/*.html"
punctuation_pattern = ' |\.$|\. |, |\/|\(|\)|\'|\"|\!|\?|\+|\n|-'


def create_bag_of_words():
    vectorizer = CountVectorizer()
    texts = []
    for file_path in glob.glob(files_pattern):
        with open(file_path, mode="r", encoding="UTF-8") as f:
            html_text = f.read()
            soup = BeautifulSoup(html_text, "html5lib")
            all_p = [tag.text for tag in soup.findAll('p')]
            text = ' '.join(all_p)
            texts.append(text)  # [w for w in re.split(punctuation_pattern, text) if w]

    return vectorizer.fit_transform(texts)


def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


if __name__ == '__main__':
    bow = create_bag_of_words()
    vec_0 = np.array([sys.float_info.epsilon for i in range(bow.shape[1])])
    dist_from_0 = []
    k = 3
    fuzzy = FuzzyKMeans(k)
    X = bow.toarray()
    fuzzy.fit(X)
    dbg = 0
    for vec in bow:
        vec_1d = vec.toarray().reshape(-1)
        dist_from_0.append(cos_sim(vec_1d, vec_0))

    centers = []
    for c in fuzzy.cluster_centers_:
        centers.append(cos_sim(c, vec_0))

    plt.plot(dist_from_0, [0 for i in range(len(dist_from_0))], 'b.')
    plt.plot(dist_from_0, fuzzy.fuzzy_labels_[:, 0], 'r.')
    plt.plot(dist_from_0, fuzzy.fuzzy_labels_[:, 1], 'y.')
    plt.plot(dist_from_0, fuzzy.fuzzy_labels_[:, 2], 'g.')
    plt.plot(centers, [0 for i in range(len(centers))], 'k+')

    plt.show()
