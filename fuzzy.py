from bs4 import BeautifulSoup
import glob
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import sys
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
    for vec in bow:
        vec_1d = vec.toarray().reshape(-1)
        dist_from_0.append(cos_sim(vec_1d, vec_0))
