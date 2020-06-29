"""
    Brief description
    
    Author: Edilson A. Correa Junior <edilsonacjr@gmail.com>
"""

import glob
import nltk
import numpy as np

from pathlib import Path

from scipy import misc
from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_openml
from sklearn.feature_extraction.text import TfidfVectorizer


def get_mnist(folder='data/'):

    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    data = fetch_mldata('MNIST original')

    x = data.data
    y = data.target

    np.save(path / 'mnist_data.npy', x)
    np.save(path / 'mnist_target.npy', y)
    return x, y


def get_olivetti(folder='data/'):

    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)

    data = fetch_olivetti_faces()

    x = data.images.reshape((len(data.images), -1))
    y = data.target

    np.save(path / 'olivetti_data.npy', x)
    np.save(path / 'olivetti_target.npy', y)
    return x, y


def get_newsgroups(folder='data/'):

    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)

    tokenize = nltk.tokenize.RegexpTokenizer('(?u)\\b\\w\\w+\\b')
    stopwords = nltk.corpus.stopwords.words('english')

    data = fetch_20newsgroups(subset='all')
    y = data.target
    x = data.data

    x_transformed = TfidfVectorizer(tokenizer=tokenize.tokenize, stop_words=stopwords, min_df=5).fit_transform(x)
    #x_transformed = x_transformed.toarray()

    np.save(path / 'newsgroups_data.npy', x_transformed)
    np.save(path / 'newsgroups_target.npy', y)
    return x, y


def get_coil(origin_path='data/coil-20-proc/*.png', folder='data/'):

    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)

    original_pic = []
    y = []
    for image_path in glob.glob(origin_path):
        image = misc.imread(image_path)
        image = misc.imresize(image, (32, 32))
        original_pic.append(image)
        y.append(image_path.split('/')[-1].split('__')[0])

    im = np.asarray(original_pic)
    x = im.reshape((len(im), -1))
    y = np.array(y)

    np.save(path / 'coil_data.npy', x)
    np.save(path / 'coil_target.npy', y)
    return x, y


def get_fashion(folder='data/'):

    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    data = fetch_openml('Fashion-MNIST')

    x = data.data
    y = data.target

    np.save(path / 'fashion_data.npy', x)
    np.save(path / 'fashion_target.npy', y)
    return x, y


def main():
    get_mnist()
    get_olivetti()
    get_newsgroups()
    get_coil()


if __name__ == '__main__':
    main()
