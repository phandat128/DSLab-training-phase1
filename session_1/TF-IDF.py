import os
import re
from collections import defaultdict

import numpy as np


def gather_20newsgroup_data():
    _path = "../datasets/20news-bydate/"
    dirs = [_path + _dir_name + "/"
            for _dir_name in os.listdir(_path)
            if not os.path.isfile(_path + _dir_name)]
    _test_dir, _train_dir = dirs[0], dirs[1]
    _newsgroup_list = [_newsgroup for _newsgroup in os.listdir(_train_dir)]
    _newsgroup_list.sort()
    with open(_path + "stop-words.txt") as f:
        _stop_words = f.read().split(",")
        print(_stop_words)
    from nltk.stem.porter import PorterStemmer
    _stemmer = PorterStemmer()

    def collect_data_from(_parent_dir, _newsgroup_list):
        print(_parent_dir)
        print(_newsgroup_list)
        _data = []
        for _group_id, _newsgroup in enumerate(_newsgroup_list):  # for each newsgroup
            _dir_path = _parent_dir + _newsgroup + "/"
            # list of (filename, filepath) in current newsgroup
            _files = [(_filename, _dir_path + _filename)
                      for _filename in os.listdir(_dir_path)
                      if os.path.isfile(_dir_path + _filename)]
            _files.sort()
            for _filename, _filepath in _files:  # for each file
                with open(_filepath) as _f:
                    _text = _f.read().lower()
                    # remove stop words, return array of stemmed remaining words
                    _remain_words = [_stemmer.stem(_word)
                                     for _word in re.split('\W+', _text)
                                     if _word not in _stop_words]
                    # combine remaining words
                    _content = ' '.join(_remain_words)
                    assert len(_content.splitlines()) == 1  # only one line
                    _data.append(str(_group_id) + '<fff>' + _filename + '<fff>' + _content)
        return _data

    _train_data = collect_data_from(_train_dir, _newsgroup_list)
    _test_data = collect_data_from(_test_dir, _newsgroup_list)

    _full_data = _train_data + _test_data
    with open(_path + "20news-train-processed.txt", "w") as f:
        f.write('\n'.join(_train_data))
    with open(_path + "20news-test-processed.txt", "w") as f:
        f.write('\n'.join(_test_data))
    with open(_path + "20news-full -processed.txt", "w") as f:
        f.write('\n'.join(_full_data))


def generate_vocabulary(_data_processed_path):
    def compute_idf(_document_freq, _corpus_size):
        # idf = log(corpus_size/1+document_frequency)
        return np.log10(_corpus_size / (1 + _document_freq))

    with open(_data_processed_path) as f:
        _lines = f.read().splitlines()
    _doc_count = defaultdict(int)
    _corpus_size = len(_lines)

    for _line in _lines:  # one line is data of one file
        _features = _line.split('<fff>')
        _text = _features[-1]
        _words = list(set(_text.split()))
        for _word in _words:
            _doc_count[_word] += 1  # count documents containing _word
    _words_idfs = [(_word, compute_idf(_doc_count[_word], _corpus_size))
                   for _word in _doc_count.keys()
                   if not _word.isdigit()]
    _words_idfs.sort(key=lambda _word_idf: -_word_idf[-1])
    print("Vocabulary size: {}".format(len(_words_idfs)))
    with open("../datasets/20news-bydate/word_idfs.txt", "w") as f:
        f.write('\n'.join([_word + "<fff>" + str(_idf) for _word, _idf in _words_idfs]))


def get_tf_idf(_data_path):
    # get precomputed idf values
    with open("../datasets/20news-bydate/word_idfs.txt") as f:
        # list of (word, idf)
        _word_idfs = [(_line.split("<fff>")[0], float(_line.split("<fff>")[1]))
                      for _line in f.read().splitlines()]
        _word_IDs = dict([(_word, _index)
                          for _index, (_word, _idf) in enumerate(_word_idfs)])
        _idfs = dict(_word_idfs)
    with open(_data_path) as f:
        _documents = [(int(_line.split("<fff>")[0]),
                       int(_line.split("<fff>")[1]),
                       _line.split("<fff>")[2])
                      for _line in f.read().splitlines()]
        _data_tf_idf = []
        for _document in _documents:  # for each document
            _group_id, _doc_id, _text = _document
            _words_in_doc = [_word for _word in _text.split() if _word in _idfs]
            _word_set = list(set(_words_in_doc))
            _max_tf = max([_words_in_doc.count(_word) for _word in _word_set])

            _words_tf_idf = []
            _sum_squares = 0.0
            for _word in _word_set:
                _tf = _words_in_doc.count(_word)
                _tf_idf = _tf * 1.0 / _max_tf * _idfs[_word]
                _words_tf_idf.append((_word_IDs[_word], _tf_idf))
                _sum_squares += _tf_idf ** 2

            _words_tf_idf_normalized = [str(_index) + ":" + str(_tf_idf / np.sqrt(_sum_squares))
                                        for _index, _tf_idf in _words_tf_idf]
            _sparse_rep = ' '.join(_words_tf_idf_normalized)
            _data_tf_idf.append((_group_id, _doc_id, _sparse_rep))
        with open("../datasets/20news-bydate/tf-idf.txt", "w") as _f:
            _f.write("\n".join([str(_group_id) + "<fff>" + str(_doc_id) + "<fff>" + _sparse_rep
                                for _group_id, _doc_id, _sparse_rep in _data_tf_idf]))


if __name__ == '__main__':
    gather_20newsgroup_data()
    data_train_path = "../datasets/20news-bydate/20news-train-processed.txt"
    generate_vocabulary(data_train_path)
    get_tf_idf(data_train_path)
