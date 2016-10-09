# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for downloading data from WMT, tokenizing, vocabularies."""

import gzip
import os
import re
import tarfile
from tensorflow.python.platform import gfile
from urllib.request import urlretrieve

__author__ = 'fuhuamosi'

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br'\d')

# URLs for WMT data.
_WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/training-giga-fren.tar"
_WMT_ENFR_DEV_URL = "http://www.statmt.org/wmt15/dev-v2.tgz"


def maybe_download(directory, filename, url):
    if not os.path.exists(directory):
        print("Creating directory %s" % directory)
        os.mkdir(directory)
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        print("Downloading %s to %s" % (url, filepath))
        filepath, _ = urlretrieve(url, filepath)
        statinfo = os.stat(filepath)
        print("Successfully downloaded", filename, statinfo.st_size, "bytes")
    return filepath


def gunzip_file(gz_path, new_path):
    with gzip.open(gz_path, 'rb') as gz_file:
        with open(new_path, 'wb') as new_file:
            for line in gz_file:
                new_file.write(line)


def get_wmt_enfr_train_set(directory):
    train_path = os.path.join(directory, 'giga-fren.release2')
    if not (os.path.exists(train_path + '.fr')
            and os.path.exists(train_path + '.en')):
        corpus_file = maybe_download(directory, 'training-giga-fren.tar',
                                     _WMT_ENFR_TRAIN_URL)
        print('Extracting tar file {}'.format(corpus_file))
        with tarfile.open(corpus_file, 'r') as corpus_tar:
            corpus_tar.extractall(directory)
        gunzip_file(train_path + '.fr.gz', train_path + '.fr')
        gunzip_file(train_path + '.en.gz', train_path + '.en')
    return train_path


def get_wmt_enfr_dev_set(directory):
    devname = 'newstest2013'
    dev_path = os.path.join(directory, devname)
    if not (os.path.exists(dev_path + '.fr')
            and os.path.exists(dev_path + '.en')):
        dev_file = maybe_download(directory, 'dev-v2.tgz',
                                  _WMT_ENFR_DEV_URL)
        print('Extracting tgz file {}'.format(dev_file))
        with tarfile.open(dev_file, 'r') as dev_tar:
            fr_dev_file = dev_tar.getmember('dev/' + devname + '.fr')
            en_dev_file = dev_tar.getmember('dev/' + devname + '.en')
            fr_dev_file.name = devname + '.fr'
            en_dev_file.name = devname + '.en'
            dev_tar.extract(fr_dev_file, directory)
            dev_tar.extract(en_dev_file, directory)
    return dev_path


def basic_tokenizer(sentence):
    words = []
    for s in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, s))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    """Create vocabulary file (if it does not exist yet) from data file.
    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.
    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path):
        print('Creating vocabulary {} from '
              'data {}'.format(vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode='rb') as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print('  processing line {}'.format(counter))
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = re.sub(_DIGIT_RE, b'0', w) if normalize_digits else w
                    vocab.setdefault(word, 0)
                    vocab[word] += 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, 'wb') as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b'\n')


def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        vocab_list = []
        with gfile.GFile(vocabulary_path, mode='rb') as f:
            vocab_list.extend(f.readlines())
        rev_vocab = [line.strip() for line in vocab_list]
        vocab = dict([(y, x) for (x, y) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError('Vocabulary file {} not found'.format(vocabulary_path))


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None,
                          normalize_digits=True):
    words = tokenizer(sentence) if tokenizer else basic_tokenizer(sentence)
    if normalize_digits:
        return [vocabulary.get(re.sub(_DIGIT_RE, b'0', w), UNK_ID) for w in words]
    return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
    if not gfile.Exists(target_path):
        print('Tokenizing data in {}'.format(data_path))
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, 'rb') as data_file:
            with gfile.GFile(target_path, 'w') as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                                      normalize_digits)
                    token_ids = [str(token_id) for token_id in token_ids]
                    tokens_file.write(' '.join(token_ids) + '\n')


def prepare_wmt_data(data_dir, en_vocabulary_size, fr_vocabulary_size, tokenizer=None):
    # Get wmt data to the specified directory.
    train_path = get_wmt_enfr_train_set(data_dir)
    dev_path = get_wmt_enfr_dev_set(data_dir)

    # Create vocabularies of the appropriate sizes.
    fr_vocab_path = os.path.join(data_dir, "vocab%d.fr" % fr_vocabulary_size)
    en_vocab_path = os.path.join(data_dir, "vocab%d.en" % en_vocabulary_size)
    create_vocabulary(fr_vocab_path, train_path + ".fr", fr_vocabulary_size, tokenizer)
    create_vocabulary(en_vocab_path, train_path + ".en", en_vocabulary_size, tokenizer)

    # Create token ids for the training data.
    fr_train_ids_path = train_path + (".ids%d.fr" % fr_vocabulary_size)
    en_train_ids_path = train_path + (".ids%d.en" % en_vocabulary_size)
    data_to_token_ids(train_path + ".fr", fr_train_ids_path, fr_vocab_path, tokenizer)
    data_to_token_ids(train_path + ".en", en_train_ids_path, en_vocab_path, tokenizer)

    # Create token ids for the development data.
    fr_dev_ids_path = dev_path + (".ids%d.fr" % fr_vocabulary_size)
    en_dev_ids_path = dev_path + (".ids%d.en" % en_vocabulary_size)
    data_to_token_ids(dev_path + ".fr", fr_dev_ids_path, fr_vocab_path, tokenizer)
    data_to_token_ids(dev_path + ".en", en_dev_ids_path, en_vocab_path, tokenizer)

    return (en_train_ids_path, fr_train_ids_path,
            en_dev_ids_path, fr_dev_ids_path,
            en_vocab_path, fr_vocab_path)
