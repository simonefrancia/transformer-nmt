# -*- coding: utf-8 -*-
#/usr/bin/python2

'''
July 2018 by Simone Francia.
simone.francia@musixmatch.com

'''

from __future__ import print_function
import tensorflow as tf
import numpy as np
import codecs
import os
from collections import Counter
import regex

FLAGS = tf.flags.FLAGS


def make_vocab(fpath, fname):
    '''Constructs vocabulary.

    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.

    Writes vocabulary line by line to `preprocessed/fname`
    '''

    text = codecs.open(fpath, 'r', 'utf-8').read()
    
    text = u''.join((text)).encode('utf-8').strip()

    text = regex.sub("[^\s\p{Latin}']", "", text)

    words = text.split()
    word2cnt = Counter(words)
    print(len(word2cnt))

    if not os.path.exists('preprocessed'): os.mkdir('preprocessed')
    with codecs.open('preprocessed/{}'.format(fname), 'w', 'utf-8') as fout:
        fout.write(
            "{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))


def load_input_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/input.vocab', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=FLAGS.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_output_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/output.vocab', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=FLAGS.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(source_sents, target_sents): 
    input2idx, idx2input = load_input_vocab()
    output2idx, idx2output = load_output_vocab()
    
    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [input2idx.get(word, 1) for word in (source_sent + u" </S>").split()] # 1: OOV, </S>: End of Text
        y = [output2idx.get(word, 1) for word in (target_sent + u" </S>").split()]
        if max(len(x), len(y)) <=FLAGS.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)
    
    # Pad      
    X = np.zeros([len(x_list), FLAGS.maxlen], np.int32)
    Y = np.zeros([len(y_list), FLAGS.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, FLAGS.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, FLAGS.maxlen-len(y)], 'constant', constant_values=(0, 0))
    
    return X, Y, Sources, Targets

def load_train_data():
    input_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(FLAGS.source_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    output_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(FLAGS.target_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]

    
    X, Y, Sources, Targets = create_data(input_sents, output_sents)
    return X, Y
    
def load_test_data():
    def _refine(line):
        line = regex.sub("<[^>]+>", "", line)
        line = regex.sub("[^\s\p{Latin}']", "", line) 
        return line.strip()
    
    input_sents = [_refine(line) for line in codecs.open(FLAGS.source_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]
    output_sents = [_refine(line) for line in codecs.open(FLAGS.target_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]
        
    X, Y, Sources, Targets = create_data(input_sents, output_sents)
    return X, Sources, Targets # (1064, 150)

def get_batch_data():
    # Load data
    X, Y = load_train_data()
    
    # calc total batch count
    num_batch = len(X) // FLAGS.batch_size

    
    """# Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32,name="input_x")
    Y = tf.convert_to_tensor(Y, tf.int32,name="input_y")
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])
            
    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=FLAGS.batch_size,
                                capacity=FLAGS.batch_size*64,
                                min_after_dequeue=FLAGS.batch_size*32,
                                allow_smaller_final_batch=False)
    
    return x, y, num_batch # (N, T), (N, T), ()"""
    return X,Y,num_batch

    # ! /usr/bin/env python



def is_number(s):
    """
    Checks wether the provided string is a number. Accepted: 1 | 1.0 | 1e-3 | 1,0
    """
    s = s.replace(',', '.')
    try:
        float(s)
        return True
    except ValueError:
        return False

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"@[A-Za-z0-9]+", " ", string)  # For Twitter use: remove hashtags
    return string.strip()

def combine_data_files(data_files):
    lines = []
    for d in data_files:
        with open(d) as fin: lines.extend(fin.readlines())

    return lines

def load_cleaned_text(filepath):
    with open(filepath, "r") as f:
        list_ = list(map(lambda x: clean_str(x), f.readlines()))

    return list_

def load_data_and_labels(data_files, output_dir=None):

    # Load data from files
    if len(data_files) > 1:
        examples = combine_data_files(data_files=data_files)
    elif len(data_files) == 1:
        examples = list(open(data_files[0], "r").readlines())
    else:
        examples = []

    # Save label of every example
    y_text = []
    x_text = []
    for line in examples:
        line = line.strip()
        split = line.split('\t', 1)
        y_text.append(split[0])
        x_text.append(split[1])

    assert len(x_text) == len(y_text)

    x_text = [clean_str(sent) for sent in x_text]

    return x_text, y_text

def load_sequence_data_and_labels(data_files, output_dir=None):

    # Load data from files
    if len(data_files) > 1:
        examples = combine_data_files(data_files=data_files)
    elif len(data_files) == 1:
        examples = list(open(data_files[0], "r").readlines())
    else:
        examples = []

    x_text = []
    y_text = []
    sentence = ''
    tags = []
    for line in examples:
        line = line.strip()
        if (len(line) == 0 or line.startswith("-DOCSTART-")):
            sentence = sentence.strip()
            if sentence:
                x_text.append(sentence)
                y_text.append(tags)
                sentence = ''
                tags = []
        else:
            splitted_line = line.split(' ')
            token = splitted_line[0]
            tag = splitted_line[-1]
            sentence += token + ' '
            tags += [tag]

    # not required as the input sequence should be already tokenized
    # x_text = [clean_str(sent) for sent in x_text]

    return x_text, y_text

def batch_iter(data_x, data_y, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    assert len(data_x) == len(data_y)

    len_data = len(data_y)

    num_batches_per_epoch = int(len_data / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(len_data))
            shuffled_data_x = data_x[shuffle_indices]
            shuffled_data_y = data_y[shuffle_indices]
        else:
            shuffled_data_x = data_x
            shuffled_data_y = data_y

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len_data)

            yield list(zip(shuffled_data_x[start_index:end_index], shuffled_data_y[start_index:end_index]))

def batch_iter_seq2seq(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]

