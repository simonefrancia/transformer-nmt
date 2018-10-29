# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Sept 2018 by Simone Francia.
simone.francia@musixmatch.com

'''
from __future__ import print_function
import tensorflow as tf

from data_load import get_batch_data, load_input_vocab, load_output_vocab , make_vocab, batch_iter_seq2seq
from multihead_attention import *
import os, codecs
from tqdm import tqdm
import shutil
import numpy as np
import regex


#_small
tf.flags.DEFINE_string("source_train", "corpora/train.tags.de-en.de", "Path to a file containing X training sentence")
tf.flags.DEFINE_string("target_train", "corpora/train.tags.de-en.en", "Path to a file containing Y training sentence")
tf.flags.DEFINE_string("source_test", "corpora/IWSLT16.TED.tst2011.de-en.en.xml", "Path to a file containing X testing sentence")
tf.flags.DEFINE_string("target_test", "corpora/IWSLT16.TED.tst2011.de-en.en.xml", "Path to a file containing Y testing sentence")

tf.flags.DEFINE_integer("batch_size", 32, "Batch Size")
tf.flags.DEFINE_float("lr", 0.0001, "Learning rate for backpropagation")
tf.flags.DEFINE_string("logdir", "logdir", "Where to save the trained model, checkpoints and stats")
tf.flags.DEFINE_integer("maxlen", 20, "Max length of output summarizations")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs")
tf.flags.DEFINE_integer("num_blocks", 6, "number of encoder/decoder blocks")
tf.flags.DEFINE_integer("hidden_units", 512, "")
tf.flags.DEFINE_integer("num_heads", 8, "")
tf.flags.DEFINE_integer("min_cnt", 20, "")
tf.flags.DEFINE_float("dropout_keep_prob", 0.1, "")
tf.flags.DEFINE_boolean("sinusoid",False,"If True, use sinusoid. If false, positional embedding.")

FLAGS = tf.flags.FLAGS


class Graph():
    def __init__(self, is_training=True):

            self.input_x = tf.placeholder(tf.int32, shape=(None, FLAGS.maxlen), name="input_x")
            self.y = tf.placeholder(tf.int32, shape=(None, FLAGS.maxlen), name="y")

            # define decoder inputs
            self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1])*2, self.y[:, :-1]), -1) # 2:<S>

        # Load vocabulary
            input2idx, idx2input = load_input_vocab()
            output2idx, idx2output = load_output_vocab()
            
            # Encoder
            with tf.variable_scope("encoder"):

                ## Embedding

                with tf.variable_scope("encoder_embedding", reuse=None):
                    lookup_table = tf.get_variable('lookup_table',
                                                   dtype=tf.float32,
                                                   shape=[len(input2idx), FLAGS.hidden_units],
                                                   initializer=tf.contrib.layers.xavier_initializer())

                    lookup_table = tf.concat((tf.zeros(shape=[1, FLAGS.hidden_units]), lookup_table[1:, :]), 0)

                    outputs = tf.nn.embedding_lookup(lookup_table, self.input_x)

                    self.enc = outputs * (FLAGS.hidden_units ** 0.5)


                ## Positional Encoding
                if FLAGS.sinusoid:

                    N, T = self.input_x.get_shape().as_list()
                    with tf.variable_scope("encoder_pos_enc", reuse=None):
                        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

                        # First part of the PE function: sin and cos argument
                        position_enc = np.array([
                            [pos / np.power(10000, 2. * i / FLAGS.hidden_units) for i in range(FLAGS.hidden_units)]
                            for pos in range(T)])

                        # Second part, apply the cosine to even columns and sin to odds.
                        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
                        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

                        # Convert to a tensor
                        lookup_table = tf.convert_to_tensor(position_enc)

                        self.enc += tf.nn.embedding_lookup(lookup_table, position_ind)

                else:

                    with tf.variable_scope("encoder_pos_enc", reuse=None):
                        inputs = tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_x)[1]), 0), [tf.shape(self.input_x)[0], 1])
                        lookup_table = tf.get_variable('lookup_table',
                                                       dtype=tf.float32,
                                                       shape=[FLAGS.maxlen, FLAGS.hidden_units],
                                                       initializer=tf.contrib.layers.xavier_initializer())

                        self.enc += tf.nn.embedding_lookup(lookup_table, inputs)

                with tf.variable_scope("dropout_keep_prob", reuse=None):
                    ## Dropout
                    self.enc = tf.layers.dropout(self.enc, rate=FLAGS.dropout_keep_prob, training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(FLAGS.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc = multihead_attention(queries=self.enc, 
                                                        keys=self.enc, 
                                                        num_units=FLAGS.hidden_units,
                                                        num_heads=FLAGS.num_heads,
                                                        dropout_rate=FLAGS.dropout_keep_prob,
                                                        is_training=is_training,
                                                        causality=False)
                        
                        ### Feed Forward

                        with tf.variable_scope("feedforward", reuse=None):
                            num_units = [4 * FLAGS.hidden_units, FLAGS.hidden_units]
                            # Inner layer
                            params = {"inputs": self.enc, "filters": num_units[0], "kernel_size": 1,
                                      "activation": tf.nn.relu, "use_bias": True}
                            outputs = tf.layers.conv1d(**params)

                            # Readout layer
                            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                                      "activation": None, "use_bias": True}
                            outputs = tf.layers.conv1d(**params)

                            # Residual connection
                            outputs += self.enc

                            epsilon = 1e-8
                            with tf.variable_scope("ln", reuse=None):
                                inputs_shape = outputs.get_shape()
                                params_shape = inputs_shape[-1:]

                                mean, variance = tf.nn.moments(outputs, [-1], keep_dims=True)
                                beta = tf.Variable(tf.zeros(params_shape))
                                gamma = tf.Variable(tf.ones(params_shape))
                                normalized = (outputs - mean) / ((variance + epsilon) ** (.5))
                                self.enc = gamma * normalized + beta


            # Decoder
            with tf.variable_scope("decoder"):
                ## Embedding

                with tf.variable_scope("decoder_embedding", reuse=None):
                    lookup_table = tf.get_variable('lookup_table',
                                                   dtype=tf.float32,
                                                   shape=[len(output2idx), FLAGS.hidden_units],
                                                   initializer=tf.contrib.layers.xavier_initializer())
                    lookup_table = tf.concat((tf.zeros(shape=[1, FLAGS.hidden_units]),
                                                  lookup_table[1:, :]), 0)
                    outputs = tf.nn.embedding_lookup(lookup_table, self.decoder_inputs)

                    self.dec = outputs * (FLAGS.hidden_units ** 0.5)

                ## Positional Encoding
                if FLAGS.sinusoid:

                    N, T = self.decoder_inputs.get_shape().as_list()
                    with tf.variable_scope("decoder_pos_enc", reuse=None):
                        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

                        # First part of the PE function: sin and cos argument
                        position_enc = np.array([
                            [pos / np.power(10000, 2. * i / FLAGS.hidden_units) for i in range(FLAGS.hidden_units)]
                            for pos in range(T)])

                        # Second part, apply the cosine to even columns and sin to odds.
                        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
                        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

                        # Convert to a tensor
                        lookup_table = tf.convert_to_tensor(position_enc)

                        self.dec += tf.nn.embedding_lookup(lookup_table, position_ind)


                else:

                    with tf.variable_scope("decoder_pos_enc", reuse=None):
                        inputs = tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1])
                        lookup_table = tf.get_variable('lookup_table',
                                                       dtype=tf.float32,
                                                       shape=[FLAGS.maxlen, FLAGS.hidden_units],
                                                       initializer=tf.contrib.layers.xavier_initializer())

                        self.dec += tf.nn.embedding_lookup(lookup_table, inputs)

                ## Dropout
                self.dec = tf.layers.dropout(self.dec, rate=FLAGS.dropout_keep_prob, training=tf.convert_to_tensor(is_training))
                
                ## Blocks
                for i in range(FLAGS.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ## Multihead Attention ( self-attention)
                        self.dec = multihead_attention(queries=self.dec, 
                                                        keys=self.dec, 
                                                        num_units=FLAGS.hidden_units,
                                                        num_heads=FLAGS.num_heads,
                                                        dropout_rate=FLAGS.dropout_keep_prob,
                                                        is_training=is_training,
                                                        causality=True, 
                                                        scope="self_attention")
                        
                        ## Multihead Attention ( vanilla attention)
                        self.dec = multihead_attention(queries=self.dec, 
                                                        keys=self.enc, 
                                                        num_units=FLAGS.hidden_units,
                                                        num_heads=FLAGS.num_heads,
                                                        dropout_rate=FLAGS.dropout_keep_prob,
                                                        is_training=is_training, 
                                                        causality=False,
                                                        scope="vanilla_attention")
                        
                        ## Feed Forward

                        with tf.variable_scope("feedforward", reuse=None):
                            # Inner layer

                            num_units = [4 * FLAGS.hidden_units, FLAGS.hidden_units]
                            params = {"inputs": self.dec, "filters": num_units[0], "kernel_size": 1,
                                      "activation": tf.nn.relu, "use_bias": True}
                            outputs = tf.layers.conv1d(**params)

                            # Readout layer
                            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                                      "activation": None, "use_bias": True}

                            outputs = tf.layers.conv1d(**params)

                            # Residual connection
                            outputs += self.dec

                            # Normalize
                            with tf.variable_scope("ln", reuse=None):
                                epsilon = 1e-8
                                inputs_shape = outputs.get_shape()
                                params_shape = inputs_shape[-1:]

                                mean, variance = tf.nn.moments(outputs, [-1], keep_dims=True)
                                beta = tf.Variable(tf.zeros(params_shape))
                                gamma = tf.Variable(tf.ones(params_shape))
                                normalized = (outputs - mean) / ((variance + epsilon) ** (.5))
                                self.dec = gamma * normalized + beta



            with tf.variable_scope("output"):
                # Final linear projection
                self.logits = tf.layers.dense(self.dec, len(output2idx), name="logits")
                self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1),name="predictions")
                self.istarget = tf.to_float(tf.not_equal(self.y, 0))
                self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))
                tf.summary.scalar('acc', self.acc)

            with tf.name_scope("loss"):
                if is_training:

                    # Loss
                    epsilon=0.1
                    inputs = tf.one_hot(self.y, depth=len(output2idx))
                    K = inputs.get_shape().as_list()[-1]  # number of channels
                    self.y_smoothed = ((1 - epsilon) * inputs) + (epsilon / K)

                    self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_smoothed)
                    self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))

                    # Training Scheme
                    self.global_step = tf.Variable(0, name='global_step', trainable=False)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                    self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

                    # Summary
                    tf.summary.scalar('mean_loss', self.mean_loss)
                    self.merged = tf.summary.merge_all()

if __name__ == '__main__':

    if os.path.exists("preprocessed/"):
        shutil.rmtree("preprocessed/")
        print("preprocessed deleted")

    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)
        print("logdir created")

    print("Create preprocessed data.....")
    make_vocab(FLAGS.source_train, "input.vocab")
    make_vocab(FLAGS.target_train, "output.vocab")
    print("....Done\n")


    # Load vocabulary    
    input2idx, idx2input = load_input_vocab()
    output2idx, idx2output = load_output_vocab()
    
    # Construct graph
    g = Graph("train")
    print("Graph loaded\n")

    print("Loading batch data.....")
    x, y, _ = get_batch_data()
    print(len(x))
    print(len(y))
    print("........Done")

    x = np.array(x)
    y = np.array(y)


    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        batches = batch_iter_seq2seq(x, y, FLAGS.batch_size, FLAGS.num_epochs)
        print("num batches")
        num_batch = (len(x) - 1) // FLAGS.batch_size + 1
        print(num_batch)

        for epoch in range(0, FLAGS.num_epochs):

            print("Epoch "+ str(epoch+1) + "")
            pbar = tqdm(total=num_batch, ncols=100)
            for step,(batch_x, batch_y) in enumerate(batches):

                _,  st, loss = sess.run([g.train_op, g.global_step, g.loss], feed_dict={g.input_x : batch_x , g.y : batch_y})
                print(st,np.mean(loss))

                pbar.update(1)
                if step == num_batch:
                    break


            saver.save(sess, FLAGS.logdir + '/model_epoch_%02d' % (epoch))


        """builder = tf.saved_model.builder.SavedModelBuilder(FLAGS.logdir + "/saved/")
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            clear_devices=True)

        builder.save()"""

    print("Done")    
    

