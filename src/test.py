# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
July 2018 by Simone Francia.
simone.francia@musixmatch.com

'''
from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np



from data_load import load_test_data, load_input_vocab, load_output_vocab
from train import Graph
from nltk.translate.bleu_score import corpus_bleu



FLAGS = tf.flags.FLAGS


def eval(): 
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    X, Sources, Targets = load_test_data()
    input2idx, idx2input = load_input_vocab()
    output2idx, idx2output = load_output_vocab()

    print(len(input2idx))
    print(len(output2idx))

     
    # Start session
    with tf.Session() as sess:
        # Restore the model
        tf_saver = tf.train.Saver()
        tf_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.logdir))

        print("Restored!")

        ## Get model name
        mname = open(FLAGS.logdir + '/checkpoint', 'r').read().split('"')[1] # model name

        ## Inference
        if not os.path.exists('results'): os.mkdir('results')
        with codecs.open("results/" + mname, "w", "utf-8") as fout:
            list_of_refs, hypotheses = [], []

            for i in range(len(X) // FLAGS.batch_size):

                ### Get mini-batches
                x = X[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size]
                sources = Sources[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size]
                targets = Targets[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size]


                ### Autoregressive inference
                preds = np.zeros((FLAGS.batch_size, FLAGS.maxlen), np.int32)
                for j in range(FLAGS.maxlen):
                    _preds = sess.run(g.preds, {g.input_x: x, g.y: preds})
                    preds[:, j] = _preds[:, j]

                print(x)
                print(preds)


                ### Write to file
                for source, target, pred in zip(sources, targets, preds): # sentence-wise
                    got = " ".join(idx2output[idx] for idx in pred).split("</S>")[0].strip()
                    fout.write("- source: " + source +"\n")
                    #fout.write("- expected: " + target + "\n")
                    fout.write("- got: " + got + "\n\n")
                    fout.flush()

                    # bleu score
                    ref = target.split()
                    hypothesis = got.split()
                    if len(ref) > 3 and len(hypothesis) > 3:
                        list_of_refs.append([ref])
                        hypotheses.append(hypothesis)

            ## Calculate bleu score
            #score = corpus_bleu(list_of_refs, hypotheses)
            #fout.write("Bleu Score = " + str(100*score))
                                          
if __name__ == '__main__':
    eval()
    print("Done")
    
    