## %%
import math
import numpy as np
import tensorflow as tf
import os
os.chdir('/home/oleksandr/Dropbox/Deep_Learning/NLTP_Stanford/TF')
from utils import *


# []
batch_size = 128
vocabluary_size = 50000
embedding_size = 128
num_sampled = 64
train_data, val_data, reverse_dict = load_data()


#
def skipgram():
    batch_inputs = tf.placeholder(tf.int32,shape=[batch_size,])
    batch_labels = tf.placeholder(tf.int64,shape=[batch_size,1])
    val_dataset  = tf.constant(val_data,dtype=tf.int32)

    with tf.variable_scope('word2vec') as scope:
        embeddings = tf.Variable(tf.random_uniform([vocabluary_size,embedding_size],
                                                    -1.0,1.0))

        batch_embeddings = tf.nn.embedding_lookup(embeddings,batch_inputs)
        weights = tf.Variable(tf.truncated_normal([vocabluary_size,embedding_size],
                                                    stddev=1.0/math.sqrt(embedding_size)))
        biases = tf.Variable(tf.zeros([vocabluary_size]))

        loss = tf.reduce_mean(tf.nn.nce_loss(weights=weights,
                                             biases=biases,
                                             labels=batch_labels,
                                             inputs=batch_embeddings,
                                             num_sampled=num_sampled,
                                             num_classes=vocabluary_size))

        norm = tf.sqrt(tf.reduce_mean(tf.square(embeddings),1,keepdims=True))
        normalized_embeddings = embeddings/norm

        val_embeddings = tf.nn.embedding_lookup(normalized_embeddings,val_dataset)
        similarity = tf.matmul(val_embeddings,normalized_embeddings,transpose_b=True)
        #similarity = 0


        return batch_inputs,  batch_labels, normalized_embeddings, loss, similarity

def run():
    batch_inputs, batch_labels, normalized_embeddings, loss, similarity = skipgram()
    init = tf.global_variables_initializer()
    op = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    with tf.Session() as sess:
        sess.run(init)
        avg_loss = 0.0

        for step, batch_data in enumerate(train_data):
            inputs, labels = batch_data
            feed_dict={batch_inputs:inputs, batch_labels:labels}
            _, loss_val = sess.run([op,loss],feed_dict=feed_dict)
            avg_loss += loss_val

            if step % 1000 == 0:
                if step > 0:
                    avg_loss /= 1000
                print('loss at iter', step, ":", avg_loss)
                avg_loss = 0.0

            if step % 5000 == 0:
                sim = similarity.eval()
                for i in range(len(val_data)):
                    top_k = 8
                    nearest = (sim[i,:]).argsort()[1:top_k+1]
                    print_closest_words(i,nearest,reverse_dict)

        final_embeddings = normalized_embeddings.eval()
        return final_embeddings

final_embeddings = run()


visualize_embeddings(final_embeddings,reverse_dict)
