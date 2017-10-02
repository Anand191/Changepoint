#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:18:54 2017

@author: anand
"""
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(color_codes=True)
seed = 82
np.random.seed(seed)
tf.set_random_seed(seed)

#%%Ground Truth Data
df = pd.read_csv("annotated.csv")
df['Annotate'] = df['Annotate'].astype(int)
X = np.column_stack((df['JOB_NAME'],df['MSU_CPU'],df['Z_Score'],df['Scaled'],df['Rolling_Mean'],df['Rolling_Median']))
jobs = np.unique(X[:,0])

#%%
def data_dist(job,N):
    rows = np.where(X[:,0]==job)[0]
    data = X[rows,-1]
    fdata = np.random.choice(data,N)
    gen_data = gen_dist(fdata,N)
    return fdata, gen_data
    #plt.plot(xcoord,data[:,-1])
    
    
def gen_dist(d,N):
    return (np.random.normal(size=N))
    #return (np.linspace(min(d),max(d),N) + np.random.random(N) * 0.01)    

#%%    
def linear(input, output_dim, scope=None, stddev=1.0):
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
            'w',
            [input.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input, w) + b
    
def minibatch(input, num_kernels=5, kernel_dim=3):
    x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - \
        tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input, minibatch_features], 1)

def optimizer(loss, var_list):
    learning_rate = 0.005
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        loss,
        global_step=step,
        var_list=var_list
    )
    return optimizer


def log(x):
    '''
    Sometimes discriminiator outputs can reach values close to
    (or even slightly less than) zero due to numerical rounding.
    This just makes sure that we exclude those values so that we don't
    end up with NaNs during optimisation.
    '''
    return tf.log(tf.maximum(x, 1e-5))

#%%
    
def discriminator(input, h_dim, minibatch_layer=True):
    h0 = tf.nn.softplus(linear(input, h_dim * 2, 'd0'))
    h1 = tf.nn.relu(linear(h0, h_dim * 2, 'd1'))
    h2 = tf.nn.relu(linear(h1, h_dim * 2, 'd2'))
    
        # without the minibatch layer, the discriminator needs an additional layer
        # to have enough capacity to separate the two distributions correctly
    if minibatch_layer:
        h3 = minibatch(h2)
    else:
        h3 = tf.nn.relu(linear(h1, h_dim * 2, scope='d3'))
    
    h4 = tf.sigmoid(linear(h3, 1, scope='d4'))
    return h4


def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    h1 = tf.nn.relu(linear(h0, h_dim, 'g1'))
    h2 = linear(h1, 1, 'g2')
    return h2        
    
    
#%%        
class GAN(object):
    def __init__(self, params):
        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.
        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=(params.batch_size, 1))
            self.G = generator(self.z, params.hidden_size)

        # The discriminator tries to tell the difference between samples from
        # the true data distribution (self.x) and the generated samples
        # (self.z).
        #
        # Here we create two copies of the discriminator network
        # that share parameters, as you cannot use the same network with
        # different inputs in TensorFlow.
        self.x = tf.placeholder(tf.float32, shape=(params.batch_size, 1))
        with tf.variable_scope('D'):
            self.D1 = discriminator(
                self.x,
                params.hidden_size,
                params.minibatch
            )
        with tf.variable_scope('D', reuse=True):
            self.D2 = discriminator(
                self.G,
                params.hidden_size,
                params.minibatch
            )

        # Define the loss for discriminator and generator networks
        # (see the original paper for details), and create optimizers for both
        self.loss_d = tf.reduce_mean(-log(self.D1) - log(1 - self.D2))
        self.loss_g = tf.reduce_mean(-log(self.D2))

        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        self.opt_d = optimizer(self.loss_d, self.d_params)
        self.opt_g = optimizer(self.loss_g, self.g_params)
        
#%%
def train(model,params):

    with tf.Session() as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        for step in range(params.num_steps + 1):
            # update discriminator
            get_data = data_dist("OMEC5",params.batch_size)
            x = get_data[0]
            z = get_data[1]
            loss_d, _, = session.run([model.loss_d, model.opt_d], {
                model.x: np.reshape(x, (params.batch_size, 1)),
                model.z: np.reshape(z, (params.batch_size, 1))
            })

            # update generator
            z = get_data[1]
            loss_g, _ = session.run([model.loss_g, model.opt_g], {
                model.z: np.reshape(z, (params.batch_size, 1))
            })

            if step % params.log_every == 0:
                print('{}: {:.4f}\t{:.4f}'.format(step, loss_d, loss_g))
        
        plot_output(model,session,params)    
                
                
def plot_output(model,session,params):         
                
    rows = np.where(X[:,0]=="OMEC5")[0]
    data = X[rows,-1]
    p_x = np.arange(len(data))
    
    xs = np.linspace(min(data), max(data), len(data))

    # decision boundary
    db = np.zeros((len(data), 1))
    for i in range(len(data)// params.batch_size):
        db[params.batch_size * i:params.batch_size * (i + 1)] = session.run(
            model.D1,
            {
                model.x: np.reshape(
                    xs[params.batch_size * i:params.batch_size * (i + 1)],
                    (params.batch_size, 1)
                )
            }
        )
                
    # generated samples                
    zs = np.linspace(min(data), max(data),len(data))
    g = np.zeros((len(data), 1))
    for i in range(len(data) // params.batch_size):
        g[params.batch_size * i:params.batch_size * (i + 1)] = session.run(
            model.G,
            {
                model.z: np.reshape(
                    zs[params.batch_size * i:params.batch_size * (i + 1)],
                    (params.batch_size, 1)
                )
            }
        )
                
    db_x = np.linspace(min(data), max(data), len(db))
    f, ax = plt.subplots(1)
    ax.plot(db_x, db, label='decision boundary')
    plt.plot(p_x, data, label='real data')
    plt.plot(p_x, g, label='generated data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Time values')
    plt.ylabel('X-Coordinate')
    plt.legend()
    plt.show()
#%%
def main(args):
    model = GAN(args)
    train(model,args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=10000,
                        help='the number of training steps to take')
    parser.add_argument('--hidden-size', type=int, default=5,
                        help='MLP hidden size')
    parser.add_argument('--batch-size', type=int, default=10 ,
                        help='the batch size')
    parser.add_argument('--minibatch', action='store_true',
                        help='use minibatch discrimination')
    parser.add_argument('--log-every', type=int, default=50,
                        help='print loss after this many steps')
    parser.add_argument('--anim-path', type=str, default=None,
                        help='path to the output animation file')
    parser.add_argument('--anim-every', type=int, default=1,
                        help='save every Nth frame for animation')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())