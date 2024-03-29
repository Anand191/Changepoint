#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 13:34:15 2017

@author: anand
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#%%
sns.set(color_codes=True)
df = pd.read_csv("annotated.csv")
df['Annotate'] = df['Annotate'].astype(int)
X = np.column_stack((df['JOB_NAME'],df['MSU_CPU'],df['Z_Score'],df['Scaled'],df['Rolling_Mean'],df['Rolling_Median']))
jobs = np.unique(X[:,0])
rows = np.where(X[:,0]=='OMEC5')[0]
data = X[rows,-1]
#%%
def data_dist(start,end):    
    fdata = data[start:end]
    return fdata

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
    learning_rate = 0.001
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
def gen_dist(start,end):
    ndata =  data_dist(start,end)
    #print ndata
    return(np.linspace(min(ndata),max(ndata),(end-start)) + np.random.random((end-start)) * 0.01)
        
def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    h1 = linear(h0, 1, 'g1')
    return h1
    

def discriminator(input, h_dim, minibatch_layer=True):
    h0 = tf.nn.relu(linear(input, h_dim * 2, 'd0'))
    h1 = tf.nn.relu(linear(h0, h_dim * 2, 'd1'))

    # without the minibatch layer, the discriminator needs an additional layer
    # to have enough capacity to separate the two distributions correctly
    if minibatch_layer:
        h2 = minibatch(h1)
    else:
        h2 = tf.nn.relu(linear(h1, h_dim * 2, scope='d2'))

    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
    return h3        
        
#%%
        
class GAN():
    def __init__(self,size):
        self.hidden_size = size
        self.minibatch = True
        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.        
        with tf.variable_scope('G'):  
            self.z = tf.placeholder(tf.float32, shape=(None, 1))
            self.G = generator(self.z, self.hidden_size)

        # The discriminator tries to tell the difference between samples from
        # the true data distribution (self.x) and the generated samples
        # (self.z).
        #
        # Here we create two copies of the discriminator network
        # that share parameters, as you cannot use the same network with
        # different inputs in TensorFlow.
        self.x = tf.placeholder(tf.float32, shape=(None, 1))
        with tf.variable_scope('D'):
            self.D1 = discriminator(
                self.x,
                self.hidden_size
            )
        with tf.variable_scope('D', reuse=True):
            self.D2 = discriminator(
                self.G,
                self.hidden_size
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
def train():
    epochs = 100
    batch = 50
    model = GAN(100)
    print('{}: {}\t{}'.format(0, 'Disc Loss', 'Gen Loss'))
    

    with tf.Session() as session:
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        session.run(init)

        for epoch in xrange(epochs+1):
            cursor = 0
            end = 0
            while (end<len(data)):
                if((cursor + batch) >= len(data)):
                    end = len(data)
                else:
                    end = cursor + batch
                x = data_dist(cursor,end)
                z = gen_dist(cursor,end)
                #model = GAN(cursor,end)
                loss_d, _, = session.run([model.loss_d, model.opt_d], {
                    model.x: np.reshape(x, ((end-cursor), 1)),
                    model.z: np.reshape(z, ((end-cursor), 1))
                })
                
                z = gen_dist(cursor,end)
                loss_g, _ = session.run([model.loss_g, model.opt_g], {
                model.z: np.reshape(z, ((end-cursor), 1))
                })                
                cursor = end
                
            if epoch % 10 == 0:
                print('{}: {:.4f}\t{:.4f}'.format(epoch, loss_d, loss_g))
                
        plot_output(model,session)
                
                
#%%
def plot_output(model,session):         
    p_x = np.arange(len(data))    
    zs = np.linspace(min(data),max(data),len(data))
    # decision boundary
    db = session.run(
            model.D1,
            {
                model.x: np.reshape(zs,(len(data), 1))
            }
        )
                
    # generated samples                
    
    g = session.run(
            model.G,
            {
                model.z: np.reshape(zs,(len(data), 1))
            }
        )
    db_x = np.arange(len(db))
    f, ax = plt.subplots(1,figsize=(14,10))
    ax.plot(db_x, db, label='decision boundary')
    plt.plot(p_x, data, label='real data')
    plt.plot(p_x, g, label='generated data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Time values')
    plt.ylabel('X-Coordinate')
    plt.legend()
    plt.show()
    
#%%
train()
        
        
        
        