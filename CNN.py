# -*- coding: utf-8 -*-
from skimage import io, transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
from Proj2_read import Pictures
import os
ob = Pictures()


# Resized w*h
w = 100
h = 100
c = 3


# Read pictures
def read_img(ob):
    imgs = []
    labels = ob.train_data['Tag']
    for im in ob.train_data['Path']:
        print('reading the images:%s' % (im))
        img = io.imread(im)
        img = transform.resize(img, (w, h))
        imgs.append(img)
        #labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

def read_val(ob):
    imgs = []
    labels = ob.val_data['Tag']
    for im in ob.val_data['Path']:
        print('reading the images:%s' % (im))
        img = io.imread(im)
        img = transform.resize(img, (w, h))
        imgs.append(img)
        # labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


x_train, y_train = read_img(ob)
x_val, y_val = read_val(ob)

# ramdom set
num_example = x_train.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = x_train[arr]
label = y_train[arr]


# -----------------construct CNN----------------------
# placeholder
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

# first layer（100——>50)
conv1 = tf.layers.conv2d(
    inputs=x,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# second layer(50->25)
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# third layer(25->12)
conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

# fourth layer(12->6)
conv4 = tf.layers.conv2d(
    inputs=pool3,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])

# fully connect
dense1 = tf.layers.dense(inputs=re1,
                         units=1024,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
dense2 = tf.layers.dense(inputs=dense1,
                         units=512,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
logits = tf.layers.dense(inputs=dense2,
                         units=5,
                         activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
# ---------------------------net over---------------------------

loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
just_test = tf.cast(tf.argmax(logits,1), tf.int32)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# read data by batches
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# training and testing

n_epoch = 40
batch_size = 64
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for epoch in range(n_epoch):
    start_time = time.time()
    print epoch

    # training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err;
        train_acc += ac;
        n_batch += 1
    #print("   train loss: %f" % (train_loss / n_batch))
    print("   train acc: %f" % (train_acc / n_batch))

    # validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        _,err, ac = sess.run([tf.argmax(logits,1),loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
        print _
        val_loss += err;
        val_acc += ac;
        n_batch += 1
    #print("   validation loss: %f" % (val_loss / n_batch))
    print("   validation acc: %f" % (val_acc / n_batch))

#save model
saver = tf.train.Saver()
model_path = os.path.abspath('.')+'/'+'CNN_model'
save_path = saver.save(sess, model_path)
print "Model saved in file: %s" % save_path

sess.close()