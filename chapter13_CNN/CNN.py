
import numpy as np
import sklearn, scipy, matplotlib
import tensorflow as tf
import os

# parameter for mnist
height = 28
width = 28
channels = 1 # 흑백이미지를 사용
n_inputs = height * width

conv1_fmaps = 32 # 필터의 개수
conv1_ksize = 3 # 필터의 사이즈
conv1_stride = 1
conv1_pad = "SAME" # 입력과 출력의 사이즈를 동일하게 유지 / 입력과 출력의 개수는 필터의 수에 따라서 달라진다

conv2_fmaps = 64
conv2_ksize = 4
conv2_stride = 1
conv2_pad = "VALID"

conv3_fmaps = 32
conv3_ksize = 3
conv3_stride = 2
conv3_pad = "VALID"

pool3_fmaps = conv3_fmaps

n_fc1 = 64
n_fc2 = 32
n_outputs = 10

# input과 output의 상관관계
# output_height = (input_height + 2*padding_size - kernel_height)*(1/stride_vertical) + 1
# output_width = (input_width + 2*padding_size - kernel_width)*(1/stride_horizon) + 1

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")

with tf.name_scope("Conv_layers"):
    # input:28*28*batch_size / filter:32, filter_size:3, stride:1 -> output = 28*28*32*batch_size
    conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                             strides=conv1_stride, padding=conv1_pad, activation=tf.nn.relu, name="conv1")
    # input:28*28*32*batch_size / filter:64, filter_size:4, stride:1 -> output = 25*25*64*batch_size
    conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                             strides=conv2_stride, padding=conv2_pad, activation=tf.nn.relu, name="conv2")
    # input:25*25*64*batch_size / filter:32, filter_size:3, stride:2 -> output = 12*12*32*batch_size
    conv3 = tf.layers.conv2d(conv2, filters=conv3_fmaps, kernel_size=conv3_ksize,
                             strides=conv3_stride, padding=conv3_pad, activation=tf.nn.relu, name="conv3")

with tf.name_scope("pool3"):
    # input:28*28*32*batch_size / filter_size:2, stride:2 -> output = 14*14*32*batch_size
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 6 * 6])

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")

with tf.name_scope("fc2"):
    fc2 = tf.layers.dense(fc1, n_fc2, activation=tf.nn.relu, name="fc2")

with tf.name_scope("output"):
    logits = tf.layers.dense(fc2, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()




(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]



def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


n_epochs = 1000
batch_size = 100

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "배치 데이터 정확도:", acc_batch, "검증 세트 정확도:", acc_val)

    acc_test = accuracy.eval(feed_dict={X: X_test,
                                        y: y_test})
    print("테스트 세트에서 최종 정확도:", acc_test)

    save_path = saver.save(sess, "./CNN_result/my_mnist_model")
