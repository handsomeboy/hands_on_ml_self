

import tensorflow as tf
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from functools import partial
# mnist dataset 정리 (train, test, valid dataset)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0 # 정규화 작업을 위한 reshape
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

# input data size & hidden layer의 neuron 개수 결정 / output으로는 10개의 class
n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 200
n_hidden3 = 100
n_outputs = 10

# input, output을 placeholder를 사용해서 정의
# 입력되는 smaple의 개수는 미정(None), # of feature = input data size
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")


training = tf.placeholder_with_default(True, shape=(), name='training')
my_batch_norm_layer = partial(tf.layers.batch_normalization, training=training, momentum=0.9)
# 구성단계
# 1. 신경망 디자인 - hidden layer:2개 (act. fn.: Relu), output:logits을 받아서 softmax에 통과시킴
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
    bn1 = my_batch_norm_layer(hidden1)
    bn1_act = tf.nn.elu(bn1)
    hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2")
    bn2 = my_batch_norm_layer(hidden2)
    bn2_act = tf.nn.elu(bn2)
    hidden3 = tf.layers.dense(bn2_act, n_hidden3, name="hidden3")
    bn3 = my_batch_norm_layer(hidden3)
    bn3_act = tf.nn.elu(bn3)
    logits_before_bn = tf.layers.dense(bn3_act, n_outputs, name="outputs")
    logits = tf.layers.batch_normalization(logits_before_bn, training=training, momentum=0.9)
    y_proba = tf.nn.softmax(logits)

# 2. cost function결정 - cross entropy함수를 이용해서 output(logits)과 target(y)의 오차를 계산,
#   이후 모든 샘플에 대한 cross entropy 평균을 계산해서 반환
with tf.name_scope("loss"):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy, name="loss")

# 3. optimizer결정 - GradientDescentOptimizer 이용
learning_rate = 0.001
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

# 4. 평가방법 - in_top_k (여기에서는 그냥 일치하는지 아닌지로 정답을 확인)
# in_top_k는 출력으로 boolean 1D tensor를 반환하기때문에 tf.cast를 사용해서 실수형으로 변환하고 평균을 구한다
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# 5. 모든 변수를 초기화하는 노드 및 훈련된 모델 파라미터를 저장하는 노드 생성
init = tf.global_variables_initializer()
saver = tf.train.Saver()


# 훈련단계
# 1. epoch, batch size 결정
#   epoch - 전체 데이터를 전부 학습하는 과정을 몇번 반복할 것인지
#   batch_size - 전체 데이터를 한번에 몇개씩 학습할 것인지
n_epochs = 40
batch_size = 50
# 2. 데이터를 랜덤하게 섞어서 미니배치를 만들어 주는 함
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X)) # 0부터 len(X)-1까지 나열된 array를 무작위로 섞어서 반환
    n_batches = len(X) // batch_size # 전체 데이터의 개수를 batch size로 나누면 전체 배치의 개수가 나옴
    for batch_idx in np.array_split(rnd_idx, n_batches): # rnd_idx array를 n_batches개씩 잘라서 반환
        X_batch, y_batch = X[batch_idx], y[batch_idx] # batch index에 해당하는 X, y의 원소들이 나와서 하나의 미니배치를 형성
        yield X_batch, y_batch

# 3. 학습과정
with tf.Session() as sess:
    # 모든 변수의 초기화 실행
    init.run()
    #
    for epoch in range(n_epochs):
        # 아래의 for문은 미니배치의 개수만큼 반복된다 (이것이 1 epoch마다 반복됨)
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch}) # 훈련데이터를 얼마나 잘 학습했는지
        acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid}) # 검증데이터를 얼마나 잘 맞추는지
        print(epoch, "배치 데이터 정확도:", acc_batch, "검증 세트 정확도:", acc_valid)

    save_path = saver.save(sess, "./fnn_with_BN_result/my_model_final.ckpt")

sess.close()
