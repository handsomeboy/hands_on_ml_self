
import tensorflow as tf
import numpy as np
import sys
import os
import matplotlib
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)

# 그림을 저장할 폴더
PROJECT_ROOT_DIR = "."
TITLE = "VAE_mnist_CNN"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", fig_id + '_' + TITLE + ".png")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")

def plot_multiple_images(images, n_rows, n_cols, pad=2):
    images = images - images.min()
    w,h = images.shape[1:]
    image = np.zeros(((w+pad)*n_rows+pad, (h+pad)*n_cols+pad))
    for y in range(n_rows):
        for x in range(n_cols):
            image[(y*(h+pad)+pad):(y*(h+pad)+pad+h),(x*(w+pad)+pad):(x*(w+pad)+pad+w)] = images[y*n_cols+x]
    plt.imshow(image, cmap="Greys", interpolation="nearest")
    plt.axis("off")

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

# make mnist dataset
def import_mnist_dataset():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:5000], X_train[5000:]
    y_valid, y_train = y_train[:5000], y_train[5000:]
    return X_train, X_valid, y_train, y_valid


reset_graph()

# parameter for mnist
height = 28
width = 28
channels = 1 # 흑백이미지를 사용
n_inputs = height * width

# encoder spec.
conv1_fmaps = 64 # 필터의 개수
conv1_ksize = 4 # 필터의 사이즈
conv1_stride = 1
conv1_pad = "VALID"

conv2_fmaps = 32
conv2_ksize = 4
conv2_stride = 1
conv2_pad = "VALID"

# coding unit (latent vector, z)
n_latent = 25

# decoder spec.
conv3_fmaps = 8
conv3_ksize = 15
conv3_stride = 1
conv3_pad = "VALID"

conv4_fmaps = 1
conv4_ksize = 10
conv4_stride = 1
conv4_pad = "VALID"

n_outputs = n_inputs

learning_rate = 0.001
initializer = tf.variance_scaling_initializer()



with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, [None, n_inputs])
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])

with tf.name_scope("encoder"):
    # input:28*28*batch_size / filter:32, filter_size:4, stride:1 -> output = 25*25*32*batch_size
    conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize, strides=conv1_stride, padding=conv1_pad, activation=tf.nn.relu, name="conv1")
    # input:25*25*32*batch_size / filter:16, filter_size:4, stride:1 -> output = 22*22*16*batch_size
    conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize, strides=conv2_stride, padding=conv2_pad, activation=tf.nn.relu, name="conv2")
    # conv2_flat = tf.reshape(conv2, shape=[-1, 22 * 22 * conv2_fmaps])
    conv2_flat = tf.layers.flatten(conv2)

with tf.name_scope("latent"):
    # input:(22*22*16)*batch_size -> output = n_latent*batch_size
    mean = tf.layers.dense(conv2_flat, n_latent, kernel_initializer=initializer, activation=None, name='latent_mu')
    sigma = tf.layers.dense(conv2_flat, n_latent, kernel_initializer=initializer, activation=None, name='latent_sigma')
    noise = tf.random_normal(tf.shape(sigma), dtype=tf.float32)
    latent = mean + sigma * noise

with tf.name_scope("decoder"):
    # input:20*batch_size -> output = 20 * batch_size
    latent_reshape = tf.reshape(latent, shape=[-1, 5 , 5 , channels])
    conv3 = tf.layers.conv2d_transpose(latent_reshape, filters=conv3_fmaps, kernel_size=conv3_ksize, strides=conv3_stride, padding=conv3_pad, activation=tf.nn.relu, name="conv3")
    conv4 = tf.layers.conv2d_transpose(conv3, filters=conv4_fmaps, kernel_size=conv4_ksize, strides=conv4_stride, padding=conv4_pad, activation=tf.nn.relu, name="conv4")
    conv4_flat = tf.reshape(conv4, shape=[-1, height*width*channels])
    logits = tf.layers.dense(conv4_flat, n_outputs, kernel_initializer=initializer, activation=None)
    outputs = tf.sigmoid(logits)

with tf.name_scope("loss"):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)
    reconstruction_loss = tf.reduce_sum(cross_entropy)
    eps = 1e-10
    latent_loss = 0.5*tf.reduce_sum(tf.square(mean) + tf.square(sigma)-1 -tf.log(eps + tf.square(sigma)))
    total_loss = reconstruction_loss + latent_loss

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(total_loss)


init = tf.global_variables_initializer()
saver = tf.train.Saver()

X_train, X_valid, y_train, y_valid = import_mnist_dataset()

n_digits = 10
n_epochs = 50
batch_size = 128

mode = 'train'
# mode = 'load'


with tf.Session() as sess:
    if mode == 'train':
        init.run()
        for epoch in range(n_epochs):
            # n_batches = len(X_train) // batch_size
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch})
            loss_val, reconstruction_loss_val, latent_loss_val = sess.run([total_loss, reconstruction_loss, latent_loss], feed_dict={X: X_batch})
            print("\r{}".format(epoch), "\ttotal training loss:", loss_val, "\treconstruction loss:", reconstruction_loss_val, "\tlatent loss:", latent_loss_val)
            saver.save(sess, "./" + TITLE)
        codings_rnd = np.random.normal(size=[n_digits, n_latent])
        outputs_val = outputs.eval(feed_dict={latent: codings_rnd})

    elif mode == 'load':
        new_saver = tf.train.import_meta_graph(TITLE + '.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
        codings_rnd = np.random.normal(size=[n_digits, n_latent])
        outputs_val = outputs.eval(feed_dict={latent: codings_rnd})

plt.figure(figsize=(28,28))
# for iteration in range(n_digits):
#     plt.subplot(n_digits, 10, iteration + 1)
#     plot_image(outputs_val[iteration])
    # plt.show()


n_rows = 2
n_cols = 5
plot_multiple_images(outputs_val.reshape(-1, 28, 28), n_rows, n_cols)
save_fig("plot")
plt.show()
