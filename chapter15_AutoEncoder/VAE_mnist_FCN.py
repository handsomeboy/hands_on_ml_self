
import tensorflow as tf
import numpy as np
import sys
import os
import matplotlib
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)

# 그림을 저장할 폴더
PROJECT_ROOT_DIR = "."
TITLE = "VAE_mnist_FCN"

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

n_inputs = 28 * 28
# encoder spec.
n_hidden1 = 1000
n_hidden2 = 500
n_hidden3 = 250
n_hidden4 = 125

# coding unit (latent vector, z)
n_latent = 25

# decoder spec.
n_hidden6 = n_hidden4 # encoder, decoder를 대칭적으로 생성하기 위함
n_hidden7 = n_hidden3
n_hidden8 = n_hidden2
n_hidden9 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.0001
initializer = tf.variance_scaling_initializer()


with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, [None, n_inputs])

with tf.name_scope("encoder"):
    hidden1 = tf.layers.dense(X, n_hidden1, kernel_initializer=initializer, activation=tf.nn.relu, name='encoder_hidden1')
    hidden2 = tf.layers.dense(hidden1, n_hidden2, kernel_initializer=initializer, activation=tf.nn.relu, name='encoder_hidden2')
    hidden3 = tf.layers.dense(hidden2, n_hidden3, kernel_initializer=initializer, activation=tf.nn.relu, name='encoder_hidden3')
    hidden4 = tf.layers.dense(hidden3, n_hidden4, kernel_initializer=initializer, activation=tf.nn.relu, name='encoder_hidden4')

with tf.name_scope("latent"):
    mean = tf.layers.dense(hidden4, n_latent, kernel_initializer=initializer, activation=None, name='latent_mu')
    # standard deviation는 양수이므로 activation function으로 relu를 사용한다
    sigma = tf.layers.dense(hidden4, n_latent, kernel_initializer=initializer, activation=tf.nn.relu, name='latent_sigma')
    noise = tf.random_normal(tf.shape(sigma), dtype=tf.float32)
    latent = mean + sigma * noise

with tf.name_scope("decoder"):
    hidden6 = tf.layers.dense(latent, n_hidden6, kernel_initializer=initializer, activation=tf.nn.relu, name='decoder_hidden6')
    hidden7 = tf.layers.dense(hidden6, n_hidden7, kernel_initializer=initializer, activation=tf.nn.relu, name='decoder_hidden7')
    hidden8 = tf.layers.dense(hidden7, n_hidden8, kernel_initializer=initializer, activation=tf.nn.relu, name='decoder_hidden8')
    hidden9 = tf.layers.dense(hidden8, n_hidden9, kernel_initializer=initializer, activation=tf.nn.relu, name='decoder_hidden9')
    logits = tf.layers.dense(hidden9, n_outputs, kernel_initializer=initializer, activation=None)
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
n_epochs = 200
batch_size = 128

# mode = 'train'
mode = 'load'

list = []

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

plt.figure(figsize=(8,50))
for iteration in range(n_digits):
    plt.subplot(n_digits, 10, iteration + 1)
    plot_image(outputs_val[iteration])
plt.show()
