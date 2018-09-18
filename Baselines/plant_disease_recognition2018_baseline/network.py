import tensorflow as tf
LEARNINGRATE = 1e-3

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape, bais=0.1):
    initial = tf.constant(bais, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, [1, 1, 1, 1], 'SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


def max_pool_3x3(x):
    return tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')


def avg_pool_3x3(x):
    return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')


def inference(features, one_hot_labels):
    # network structure
    # conv1
    W_conv1 = weight_variable([5, 5, 3, 64], stddev=1e-4)
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(features, W_conv1) + b_conv1)
    h_pool1 = max_pool_3x3(h_conv1)
    # norm1
    norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    # conv2
    W_conv2 = weight_variable([5, 5, 64, 64], stddev=1e-2)
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(norm1, W_conv2) + b_conv2)
    # norm2
    norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    h_pool2 = max_pool_3x3(norm2)

    # conv3
    W_conv3 = weight_variable([5, 5, 64, 64], stddev=1e-2)
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_3x3(h_conv3)

    # fc1
    W_fc1 = weight_variable([16 * 16 * 64, 128])
    b_fc1 = bias_variable([128])
    h_pool3_flat = tf.reshape(h_pool3, [-1, 16*16*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # introduce dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # fc2
    W_fc2 = weight_variable([128, 80])
    b_fc2 = bias_variable([80])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    # calculate loss
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=y_conv))
    train_step = tf.train.AdamOptimizer(LEARNINGRATE).minimize(cross_entropy)
    
return train_step, cross_entropy, y_conv, keep_prob
