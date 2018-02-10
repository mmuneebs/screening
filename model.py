import tensorflow as tf
from tensorflow.contrib import slim
from settings import CONFIG

CONFIG.add_arg('--labelsmooth', type=float, default=0., help='label smoothing')
CONFIG.add_arg('--padding', type=str, default='valid', help='padding for conv')
CONFIG.add_arg('--dropout', type=float, default=0.3, help='dropout rate')
ACTIVATION = tf.nn.relu
do3dconv = False
angle_channel = True


def build(batch, is_training):
    feat, labels = batch
    feat = tf.placeholder_with_default(feat, feat.get_shape())
    tf.add_to_collection('feat', feat)
    print(feat.get_shape())
    tf.summary.image('slice', feat[0:1, :, :, 0:1]/tf.reduce_max(feat[0]))  # normalize
    feat = tf.stack([feat[:, :, :, b] for b in [0]], axis=3)    # take only one/few views out of all
    tf.summary.histogram('feat', feat)
    # feat = tf.expand_dims(feat, axis=-1)
    print(feat.get_shape())

    def cnn_per_angle(a):
        a = tf.layers.conv2d(a, filters=1, kernel_size=5, strides=3, padding=CONFIG.padding)
        # a = norm(a, is_training)
        a = ACTIVATION(a)
        a = tf.layers.dropout(a, rate=CONFIG.dropout, training=is_training)
        print(a.get_shape())
        a = tf.layers.average_pooling2d(a, pool_size=4, strides=4)
        print(a.get_shape())
        # a = tf.layers.conv2d(a, filters=2, kernel_size=3, strides=2, padding=CONFIG.padding)
        # a = norm(a, is_training)
        # a = ACTIVATION(a)
        # a = tf.layers.dropout(a, rate=CONFIG.dropout, training=is_training)
        # print(a.get_shape())
        # a = tf.layers.average_pooling2d(a, pool_size=2, strides=2)
        # print(a.get_shape())

        # a = tf.layers.conv2d(a, filters=256, kernel_size=3, strides=1, padding=CONFIG.padding)
        # a = norm(a, is_training)
        # a = tf.reduce_mean(a, axis=[-2, -3])
        # print(a.get_shape())
        return a

    def cnn_3d(a):
        a = tf.layers.conv3d(a, filters=32, kernel_size=[7, 7, 3], strides=[3, 3, 2], padding='same')
        a = norm(a, is_training)
        a = ACTIVATION(a)
        print(a.get_shape())
        a = tf.layers.average_pooling3d(a, pool_size=[2, 2, 2], strides=[2, 2, 2], padding='same')
        print(a.get_shape())
        a = tf.layers.conv3d(a, filters=32, kernel_size=[3, 3, 3], strides=[2, 2, 2], padding='same')
        a = norm(a, is_training)
        a = ACTIVATION(a)
        print(a.get_shape())
        a = tf.layers.average_pooling3d(a, pool_size=[2, 2, 1], strides=[2, 2, 1], padding='valid')
        print(a.get_shape())
        a = tf.layers.conv3d(a, filters=64, kernel_size=[3, 3, 3], strides=[1, 1, 2], padding='same')
        a = norm(a, is_training)
        a = ACTIVATION(a)
        print(a.get_shape())
        a = tf.layers.conv3d(a, filters=128, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding='same')
        a = norm(a, is_training)
        a = ACTIVATION(a)
        print(a.get_shape())
        a = tf.layers.conv3d(a, filters=64, kernel_size=[3, 3, 1], strides=[1, 1, 1], padding='same')
        a = norm(a, is_training)
        a = ACTIVATION(a)
        print(a.get_shape())

        return a

    if angle_channel:
        x = cnn_per_angle(feat)
    elif not do3dconv:
        # x = tf.stack([cnn_per_angle(feat[:, :, :, b, :]) for b in xrange(64)], axis=3)
        feat = tf.transpose(feat, [3, 0, 1, 2, 4])
        x = tf.map_fn(fn=cnn_per_angle, elems=feat, parallel_iterations=64, swap_memory=False)
        x = tf.transpose(x, [1, 2, 3, 0, 4])
        # x = tf.transpose(x, [1, 2, 0])    # if already flat or reduced
        print(x.get_shape())
        # x = tf.reduce_max(x, axis=[0])
        xshape = x.get_shape().as_list()
        x = tf.reshape(x, [tf.shape(x)[0], xshape[1], xshape[2], xshape[3]*xshape[4]])
        print(x.get_shape())
        x = tf.layers.conv2d(x, filters=1, kernel_size=3, strides=1, padding=CONFIG.padding)
        x = norm(x, is_training)
        x = ACTIVATION(x)
        x = tf.layers.dropout(x, rate=CONFIG.dropout, training=is_training)
        print(x.get_shape())

    else:
        x = cnn_3d(feat)

    x = slim.flatten(x)
    print(x.get_shape())

    def _out_proj(b):
        y = tf.layers.dense(b, units=50)
        y = norm(y, is_training)
        return tf.layers.dense(y, units=2)
    # logits = tf.stack([_out_proj(x) for _ in xrange(17)], axis=1)
    # x = tf.layers.dense(x, units=50)
    # x = norm(x, is_training)
    # x = ACTIVATION(x)
    # x = tf.layers.dropout(x, rate=CONFIG.dropout, training=is_training)
    print(x.get_shape())
    x = tf.layers.dense(x, units=17*2)
    logits = tf.reshape(x, shape=[-1, 17, 2])

    print(logits.get_shape())
    smax = tf.nn.softmax(logits)
    tf.summary.histogram('train/softmax_all', smax[:, :, 1])
    tf.summary.histogram('valid/softmax_all', smax[:, :, 1], collections=['validation'])
    tf.summary.histogram('train/labels_all', labels)

    onehot_labels = tf.stack([1-labels, labels], axis=-1)
    if CONFIG.labelsmooth > 0:
        smooth_positives = 1.0 - CONFIG.labelsmooth
        smooth_negatives = CONFIG.labelsmooth / 2
        onehot_labels = onehot_labels * smooth_positives + smooth_negatives

    cost = tf.losses.log_loss(onehot_labels, smax, reduction='weighted_mean')
    pred = tf.argmax(smax, axis=2)
    accuracy, acc_op = tf.metrics.accuracy(labels, pred)

    tf.summary.scalar('train/cross_entropy', cost)
    tf.summary.scalar('valid/cross_entropy', cost, collections=['validation'])
    tf.summary.scalar('train/accuracy', acc_op)
    tf.summary.scalar('valid/accuracy', acc_op, collections=['validation'])
    tf.add_to_collection('loss', cost)
    tf.add_to_collection('softmax', smax)

    return cost, logits


def norm(inp, is_training):
    return slim.batch_norm(inp, is_training=is_training, updates_collections=None, scale=True)
