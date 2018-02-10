import tensorflow as tf
from settings import CONFIG

CONFIG.add_arg('--ext', type=str, default='aps', help='Extension/format')
CONFIG.add_arg('--datadir', type=str, default='/datadrive/kaggle/data/', help='Data folder')
CONFIG.add_arg('--batch', type=int, default=8, help='batch size')


def setup():
    # read csv including train and val labels with filenames (no ext)
    csv_path = 'stage1_labels.csv'
    csv_tld = tf.contrib.data.TextLineDataset(csv_path).skip(1).filter(
        lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))

    def linerd(line):
        record_defaults = [tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.float32)]
        f, l = tf.decode_csv(line, record_defaults)
        return f, l

    csv_dec = csv_tld.map(linerd)
    filemap = csv_dec.batch(17).map(
        lambda f, l: (tf.substr(f[0], 0, 32), l))  # combine into set of 17 and output (key, labels)
    tr_data = filemap.take(1047).shuffle(buffer_size=1047)
    val_data = filemap.skip(1047)
    # tr_data = Dataset.from_tensor_slices((train_imgs, train_labels))
    # val_data = Dataset.from_tensor_slices((val_imgs, val_labels))

    # apply transforms
    if CONFIG.ext == 'aps':
        tr_data = tr_data.map(read_aps, num_threads=20, output_buffer_size=int(CONFIG.batch*2.5))
        val_data = val_data.map(read_aps, num_threads=10, output_buffer_size=int(CONFIG.batch*2.5))
    elif CONFIG.ext == 'a3daps':
        tr_data = tr_data.map(read_a3daps, num_threads=20, output_buffer_size=int(CONFIG.batch*2.5))
        val_data = val_data.map(read_a3daps, num_threads=10, output_buffer_size=int(CONFIG.batch*2.5))
    tr_data = tr_data.repeat(1).batch(CONFIG.batch)
    val_data = val_data.repeat(1).batch(CONFIG.batch)
    return tr_data, val_data


def read_aps(filename, label):
    # .aps
    nx = 512
    ny = 660
    nt = 16

    scaling_bytes = 4  # float32
    scaling_pos = 292  # after 292 bytes of header crap
    data_offset = 512
    data_word_size = 2
    record_bytes = data_offset + nx * ny * nt * data_word_size
    image_bytes = nx * ny * nt * data_word_size

    datadir = CONFIG.datadir
    if not CONFIG.datadir.endswith('/'):
        datadir = CONFIG.datadir + '/'
    value = tf.read_file(datadir + filename + '.aps')
    data_scale_factor = tf.decode_raw(tf.substr(value, scaling_pos, scaling_bytes), tf.float32)
    data16 = tf.decode_raw(tf.substr(value, data_offset, image_bytes), tf.uint16) #tf.int16)
    # data16 = tf.bitcast(data16, tf.uint16)
    data16 = tf.reshape(data16, [nt, ny, nx])
    data16 = tf.transpose(data16)
    dataf = tf.to_float(data16)#, tf.float32)
    data_n = dataf/65535. * (data_scale_factor * 1e8)  # roughly 0 to 2 range
    return data_n, label


def read_a3daps(filename, label):
    # .a3daps
    nx = 512
    ny = 660
    nt = 64

    scaling_bytes = 4  # float32
    scaling_pos = 292  # after 292 bytes of header crap
    data_offset = 512
    data_word_size = 2
    record_bytes = data_offset + nx * ny * nt * data_word_size
    image_bytes = nx * ny * nt * data_word_size

    datadir = CONFIG.datadir
    if not CONFIG.datadir.endswith('/'):
        datadir = CONFIG.datadir + '/'
    value = tf.read_file(datadir + filename + '.a3daps')
    data_scale_factor = tf.decode_raw(tf.substr(value, scaling_pos, scaling_bytes), tf.float32)
    data16 = tf.decode_raw(tf.substr(value, data_offset, image_bytes), tf.uint16) #tf.int16)
    # data16 = tf.bitcast(data16, tf.uint16)
    data16 = tf.reshape(data16, [nt, ny, nx])
    data16 = tf.transpose(data16)
    dataf = tf.to_float(data16)
    data_n = dataf/65535. * (data_scale_factor * 1e5)  # roughly 0 to 1 range
    return data_n, label


if __name__ == '__main__':
    # debug
    from tensorflow.python import debug as tf_debug
    file1 = tf.constant('0043db5e8c819bffc15261b1f1ac5e42')
    img, lab = read_a3daps(file1, 0)
    with tf.Session() as sess:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        d, = sess.run([img])
    print d.shape
    import cv2
    import numpy as np
    print np.max(d)
    d /= np.max(d)
    cv2.imshow('viewer', d[:, :, 0:1])
    cv2.waitKey()
