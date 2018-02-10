import tensorflow as tf
import numpy as np
import pandas as pd
import loader
from settings import CONFIG
dir(tf.contrib)
tf.logging.set_verbosity('INFO')

CONFIG.add_arg('--ckpt_path', type=str, default='ckpt/model.ckpt-0', help='ckpt path')
CONFIG.add_arg('--list_path', type=str, default='/Users/muneeb/ml/tsa/stage1_sample_submission.csv', help='file list')

saver = tf.train.import_meta_graph(CONFIG.ckpt_path+'.meta')
df = pd.read_csv(CONFIG.list_path)

handle = tf.get_collection('handle')[0]
feat_ph = tf.get_collection('feat')[0]
smax = tf.get_collection('softmax')[0]
prob = smax[:, :, 1]

# setup reader
csv_tld = tf.contrib.data.TextLineDataset(CONFIG.list_path).skip(1).filter(
    lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))
def linerd(line):
    record_defaults = [tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.float32)]
    f, l = tf.decode_csv(line, record_defaults)
    return f, l
csv_dec = csv_tld.map(linerd)
filemap = csv_dec.batch(17).map(
    lambda f, l: (tf.substr(f[0], 0, 32), l))  # combine into set of 17 and output (key, labels)
p_data = filemap.map(loader.read_aps, num_threads=4, output_buffer_size=8)
p_data = p_data.batch(1)

iterator = p_data.make_one_shot_iterator()
iter_str = iterator.string_handle()
next_element = iterator.get_next()
feat, labels = next_element

lgt = []
with tf.Session() as sess:
    str_handle = sess.run(iter_str)
    saver.restore(sess, CONFIG.ckpt_path)
    tf.logging.info('Restored checkpoint: {}'.format(CONFIG.ckpt_path))
    while True:
    # for i in xrange(3):
        try:
            lgt.extend(sess.run([prob], feed_dict={handle: str_handle}))
        except tf.errors.OutOfRangeError:
            print('Completed.')
            break
print('shape: {}'.format(len(lgt)))
# np.savetxt('output.txt', np.reshape(np.array(lgt)[:, 0], [-1]), fmt='%.6f')
scores = np.reshape(np.array(lgt)[:, 0], [-1])
df['Probability'] = scores
df.to_csv('sub.csv', index=False)
print('Written submission file.')
