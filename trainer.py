import tensorflow as tf
from tensorflow.contrib.data import Iterator
from tensorflow.python import debug as tf_debug
import numpy as np
import os
import loader
import model
from settings import CONFIG
tf.logging.set_verbosity('INFO')

CONFIG.add_arg('--debug', dest='debug', action='store_true', help='debug')
CONFIG.set_defaults(debug=False)
CONFIG.add_arg('--epochs', type=int, default=30, help='max epochs')
CONFIG.add_arg('--lrate', type=float, default=0.2, help='init learning rate')
CONFIG.add_arg('--traindir', type=str, default='/ml/tsa/ckpt', help='train/ckpt dir')
CONFIG.add_arg('--decaysteps', type=int, default=50, help='decay steps')
CONFIG.add_arg('--clip_grad', type=bool, default=False, help='Clip gradients')

trn_data, val_data = loader.setup()

# create Iterators
handle = tf.placeholder(tf.string, shape=[])
tf.add_to_collection('handle', handle)
is_training_ph = tf.placeholder_with_default(False, [])
iterator = Iterator.from_string_handle(handle, trn_data.output_types, trn_data.output_shapes)
# iterator = Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)
trn_iter = trn_data.make_initializable_iterator()
val_iter = val_data.make_initializable_iterator()
trn_str = trn_iter.string_handle()
val_str = val_iter.string_handle()
next_element = iterator.get_next()

# build model and optimizer
cost, logits = model.build(next_element, is_training_ph)

global_step = tf.train.get_or_create_global_step()
lrate = tf.train.exponential_decay(learning_rate=CONFIG.lrate, global_step=global_step, decay_steps=CONFIG.decaysteps,
                                   decay_rate=0.96, staircase=True)
tf.summary.scalar('lrate', lrate)
opt = tf.train.AdagradOptimizer(learning_rate=lrate)
gradvars = opt.compute_gradients(cost)
grads, gvars = zip(*gradvars)
global_norm = tf.global_norm(grads)
tf.summary.scalar('global_norm', global_norm)
if CONFIG.clip_grad:
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=5., use_norm=global_norm)    # gradient clipping
    gradvars = zip(grads, gvars)
# for grad, var in gradvars:
#     if grad is not None:
#         tf.summary.histogram(var.op.name + '_gradients', grad)
#         tf.summary.histogram(var.op.name, var)
train_op = opt.apply_gradients(gradvars, global_step=global_step)
with tf.control_dependencies([train_op]):
    train_op = tf.identity(cost)

bsaver = tf.train.Saver()
sop = tf.summary.merge_all()
vsop = tf.summary.merge_all(key='validation')
# hooks = []
# if DEBUG:
#     dbg_hook = tf_debug.LocalCLIDebugHook()
#     dbg_hook.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)
#     hooks.append(dbg_hook)
# sess = tf.train.MonitoredTrainingSession(chief_only_hooks=hooks, checkpoint_dir='ckpt', save_summaries_secs=600)
sv = tf.train.Supervisor(logdir=CONFIG.traindir, summary_op=None)    # summary thread runs without iterator init

step = 0
vstep = 0
epoch = 0
cea = []
best_loss = 1000.
is_training = True
with sv.managed_session() as sess:
    if CONFIG.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)
    trn_handle = sess.run(trn_str)
    val_handle = sess.run(val_str)
    sess.run(trn_iter.initializer)
    print('Starting training.')

    while not sv.should_stop():
        try:
            # elem = sess.run(next_element)
            # for e in elem: print(e.shape)
            if is_training:
                if step % 10 == 0:  # write summaries
                    ce, step, sstr = sess.run([train_op, global_step, sop],
                                              feed_dict={handle: trn_handle, is_training_ph: is_training})
                    sv.summary_computed(sess, sstr, step)
                else:
                    ce, step = sess.run([train_op, global_step],
                                        feed_dict={handle: trn_handle, is_training_ph: is_training})
            else:
                if vstep % 5 == 0:
                    ce, sstr = sess.run([cost, vsop], feed_dict={handle: val_handle})
                    sv.summary_computed(sess, sstr, vstep)
                else:
                    ce, = sess.run([cost], feed_dict={handle: val_handle})
                vstep += 1
            cea.append(ce)
        except tf.errors.OutOfRangeError:
            ceam = np.mean(cea)
            if is_training:
                epoch += 1
                print('epoch: {}, train_error: {:.4f}'.format(epoch, ceam))
                sess.run(val_iter.initializer)
                vstep = step
            else:
                print('epoch: {}, valid_error: {:.4f}'.format(epoch, ceam))
                if ceam < best_loss:
                    bsaver.save(sess, os.path.join(CONFIG.traindir, 'best'), global_step=global_step,
                                latest_filename='best')
                    tf.logging.info('Saved best model.')
                    best_loss = ceam
                if epoch == CONFIG.epochs:
                    break
                sess.run(trn_iter.initializer)
            cea = []
            is_training = not is_training

    print('End of training, closing session.')
print('PROGRAM END')
