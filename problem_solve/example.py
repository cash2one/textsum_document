import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [2,2])
with tf.variable_scope('train'):
    w1 = tf.get_variable('w1', [2,2])
#---------------------1-----------------------
  #  b = tf.constant(np.array([[1,2],[3,4]], np.float32), name='b') # comment this code has the same function with the next three line codes
#---------------------1---------------------
#---------------------2--------------------
 #   init = np.array([[1,2],[3,4]])
 #   b = tf.get_variable('b', [2,2], initializer=tf.constant_initializer(init))
 #   b = tf.stop_gradient(b) # we use tf.stop_gradient(b) to fix b
#--------------------2----------------------
# The block code in 1, 2, 3 has the same function
#--------------------3----------------------
    init = np.array([[1,2],[3,4], [5,6]])
    B = tf.get_variable('B', [3,2], initializer=tf.constant_initializer(init))
    B = tf.stop_gradient(B) # we use tf.stop_gradient(B) to fix B
    b = tf.nn.embedding_lookup(B, [1,2])
#-------------------3------------------------

if w1.name == 'train/w1:0':
    print 'good'
#tf.scalar_summary('w1', 1)
#tf.scalar_summary('b', b)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
loss = tf.reduce_sum(tf.square(x*w1-b*x))
tf.scalar_summary('loss', loss)
summary  = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('./summary')

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(100):
        if i%100 == 0:
            print i
        feed_dict = {x: np.ones([2,2])}
        _, summary_str = sess.run([optimizer.minimize(loss), summary], feed_dict=feed_dict)
        print sess.run(w1)
        print sess.run(b)
        summary_writer.add_summary(summary_str)
    
