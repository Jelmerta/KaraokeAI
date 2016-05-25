import batchGenerator 
import tensorflow as tf

def main():
	sess = tf.InteractiveSession()
	bg = batchGenerator.batchGenerator("../features/input", "../features/output")
	
	x = tf.placeholder(tf.float32, shape=[None, 650])
	y_ = tf.placeholder(tf.float32, shape=[None, 1])
	
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	
	x_image = tf.reshape(x, [-1,50,13,1]) # (could be 50/13)
	
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
	
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	
	W_fc1 = weight_variable([13 * 4 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 13*4*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	W_fc2 = weight_variable([1024, 1])
	b_fc2 = bias_variable([1])

	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	
	#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
	#cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
	# use one of these if doesn't work
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess.run(tf.initialize_all_variables())
	for i in range(1000):
		print i
		batch = bg.getBatch(0, 64)
		if i%10 == 0:
			train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
			loss = sess.run(cross_entropy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
			print("step %d, training accuracy %g, loss %g"%(i, train_accuracy, loss))
		train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
		
	
	randomTestSet = bg.getBatch(1, 200) # multiple of the same data might be used, but doesn't matter since data is incredibly big
	print("test accuracy %g"%accuracy.eval(feed_dict={x: randomTestSet[0], y_: randomTestSet[1], keep_prob: 1.0}))
	#saveNetwork()
	
# def saveNetwork():
	# saver = tf.train.Saver(tf.all_variables())
	
	  # print("Model saved in file: %s" % save_path)
	  
# def loadNetwork():
	# v1 = tf.Variable(..., name="v1")
	# v2 = tf.Variable(..., name="v2")
	#...
	# Add ops to save and restore all the variables.
	# saver = tf.train.Saver()

	# Later, launch the model, use the saver to restore variables from disk, and
	# do some work with the model.
	# with tf.Session() as sess:
	  # Restore variables from disk.
	  # saver.restore(sess, "/tmp/model.ckpt")
	  # print("Model restored.")
	  # Do some work with the model
	  # ...
	
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)	
	
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
	
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	
if __name__ == "__main__":
    main()