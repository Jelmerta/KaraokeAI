import batchGenerator
import tensorflow as tf

def main():
	sess = tf.InteractiveSession()
	batch = batchGenerator.getBatch("features/input", "features/output", 32)
	
	x = tf.placeholder(tf.float32, shape=[None, 784]) # 13
	y_ = tf.placeholder(tf.float32, shape=[None, 10]) # 2
	
	W = tf.Variable(tf.zeros([784,10])) # 13, 2
	b = tf.Variable(tf.zeros([10])) # 2
	
	sess.run(tf.initialize_all_variables())

	y = tf.nn.softmax(tf.matmul(x,W) + b)

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	for i in range(1000):
		batch = mnist.train.next_batch(50)
		train_step.run(feed_dict={x: batch[0], y_: batch[1]})
		
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == "__main__":
    main()