import tensorflow as tf
import numpy as np
import sys
import glob

RANDOM_TRAINING = False
LOOPAMOUNT = 1
THRESHOLD = 0.48

if RANDOM_TRAINING:
	import batchGenerator
else:
	import staticBatchGenerator
	import songBatchGenerator 

def main():
	sess = tf.InteractiveSession()
	if RANDOM_TRAINING:
		bg = batchGenerator.batchGenerator("train/features/", "jamendo_lab")	
	else:
		staticbg = staticBatchGenerator.staticBatchGenerator("train/features/", "jamendo_lab")
	
	x = tf.placeholder(tf.float32, shape=[None, 1200])
	y_ = tf.placeholder(tf.float32, shape=[None, 2])
	
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	
	x_image = tf.reshape(x, [-1,40,30,1])
	
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
	
	W_conv2 = weight_variable([3, 3, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	
	W_fc1 = weight_variable([10 * 8 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 10*8*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	#keep_prob = tf.placeholder(tf.float32)
	#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	#W_fc2 = weight_variable([1024, 2])
	#b_fc2 = bias_variable([2])

	#y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	W_fc2 = weight_variable([1024, 2])
	b_fc2 = bias_variable([2])

	h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

	h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

	W_fc3 = weight_variable([2, 2])
	b_fc3 = bias_variable([2])
	
	y_conv=tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
	
	#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])) # this one doesnt seem to work well
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
	#cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0))) # this one also works well
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	#prediction = y_conv
	#prediction = tf.argmax(y_conv, 1)
	prediction = y_conv[:,1]

	sess.run(tf.initialize_all_variables())
	saver = tf.train.Saver()

	args = sys.argv[1:]
	if int(args[0]) == 1 or int(args[0]) == 2:
		loadNetwork(saver, sess)

	if int(args[0]) == 0: # Training

		if RANDOM_TRAINING:
			for i in range(20000):
				batch = bg.getBatch(64)
				if i%100 == 0:
		 			train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
					train_loss = sess.run(cross_entropy, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
					print("step %d, training accuracy %g, loss %g"%(i, train_accuracy, train_loss))
				train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

		else: # Static training (way faster and ensures that all data in training set gets used)
			input, output = staticbg.getSet()

			i = 0
			loopCount = 0
			while i < input.shape[0] - 64:
				if i%100 == 0:
					train_accuracy = sess.run(accuracy, feed_dict={x:input[i:i+64], y_: output[i:i+64], keep_prob: 1.0})
					train_loss = sess.run(cross_entropy, feed_dict={x:input[i:i+64], y_: output[i:i+64], keep_prob: 1.0})
					print("loop %d, step %d, training accuracy %g, loss %g"%(loopCount+1, i, train_accuracy, train_loss))
				train_step.run(feed_dict={x: input[i:i+64], y_: output[i:i+64], keep_prob: 0.5})
				i += 64
			
				if i >= input.shape[0] - 64:
					if loopCount < LOOPAMOUNT-1:
						p = np.random.permutation(len(input))
						input = input[p]
						output = output[p]
						loopCount += 1
						i = 0
					else:
						break

		saveNetwork(saver, sess)
	elif int(args[0]) == 1: # Validating
		MFCCFileNames = glob.glob("valid/features/" + "/*h5")	
		
		thresholds = np.arange(0.4,0.62,0.01)
		bestThreshold = 0.5
		bestAccuracy = 0
		
		for threshold in thresholds:
			totalSongAccuracy = 0
			songCount = 0
			for MFCCFileName in MFCCFileNames:
				index = MFCCFileName.rfind("/")
				labelFileName = list("jamendo_lab") + list("/") + list(MFCCFileName[index+1:])
				labelFileName[-2] = 'l'
				labelFileName[-1] = 'a'
				labelFileName.append('b')
				
				sbg = songBatchGenerator.songBatchGenerator(MFCCFileName, "".join(labelFileName))
				songBatch = sbg.getBatch()
				
				songPrediction = prediction.eval(feed_dict={x: songBatch[0], y_: songBatch[1], keep_prob: 1.0})
				songPrediction = songPrediction > threshold
				y = songBatch[1]

				songCount += 1
				totalSongAccuracy += float(np.sum(songPrediction == y[:,1])) / songPrediction.shape[0]

			if totalSongAccuracy/songCount > bestAccuracy:
				bestAccuracy = totalSongAccuracy/songCount
				bestThreshold = threshold

		#print bestAccuracy
		print 'Best threshold found on the validation set: ', bestThreshold

	else: # Testing		
		MFCCFileNames = glob.glob("test/features/" + "/*h5")
		totalAccuracy = 0
		totalSongAccuracy = 0
		totalWindows = 0
		songCount = 0
		for MFCCFileName in MFCCFileNames:
			index = MFCCFileName.rfind("/")
			labelFileName = list("jamendo_lab") + list("/") + list(MFCCFileName[index+1:])
			labelFileName[-2] = 'l'
			labelFileName[-1] = 'a'
			labelFileName.append('b')

			sbg = songBatchGenerator.songBatchGenerator(MFCCFileName, "".join(labelFileName))
			songBatch = sbg.getBatch()

			print MFCCFileName
			print songBatch[0].shape
	
			songPrediction = prediction.eval(feed_dict={x: songBatch[0], y_: songBatch[1], keep_prob: 1.0})
			songPrediction = songPrediction > THRESHOLD
			y = songBatch[1]

			#acc = accuracy.eval(feed_dict={x: songBatch[0], y_: songBatch[1], keep_prob: 1.0}) # Does seem to perform slightly better / equally good
			acc = float(np.sum(songPrediction == y[:,1])) / songPrediction.shape[0]

			print "test accuracy %g"%acc
			print ''
			totalAccuracy += songBatch[0].shape[0] * acc
			totalSongAccuracy += acc
			totalWindows += songBatch[0].shape[0]
			songCount += 1
		
		print 'Average accuracy over all songs: %g'%(totalSongAccuracy/songCount)
		print totalSongAccuracy
		print songCount
		print 'Average accuracy over all songs, taken song length into account: %g'%(totalAccuracy/totalWindows)
	
def saveNetwork(saver, sess):
	saver.save(sess, "models/oneLoop.ckpt")
	  
def loadNetwork(saver, sess):
	saver.restore(sess, "models/oneLoop.ckpt")
	
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
