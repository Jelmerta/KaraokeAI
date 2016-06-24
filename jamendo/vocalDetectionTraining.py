import tensorflow as tf
import numpy as np
import sys
import glob
import random
import matplotlib.pyplot as pp
from scipy import ndimage
from scipy import stats
from collections import Counter

RANDOM_TRAINING = False
LOOPAMOUNT = 1
SHOW_PLOTS = False
GAUSSIAN_SMOOTHING = False # Doesnt deliver good results, tops too low
MEDIAN_SMOOTHING = True # Also surprisingly doesn't help much


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
		statictrainbg = staticBatchGenerator.staticBatchGenerator("train/features/", "jamendo_lab")
	
	x = tf.placeholder(tf.float32, shape=[None, 3360])
	y_ = tf.placeholder(tf.float32, shape=[None, 2])
	
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	
	x_image = tf.reshape(x, [-1,84,40,1])
	
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
	
	W_conv2 = weight_variable([3, 3, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	h_pool2_flat = tf.reshape(h_pool2, [-1, 21*10*64])
	
	W_fc1 = weight_variable([21 * 10 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	W_fc2 = weight_variable([1024, 2])
	b_fc2 = bias_variable([2])

	h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

	W_fc3 = weight_variable([2, 2])
	b_fc3 = bias_variable([2])
	
	y_conv = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)
	#y_conv = tf.nn.sigmoid(tf.matmul(h_fc2, W_fc3) + b_fc3)
	
	#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])) # this one doesnt seem to work well
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
	#cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0))) # this one also works well
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
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
			traininput, trainoutput = statictrainbg.getSet()
			#traininputMean = np.mean(traininput, axis=0)
			#traininputStd = np.std(traininput, axis=0)
			traininputMean = 0
			traininputStd = 1			
			normalizedtrainInput = (traininput - traininputMean) / traininputStd

			staticvalidbg = staticBatchGenerator.staticBatchGenerator("valid/features/", "jamendo_lab")
			validinput, validoutput = staticvalidbg.getSet()
			
			normalizedvalidInput = (validinput - traininputMean) / traininputStd

			i = 0
			loopCount = 0
			while i < traininput.shape[0] - 64:
				if i%100 == 0:

					train_accuracy = sess.run(accuracy, feed_dict={x:normalizedtrainInput[i:i+64], y_: trainoutput[i:i+64], keep_prob: 1.0})
					train_loss = sess.run(cross_entropy, feed_dict={x:normalizedtrainInput[i:i+64], y_: trainoutput[i:i+64], keep_prob: 1.0})
					print("loop %d, step %d, training accuracy %g, loss %g"%(loopCount+1, i, train_accuracy, train_loss))

				train_step.run(feed_dict={x: normalizedtrainInput[i:i+64], y_: trainoutput[i:i+64], keep_prob: 0.5})

					
				if i%6400 == 0:
					idx = random.randint(0, normalizedvalidInput.shape[0]-500)

					valid_accuracy = sess.run(accuracy, feed_dict={x:normalizedvalidInput[idx:idx+500], y_: validoutput[idx:idx+500], keep_prob: 1.0})
					valid_loss = sess.run(cross_entropy, feed_dict={x:normalizedvalidInput[idx:idx+500], y_: validoutput[idx:idx+500], keep_prob: 1.0}) 
					print("loop %d, step %d, valid accuracy %g, loss %g"%(loopCount+1, i, valid_accuracy, valid_loss))

				i += 64
			
				if i >= traininput.shape[0] - 64:
					if loopCount < LOOPAMOUNT-1:
						p = np.random.permutation(len(traininput))
						normalizedInput = traininput[p]
						trainoutput = trainoutput[p]
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
		totalSpec = 0
		totalRecall = 0
		totalPrecision = 0
		totalFmeasure = 0
	
		totalSongAccuracy = 0

		vocalWindowCount = 0
		totalClassifications = 0
		songCount = 0

		for MFCCFileName in MFCCFileNames:
			index = MFCCFileName.rfind("/")
			labelFileName = list("jamendo_lab") + list("/") + list(MFCCFileName[index+1:])
			labelFileName[-2] = 'l'
			labelFileName[-1] = 'a'
			labelFileName.append('b')

			sbg = songBatchGenerator.songBatchGenerator(MFCCFileName, "".join(labelFileName))
			songBatch = sbg.getBatch()
			y = songBatch[1]
			songLength = y.shape[0]

			print MFCCFileName
			print songBatch[0].shape

			songPrediction = prediction.eval(feed_dict={x: songBatch[0], y_: songBatch[1], keep_prob: 1.0})
			if SHOW_PLOTS:
				fig = pp.figure()
				pp.plot(songPrediction)
				pp.axis([0, songLength, 0, 0.6])
				pp.show()

			songPredictionThreshold = songPrediction > THRESHOLD	
			if SHOW_PLOTS:
				fig = pp.figure()
				pp.imshow(np.logical_not(songPredictionThreshold).reshape(1, songLength), aspect='auto', cmap = pp.cm.gray, interpolation='nearest')
				pp.axis([0, songLength, 0, 0.5])
				pp.show()
			
			vocalWindowCount += sum(y[:,1])
			if SHOW_PLOTS:
				fig = pp.figure()
				pp.imshow(np.logical_not(y[:, 1]).reshape(1, songLength), aspect='auto', cmap = pp.cm.gray, interpolation='nearest')
				pp.axis([0, songLength, 0, 0.5])
				pp.show()

			#acc = accuracy.eval(feed_dict={x: songBatch[0], y_: songBatch[1], keep_prob: 1.0}) # without different threshold: Does seem to perform slightly better / equally good
			if GAUSSIAN_SMOOTHING:
				songPredictionSmoothed = ndimage.gaussian_filter1d(songPrediction, 1)

				if SHOW_PLOTS:
					fig = pp.figure()
					pp.plot(songPredictionSmoothed)
					pp.axis([0, songLength, 0, 0.6])
					pp.show()

			if MEDIAN_SMOOTHING:
				songPredictionSmoothed = medfilt(songPrediction, 9)

				if SHOW_PLOTS:
					fig = pp.figure()
					pp.plot(songPredictionSmoothed)
					pp.axis([0, songLength, 0, 0.6])
					pp.show()

			if GAUSSIAN_SMOOTHING:
				songPredictionSmoothedThresholded = songPredictionSmoothed > THRESHOLD
				acc = float(np.sum(songPredictionSmoothedThresholded == y[:,1])) / songLength

			elif MEDIAN_SMOOTHING:
				songPredictionSmoothedThresholded = songPredictionSmoothed > THRESHOLD
				
				counts = Counter(zip(songPredictionSmoothedThresholded, y[:,1]))
				true_pos = float(counts[1, 1])				
				false_pos = float(counts[1, 0])
				false_neg = float(counts[0, 1])
				true_neg = float(counts[0, 0])			

				acc = (true_pos + true_neg) / songLength 
				spec = true_neg / (false_pos + true_neg)
				recall = true_pos / (true_pos + false_neg)
				precision = true_pos / (true_pos + false_pos)
				fmeasure = 2 * (precision * recall) / (precision + recall)
			else:
				acc = float(np.sum(songPredictionThreshold == y[:,1])) / songLength

			print "test accuracy: %g"%acc
			print "f-measure: %g"%fmeasure
			print ''
			totalAccuracy += songLength * acc
			totalSpec += songLength * spec
			totalRecall += songLength * recall
			totalPrecision += songLength * precision
			totalFmeasure += songLength * fmeasure
			totalSongAccuracy += acc
			
			totalClassifications += songLength
			songCount += 1
		
		print 'Average accuracy over all songs, not taking song length into account: %g'%(totalSongAccuracy/songCount)
		print 'Average accuracy over all songs, taking song length into account: %g'%(totalAccuracy/totalClassifications)
		print 'Average spec over all songs, taking song length into account: %g'%(totalSpec/totalClassifications)
		print 'Average recall over all songs, taking song length into account: %g'%(totalRecall/totalClassifications)
		print 'Average precision over all songs, taking song length into account: %g'%(totalRecall/totalClassifications)
		print 'Average fmeasure over all songs, taking song length into account: %g'%(totalFmeasure/totalClassifications)
		print 'Percentage of vocal windows in the test set: %g'%(float(vocalWindowCount)/totalClassifications)
	
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

def medfilt (x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)
	
if __name__ == "__main__":
    main()
