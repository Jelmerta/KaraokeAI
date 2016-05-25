#!/usr/bin/python

import glob
import os
import sys
import random
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data	
	
MEL_FEATURE_AMOUNT = 13
BLOCKS_IN_INPUT_FEATURE = 50

TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.2

DEBUG = 0

USE_HDF5 = 1
if USE_HDF5:
	import h5py

class batchGenerator():
	def __init__(self, MFCCFolderPath, labelFolderPath):
		mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
		print mnist.test.images.shape
		self.MFCCFolderPath = MFCCFolderPath
		self.labelFolderPath = labelFolderPath

		self.sets = self.split_to_sets()
	
	def split_to_sets(self):
		labelFiles = glob.glob(self.labelFolderPath + "/*.lbl")
		randomIndexList = range(0, len(labelFiles))
		random.shuffle(randomIndexList)
		cdf = np.cumsum([TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT])
		stops = map(int, cdf * len(randomIndexList))
		splitIndices = [randomIndexList[a:b] for a, b in zip([0]+stops, stops)]

		trainingSet = splitIndices[0]
		for i in range(len(trainingSet)):
			trainingSet[i] = labelFiles[trainingSet[i]]

		validationSet = splitIndices[1]
		for i in range(len(validationSet)):
			validationSet[i] = labelFiles[validationSet[i]]

		testSet = splitIndices[2]
		for i in range(len(testSet)):
			testSet[i] = labelFiles[testSet[i]]

		return [trainingSet, validationSet, testSet]

	def labelToMFCCFileName(self, labelFileName):
		index = labelFileName.rfind("/")
		MFCCFileName = list(self.MFCCFolderPath) + list("/") + list(labelFileName[index+1:])
		if USE_HDF5:
			MFCCFileName[-3] = 'h'
			MFCCFileName[-2] = '5'
			MFCCFileName[-1] = ''
		else:
			MFCCFileName[-3] = 'n'
			MFCCFileName[-2] = 'p'
			MFCCFileName[-1] = 'y'
		return "".join(MFCCFileName)
	
	def getBatch(self, setIndex, batchSize):
		randomBatch = batch(batchSize)
		batchIndex = 0
		
		while batchIndex < batchSize:
			randomLabelFileName = random.choice(self.sets[0]) # Using training set here to get a batch
			randomMFCCFileName = self.labelToMFCCFileName(randomLabelFileName)
			if os.path.isfile(randomLabelFileName):
				if os.path.isfile(randomMFCCFileName):
					if(USE_HDF5):
						h5f = h5py.File(randomMFCCFileName, 'r')
						MFCCMatrix = h5f['mfcc'][:]
						h5f.close()
					else:
						MFCCMatrix = np.load(randomMFCCFileName)
					
				else:
					if(DEBUG):
						print 'can\'t find MFCC file, continuing.'
					continue
			
				labelFile = open(randomLabelFileName, "r+")
				labelList = [char for char in labelFile.readline()]
				
				if len(labelList) == 0 or MFCCMatrix.shape[0] == 0:
					continue
			
				labelAmount = len(labelList)
				matrixFrameAmount = MFCCMatrix.shape[0]/BLOCKS_IN_INPUT_FEATURE
				
				lowestAmount = min(labelAmount, matrixFrameAmount)
				MFCCMatrix = MFCCMatrix[-lowestAmount*BLOCKS_IN_INPUT_FEATURE:]
				labelList = labelList[-lowestAmount:]
				
				randomLabelIndex = random.randint(0,lowestAmount-1)
			
				randomBatch.inputFeature[batchIndex] = MFCCMatrix[randomLabelIndex*BLOCKS_IN_INPUT_FEATURE:randomLabelIndex*BLOCKS_IN_INPUT_FEATURE+BLOCKS_IN_INPUT_FEATURE].reshape((1,BLOCKS_IN_INPUT_FEATURE*MEL_FEATURE_AMOUNT))
				if(int(labelList[randomLabelIndex]) == 1):
					randomBatch.outputFeature[batchIndex] = 1
				batchIndex += 1

			elif(DEBUG):
				print 'can\'t find MFCC file'
		
		print randomBatch.inputFeature.shape
		return randomBatch.inputFeature.reshape((1, MEL_FEATURE_AMOUNT*BLOCKS_IN_INPUT_FEATURE*batchSize)), randomBatch.outputFeature

class batch():
	def __init__(self, batchSize):
		self.batchSize = batchSize
		self.inputFeature = np.zeros((batchSize, MEL_FEATURE_AMOUNT * BLOCKS_IN_INPUT_FEATURE))
		self.outputFeature = np.zeros(batchSize)

def main():
	if (len(sys.argv) != 3):
		print 'Incorrect amount of parameters'
		sys.exit(2)

	MFCCFolderPath = sys.argv[1]
	labelFolderPath = sys.argv[2]

	bg = batchGenerator(MFCCFolderPath, labelFolderPath)

if __name__ == "__main__":
	main()
