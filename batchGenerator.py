#!/usr/bin/python

import glob
import os
import sys
import random
import numpy as np
import h5py
	
MEL_FEATURE_AMOUNT = 13
BLOCKS_IN_INPUT_FEATURE = 50

TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.2

DEBUG = 1

class batchGenerator():
	def __init__(self, MFCCFolderPath, labelFolderPath):
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
		MFCCFileName[-3] = 'h'
		MFCCFileName[-2] = '5'
		MFCCFileName[-1] = ''
		return "".join(MFCCFileName)
	
	def getBatch(self, setIndex, batchSize):
		randomBatch = batch(batchSize)
		trainingExampleIndex = 0
		
		while trainingExampleIndex < batchSize:
			randomLabelFileName = random.choice(self.sets[0]) # Using training set here to get a batch
			randomMFCCFileName = self.labelToMFCCFileName(randomLabelFileName)
			if os.path.isfile(randomLabelFileName):
				if os.path.isfile(randomMFCCFileName):
					h5f = h5py.File(randomMFCCFileName, 'r')
					MFCCMatrix = h5f['mfcc'][:]
					h5f.close()
				else:
					if(DEBUG):
						print 'can\'t find MFCC file'
					continue
			
				labelFile = open(randomLabelFileName, "r+")
				labelList = [char for char in labelFile.readline()]
			
				labelAmount = statinfo = os.stat(randomLabelFileName).st_size
				print MFCCMatrix.shape[0]
				print labelAmount
				if abs(MFCCMatrix.shape[0]/BLOCKS_IN_INPUT_FEATURE - labelAmount) <= BLOCKS_IN_INPUT_FEATURE/2:
					randomLabelIndex = random.randint(0,labelAmount-2)
			
					randomBatch.inputFeature[trainingExampleIndex] = MFCCMatrix[randomLabelIndex*BLOCKS_IN_INPUT_FEATURE:randomLabelIndex*BLOCKS_IN_INPUT_FEATURE+50].reshape((1,BLOCKS_IN_INPUT_FEATURE*MEL_FEATURE_AMOUNT))
					if(int(labelList[randomLabelIndex]) == 0):
						randomBatch.outputFeature[trainingExampleIndex, 0] = 1
					elif(int(labelList[randomLabelIndex]) == 1):
						randomBatch.outputFeature[trainingExampleIndex, 1] = 1
					else:
						print 'This should not happen'
					trainingExampleIndex += 1
				elif DEBUG:
					print 'File lengths don\'t match'
			elif(DEBUG):
				print 'can\'t find MFCC file'
		
		return randomBatch.inputFeature, randomBatch.outputFeature

class batch():
	def __init__(self, batchSize):
		self.batchSize = batchSize
		self.inputFeature = np.zeros((batchSize, MEL_FEATURE_AMOUNT*BLOCKS_IN_INPUT_FEATURE))
		self.outputFeature = np.zeros((batchSize, 2))

def main():
	if (len(sys.argv) != 3):
		print 'Incorrect amount of parameters'
		sys.exit(2)

	MFCCFolderPath = sys.argv[1]
	labelFolderPath = sys.argv[2]

	bg = batchGenerator(MFCCFolderPath, labelFolderPath)

if __name__ == "__main__":
	main()
