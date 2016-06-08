#!/usr/bin/python

import os
import sys
import numpy as np
	
MEL_FEATURE_AMOUNT = 30
BLOCKS_IN_WINDOW = 8
WINDOW_DURATION = 0.2

DEBUG = 1

import h5py

class songBatchGenerator():
	def __init__(self, MFCCFilePath, labelFilePath):
		self.MFCCFilePath = MFCCFilePath
		self.labelFilePath = labelFilePath

	def getLabel(self, labelFileName, time):
		with open(labelFileName, "r") as f:
			for line in f:
				currentTimingList = line.split(" ")
				if time < float(currentTimingList[1]) and time >= float(currentTimingList[0]):
					if 'nosing' in currentTimingList[2]:
						return 0
					else:
						return 1
	def getBatch(self):
		if os.path.isfile(self.labelFilePath):
			if os.path.isfile(self.MFCCFilePath):
				h5f = h5py.File(self.MFCCFilePath, 'r')
				MFCCMatrix = h5f['mfcc'][:]
				h5f.close()
			else:
				if(DEBUG):
					print 'can\'t find MFCC file, Stopping.'
				return
				
			matrixIndexAmount = MFCCMatrix.shape[0]/(BLOCKS_IN_WINDOW)		
			songBatch = batch(matrixIndexAmount-5)

			for i in range(2, matrixIndexAmount-3):
				songBatch.inputFeature[i-2] = MFCCMatrix[i*(BLOCKS_IN_WINDOW)-2*BLOCKS_IN_WINDOW:i*(BLOCKS_IN_WINDOW)+3*BLOCKS_IN_WINDOW].reshape((1,5*BLOCKS_IN_WINDOW*MEL_FEATURE_AMOUNT))
			
				indexSeconds = i * WINDOW_DURATION + WINDOW_DURATION/2

				if self.getLabel(self.labelFilePath, indexSeconds) == 0:
					songBatch.outputFeature[i-2, 0] = 1					
				else:
					songBatch.outputFeature[i-2, 1] = 1
		elif(DEBUG):
			print 'can\'t find label file'
		return songBatch.inputFeature, songBatch.outputFeature

class batch():
	def __init__(self, batchSize):
		self.batchSize = batchSize
		self.inputFeature = np.zeros((batchSize, 5*MEL_FEATURE_AMOUNT * BLOCKS_IN_WINDOW))
		self.outputFeature = np.zeros((batchSize, 2))

def main():
	if (len(sys.argv) != 3):
		print 'Incorrect amount of parameters'
		sys.exit(2)

	MFCCFilePath = sys.argv[1]
	labelFilePath = sys.argv[2]

	return songBatchGenerator(MFCCFilePath, labelFilePath)

if __name__ == "__main__":
	main()
