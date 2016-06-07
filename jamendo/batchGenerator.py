#!/usr/bin/python

import glob
import os
import sys
import random
import numpy as np
	
MEL_FEATURE_AMOUNT = 30
BLOCKS_IN_WINDOW = 8
WINDOW_DURATION = 0.2

DEBUG = 0

import h5py

class batchGenerator():
	def __init__(self, MFCCFolderPath, labelFolderPath):
		self.MFCCFolderPath = MFCCFolderPath
		self.labelFolderPath = labelFolderPath
		self.getMFCCFiles()
	
	def getMFCCFiles(self):
		self.MFCCFileNames = glob.glob(self.MFCCFolderPath + "/*h5")
		random.shuffle(self.MFCCFileNames)

	def MFCCToLabelFileName(self, MFCCFileName):
		index = MFCCFileName.rfind("/")
		labelFileName = list(self.labelFolderPath) + list("/") + list(MFCCFileName[index+1:])
		labelFileName[-2] = 'l'
		labelFileName[-1] = 'a'
		labelFileName.append('b')

		return "".join(labelFileName)
	
	def getLabel(self, labelFileName, time):
		with open(labelFileName, "r") as f:
			for line in f:
				currentTimingList = line.split(" ")
				if time < float(currentTimingList[1]) and time >= float(currentTimingList[0]):
					if 'nosing' in currentTimingList[2]:
						return 0
					else:
						return 1
	
	def getBatch(self, batchSize):
		randomBatch = batch(batchSize)
		batchIndex = 0
		
		while batchIndex < batchSize:
			randomMFCCFileName = random.choice(self.MFCCFileNames)
			randomLabelFileName = self.MFCCToLabelFileName(randomMFCCFileName)

			if os.path.isfile(randomLabelFileName):
				if os.path.isfile(randomMFCCFileName):
					h5f = h5py.File(randomMFCCFileName, 'r')
					MFCCMatrix = h5f['mfcc'][:]
					h5f.close()
				else:
					if(DEBUG):
						print 'can\'t find MFCC file, continuing.'
					continue
				
				matrixIndexAmount = MFCCMatrix.shape[0]/(BLOCKS_IN_WINDOW)
				randomWindowIndex = random.randint(2, matrixIndexAmount-3)

				#print ''
				#print randomLabelFileName
				#print MFCCMatrix.shape
				#print randomWindowIndex*(BLOCKS_IN_WINDOW/2)
				#print indexSeconds
				#print self.getLabel(randomLabelFileName, indexSeconds)
				#print MFCCMatrix[randomWindowIndex*(BLOCKS_IN_WINDOW)-2*BLOCKS_IN_WINDOW:randomWindowIndex*(BLOCKS_IN_WINDOW)+3*BLOCKS_IN_WINDOW].shape
				randomBatch.inputFeature[batchIndex] = MFCCMatrix[randomWindowIndex*(BLOCKS_IN_WINDOW)-2*BLOCKS_IN_WINDOW:randomWindowIndex*(BLOCKS_IN_WINDOW)+3*BLOCKS_IN_WINDOW].reshape((1,5*BLOCKS_IN_WINDOW*MEL_FEATURE_AMOUNT))
				
				indexSeconds = randomWindowIndex * WINDOW_DURATION + WINDOW_DURATION/2
				#print ''
				#print randomWindowIndex
				#print self.getLabel(randomLabelFileName, indexSeconds)
				#print indexSeconds
				if self.getLabel(randomLabelFileName, indexSeconds) == 0:
					randomBatch.outputFeature[batchIndex, 0] = 1					
				else:
					randomBatch.outputFeature[batchIndex, 1] = 1
					
				batchIndex += 1

			elif(DEBUG):
				print 'can\'t find MFCC file'
		return randomBatch.inputFeature, randomBatch.outputFeature

class batch():
	def __init__(self, batchSize):
		self.batchSize = batchSize
		self.inputFeature = np.zeros((batchSize, 5*MEL_FEATURE_AMOUNT * BLOCKS_IN_WINDOW))
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
