#!/usr/bin/python

import songBatchGenerator
import os
import sys
import numpy as np
import glob
import random
	
MEL_FEATURE_AMOUNT = 30
BLOCKS_IN_WINDOW = 8
WINDOW_DURATION = 0.2

DEBUG = 1

import h5py

class staticBatchGenerator():
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
	
	def getSet(self):
		firstTime = True
		for MFCCFileName in self.MFCCFileNames:
			sbg = songBatchGenerator.songBatchGenerator(MFCCFileName, self.MFCCToLabelFileName(MFCCFileName))
			songBatch = sbg.getBatch()
			if firstTime:
				input = songBatch[0]
				output = songBatch[1]
				firstTime = False
			else:
				input = np.append(input, songBatch[0], axis=0)
				output = np.append(output, songBatch[1], axis=0)

		p = np.random.permutation(len(input))
		return input[p], output[p]

def main():
	if (len(sys.argv) != 3):
		print 'Incorrect amount of parameters'
		sys.exit(2)

	MFCCFolderPath = sys.argv[1]
	labelFolderPath = sys.argv[2]

	return staticBatchGenerator(MFCCFolderPath, labelFolderPath)

if __name__ == "__main__":
	main()
