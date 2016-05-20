#!/usr/bin/python

import glob
import os
import sys
import random
import numpy as np
	
MEL_FEATURE_AMOUNT = 13
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.05
TEST_SPLIT = 0.25
DEBUG = 0
	
def main():
	if (len(sys.argv) != 4):
		print 'Incorrect amount of parameters'
		sys.exit(2)
	
	# Example call: python batchGenerator.py MFCCFolderPath labelFolderPath 32
	MFCCFolderPath = sys.argv[1]
	labelFolderPath = sys.argv[2]
	batchSize = int(sys.argv[3])
	
	return getBatch(MFCCFolderPath, labelFolderPath, batchSize)
	
def getBatch(MFCCFolderPath, labelFolderPath, batchSize):
	labelFiles = glob.glob(labelFolderPath + "/*.lbl")
	batch = np.zeros((int(batchSize), MEL_FEATURE_AMOUNT+1))
	trainingExampleIndex = 0
	
	while trainingExampleIndex < batchSize:
		randomLabelFileName = random.choice(labelFiles)
		
		index = randomLabelFileName.rfind("/")
		randomMFCCFileName = list(MFCCFolderPath) + list("/") + list(randomLabelFileName[index+1:])
		randomMFCCFileName[-3] = 'n'
		randomMFCCFileName[-2] = 'p'
		randomMFCCFileName[-1] = 'y'
		randomMFCCFileName = "".join(randomMFCCFileName)

		if os.path.isfile(randomLabelFileName):
			if os.path.isfile(randomMFCCFileName):
				MFCCMatrix = np.load(randomMFCCFileName)
			else:
				continue
			
			labelFile = open(randomLabelFileName, "r+")
			labelList = [char for char in labelFile.readline()]
			
			labelAmount = statinfo = os.stat(randomLabelFileName).st_size
			if abs(MFCCMatrix.shape[0] - labelAmount) <=1:
				randomLabelIndex = random.randint(0,labelAmount-2)
			
				batch[trainingExampleIndex] = np.append(MFCCMatrix[randomLabelIndex], labelList[randomLabelIndex])
				trainingExampleIndex += 1
			elif DEBUG:
				print 'File lengths don\'t match'
	
	if DEBUG:
		print batch
		
	return batch
	
if __name__ == "__main__":
    main()
