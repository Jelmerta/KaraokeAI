#!/usr/bin/python

import glob
import os
import sys
import random
import numpy as np
import warnings

warnings.simplefilter("ignore")
	
MEL_FEATURE_AMOUNT = 13
DEBUG = 0
	
def main():
	if (len(sys.argv) != 3):
		print 'Incorrect amount of parameters'
		sys.exit(2)
	
	# Example call: python batchGenerator.py . 32
	folderPath = sys.argv[1]
	batchSize = int(sys.argv[2])
	
	os.chdir(folderPath)
	labelFiles = glob.glob("*.lbl")

	return getBatch(labelFiles, batchSize)
	
def getBatch(labelFiles, batchSize):
	batch = np.zeros((int(batchSize), MEL_FEATURE_AMOUNT+1))
	trainingExampleIndex = 0
	
	while trainingExampleIndex < batchSize:
		randomLabelFileName = random.choice(labelFiles)
		randomMFCCFileName = randomLabelFileName.replace("lbl", "mp3.mfcc.csv")

		if os.path.isfile(randomLabelFileName):
			MFCCMatrix = np.genfromtxt(randomMFCCFileName, delimiter=',', usecols=np.arange(0, 13), invalid_raise=False)
			
			labelFile = open(randomLabelFileName, "r+")
			labelList = [char for char in labelFile.readline()]
			
			labelAmount = statinfo = os.stat(randomLabelFileName).st_size
			randomLabelIndex = random.randint(0,labelAmount-2)
			
			batch[trainingExampleIndex] = np.append(MFCCMatrix[randomLabelIndex], labelList[randomLabelIndex])
			trainingExampleIndex += 1
	
	if DEBUG:
		print batch
		
	return batch
	
if __name__ == "__main__":
    main()