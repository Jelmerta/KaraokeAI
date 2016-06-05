#!/usr/bin/env python

import sys
import os
import numpy as np

DEBUG = 0
	
# Initialise the player instance
def labeler(timingFileName, outputFilePath, interval):
	self.SingingVector = np.zeros((self.FileSize,), dtype=np.int)

	# Check if the CDG file exists
	if not os.path.isfile(self.FileName):
		ErrorString = "No such file: " + self.FileName
		self.ErrorNotifyCallback (ErrorString)
		raise NoSuchFile
		return
	else:
		labelList = []
		currentTime = 0
		for line in open(self.FileName, "rb"):
			currentTimingList = line.split(" ")
			currentTimeDifference = float(currentTimingList[1]) - float(currentTimingList[0])
			if currentTimingList[2] == 'nosing':
				labelList.append(np.zeros(currentTimeDifference/interval))
			else:
				labelList.append(np.ones(currentTimeDifference/interval))

def getFeatureVector(self, classifiedInstructions, interval):
	packetInterval = int(round(interval/1.0 * CDG_PACKETS_PER_SECOND))
	featureVector = np.zeros((self.FileSize/24)/packetInterval, dtype=np.int)
	i = 0
	for packetIndex in range(self.packetCount - packetInterval/2 - 1, -1, -packetInterval):
		closestValue = self.findClosestValue(classifiedInstructions, packetIndex)
		if (closestValue >= packetIndex - packetInterval / 2) and (closestValue <= packetIndex + packetInterval / 2 - 1):
			featureVector[i] = 1
		i += 1
	return featureVector[::-1]
		
def main():
	args = sys.argv[1:]
	if (len(sys.argv) != 4):
		sys.exit(2)
	l = labeler(sys.argv[1], sys.argv[2], sys.argv[3])
	if len(l) == 0:
		sys.exit(0)
	else:
		sys.exit(1)

if __name__ == "__main__":
    main()
