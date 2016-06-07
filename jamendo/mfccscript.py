#!/usr/bin/env python

import glob
from yaafelib import *
import os
import sys
import numpy as np

import h5py
	
DEBUG = 0
if DEBUG:
	from mutagen.mp3 import MP3

'''Parameters are :
- CepsIgnoreFirstCoeff (default=1): 0 means to keep the first cepstral coeffcient, 1 means to ignore it
- CepsNbCoeffs (default=13): Number of cepstral coefficient to keep.
- FFTWindow (default=Hanning): Weighting window to apply before fft. Hanning|Hamming|None
- MelMaxFreq (default=6854.0): Maximum frequency of the mel filter bank
- MelMinFreq (default=130.0): Minimum frequency of the mel filter bank
- MelNbFilters (default=40): Number of mel filters
- blockSize (default=1024): output frames size
- stepSize (default=512): step between consecutive frames
'''

# Frequency range is hard to define, most singing is between 85-300, but there are exceptions such as a screaming woman that can go up to 3khz. (1280 is approximately the highest for a singing voice) For pretty much all songs, 85-300 is enough. For now, I have chosen 50-1500, but this can easily change.

# At an interval of 1/10s, I think a stepSize of 4410 (a tenth of 44100) should be used.

# The blockSize is 80 for now, but could get higher if necessary

# All the other variables remain default and probably don't need changing

def mfccMaker(folderPath, outputPath, sampleRate, minFreq, maxFreq, blockSize, stepSize):
	fp = FeaturePlan(sample_rate=sampleRate)
	fp.addFeature("mfcc: MFCC MelMinFreq=" + str(minFreq) + " MelMaxFreq=" +str(maxFreq) + " CepsNbCoeffs=30" + " MelNbFilters=30" + " CepsIgnoreFirstCoeff=0" + " blockSize=" + str(blockSize) + " stepSize=" + str(stepSize) + "\"")
	
	df = fp.getDataFlow()
	
	engine = Engine()
	engine.load(df)
	engine.getInputs()
	
	afp = AudioFileProcessor()

	for file in glob.glob(folderPath + "/*.wav"):
		index = file.rfind("/")
		mfccFileName = list(outputPath) + list("/") + list(file[index+1:])
		mfccFileName[-3] = 'h'
		mfccFileName[-2] = '5'
		mfccFileName[-1] = ''
		mfccFileName = "".join(mfccFileName).replace('_', ' - ')
		
		if not os.path.isfile(mfccFileName):
			
			afp.processFile(engine, file)
			
			feats = engine.readAllOutputs()
			print ''
			print mfccFileName
			print feats['mfcc'].shape
			if feats['mfcc'].shape[0] == 0:
				print 'Since no feature has been found, no mfcc file has been created.'
				continue
		
			h5f = h5py.File(mfccFileName, 'w')
			h5f.create_dataset('mfcc', data=feats['mfcc'])
			h5f.close()
			
		else:
			print 'MFCC File already exists. Continuing.'

def getFileSize(fileName):
	statinfo = os.stat(fileName)
	return statinfo.st_size			
			
# Example call:
# python mfccscript.py ../data/Karaoke/mp3 ../features/input/ 44100 50 1500 88.2 88.2 && python mfccscript.py ../data/Karaoke/mp3 ../features/input/ 48000 50 1500 96 96   			
def main():
    args = sys.argv[1:]
    mfccMaker(args[0], args[1], args[2], args[3], args[4], args[5], args[6])

if __name__ == "__main__":
    main()
