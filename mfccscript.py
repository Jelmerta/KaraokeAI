#!/usr/bin/env python

import glob
from yaafelib import *
import os
import sys
import numpy as np

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

def mfccMaker(folderPath, sampleRate, minFreq, maxFreq, blockSize, stepSize):
	os.chdir(folderPath)

	fp = FeaturePlan(sample_rate=sampleRate)
	fp.addFeature("mfcc: MFCC MelMinFreq=" + str(minFreq) + " MelMaxFreq=" +str(maxFreq) + " blockSize=" + str(blockSize) + " stepSize=" + str(stepSize) + "\"")

	
	df = fp.getDataFlow()
	#df.display()
	
	engine = Engine()
	engine.load(df)
	engine.getInputs()

	afp = AudioFileProcessor()

	for file in glob.glob("*.mp3"):
		afp.processFile(engine, file)
		feats = engine.readAllOutputs()

		mfccFileName = list(file)
		mfccFileName[-3] = 'n'
		mfccFileName[-2] = 'p'
		mfccFileName[-1] = 'y'
		mfccFileName = "".join(mfccFileName)

		np.save("/home/jelmer/features/input/"+mfccFileName, feats['mfcc'])

def main():
    args = sys.argv[1:]
    mfccMaker(args[0], args[1], args[2], args[3], args[4], args[5])

if __name__ == "__main__":
    main()
