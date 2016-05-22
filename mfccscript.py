#!/usr/bin/env python

import glob
from yaafelib import *
import os
import sys
import numpy as np

USE_HDF5 = 0
if USE_HDF5:
	import h5py

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
	fp.addFeature("mfcc: MFCC MelMinFreq=" + str(minFreq) + " MelMaxFreq=" +str(maxFreq) + " blockSize=" + str(blockSize) + " stepSize=" + str(stepSize) + "\"")
	
	df = fp.getDataFlow()
	
	engine = Engine()
	engine.load(df)
	engine.getInputs()

	afp = AudioFileProcessor()
	for file in glob.glob(folderPath + "/*.mp3"):
		index = file.rfind("/")
		mfccFileName = list(outputPath) + list("/") + list(file[index+1:])
		if USE_HDF5:
			mfccFileName[-3] = 'h'
			mfccFileName[-2] = '5'
			mfccFileName[-1] = ''
		else:
			mfccFileName[-3] = 'n'
			mfccFileName[-2] = 'p'
			mfccFileName[-1] = 'y'
		mfccFileName = "".join(mfccFileName).replace('_', ' - ')
		
		if not os.path.isfile(mfccFileName):
			afp.processFile(engine, file)
			feats = engine.readAllOutputs() # maybe a try block?
			
			if(USE_HDF5):
				h5f = h5py.File(mfccFileName, 'w')
				h5f.create_dataset('mfcc', data=feats['mfcc'])
				h5f.close()
			else:
				np.save(mfccFileName, feats['mfcc'])
				print feats['mfcc'].shape
				
				labelFile = list('/home/jelmer/features/output/') + list(file[index+1:])
				labelFile[-3] = 'l'
				labelFile[-2] = 'b'
				labelFile[-1] = 'l'
				labelFile = "".join(labelFile)
				if os.path.isfile(labelFile):
					seconds = getFileSize(labelFile) / 10.0
					print seconds
			
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