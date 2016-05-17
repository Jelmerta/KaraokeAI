#!/usr/bin/env python

import sys
import os
import struct
import numpy as np
import math
from bisect import bisect_left
from scipy import ndimage

# CDG Command Code
CDG_COMMAND 				= 0x09

# CDG Instruction Codes
CDG_INST_TILE_BLOCK			= 6
CDG_INST_TILE_BLOCK_XOR		= 38

# Bitmask for all CDG fields
CDG_MASK 					= 0x3F

CDG_PACKETS_PER_SECOND 		= 300

DEBUG = 1
		
class Instruction():
	def __init__(self, packetCount, x, y, color):
		self.timing = packetCount
		self.x = x
		self.y = y
		self.color = color
	
	def toString(self):
		print str(self.timing) +  ' ' + str(self.x) + ' ' + str(self.y) + ' ' + str(self.color)
		
class Sequence():
	def __init__(self):
		self.index = 0
		self.instructionList = []
		self.colorList = []
		self.coordDict = {}
		self.classifiedList = []
		
	def colorAmount(self):
		return len(self.colorList)

	def containsVocals(self):
		debugMessage = ''
		containsVocalsBool = True
		if(1 in self.coordDict.values()):
			if(DEBUG):
				debugMessage += 'Debug sequence ' + str(self.index) + ': This sequence has too little repeating coordinates. (hopefully)\n'
			containsVocalsBool = False
			
		if(len(self.colorList) <= 1):
			if(DEBUG):
				debugMessage += 'Debug sequence ' + str(self.index) + ': This sequence has too little colors. (hopefully)\n'
			containsVocalsBool = False
			
		if(any(i > 4 for i in self.coordDict.values())):
			if(DEBUG):
				debugMessage += 'Debug sequence ' + str(self.index) + ': This sequence has too many repeating coords. (hopefully)\n'
			containsVocalsBool = False
							
		if(len(self.colorList) > 4):
			if(DEBUG):
				debugMessage += 'Debug sequence ' + str(self.index) + ': This sequence has too many colors. (hopefully)\n'
			containsVocalsBool = False
		
		if(len(self.colorList) % 2 is 1):
			if(DEBUG):
				debugMessage += 'Debug sequence ' + str(self.index) + ': This sequence has uneven amount of colors.\n'
			containsVocalsBool = False
						
		if(containsVocalsBool):
			if(DEBUG):
				debugMessage += 'Debug sequence ' + str(self.index) + ': This sequence seems too contain vocals\n'
		
		print debugMessage
		return containsVocalsBool
		
	def classifyInstructions(self):
		notSingingColorsList = self.colorList[::2]
		singingColorsList = self.colorList[1::2]
		sequencePacketCount = self.instructionList[-1].timing - self.instructionList[0].timing
		
		for instruction in self.instructionList:
			if(instruction.color in singingColorsList):
				self.classifiedList.append(instruction.timing)
		
		return self.classifiedList
		
class cdgPlayer:
	# Initialise the player instance
	def __init__(self, cdgFileName):
		self.FileName = cdgFileName
		FilePath = os.path.realpath(__file__)
		FileSize = os.path.getsize(FilePath)
		
		# For every 1/300s, the index of the array is either 0 or 1 depending on if someone is singing currently.
		# Initialized as array of zeros.
		self.SingingVector = np.zeros((FileSize,), dtype=np.int)
		
		self.packetCount = 0
		self.amountOfInstructions = 0
		self.sequenceCount = 0
		self.vocalSequenceCount = 0

		# Check if the CDG file exists
		if not os.path.isfile(self.FileName):
			ErrorString = "No such file: " + self.FileName
			self.ErrorNotifyCallback (ErrorString)
			raise NoSuchFile
			return

		self.decode()

	def decode(self):
		# Open the cdg file
		self.cdgFile = open(self.FileName, "rb") 
		if(DEBUG):
			print "Currently classifying: " + self.FileName

		# Main processing loop		
		allClassifiedInstructions = []
		while 1:
			sequence = self.cdgGetNextSequence()
			if(sequence):
				sequence.index = self.sequenceCount
				if(len(sequence.instructionList) != 0):
					if(sequence.containsVocals()):
						self.vocalSequenceCount = self.vocalSequenceCount + 1
						allClassifiedInstructions = np.append(allClassifiedInstructions, sequence.classifyInstructions())
				else:
					print 'empty sequence'
				self.sequenceCount = self.sequenceCount + 1
			else:
				featureVector = self.getFeatureVector(allClassifiedInstructions, 0.1)
				if(DEBUG):
					np.set_printoptions(threshold=np.nan)
					print featureVector
				self.cdgFile.close()
				
				writeFileName = list(self.FileName)
				writeFileName[-3] = "l"
				writeFileName[-2] = "b"
				writeFileName[-1] = "l"
				self.writeToFile("".join(writeFileName), featureVector)
				return
				
	def writeToFile(self, fileName, list):
		with open(fileName, 'w') as f:
			f.write("".join(str(item) for item in list))
				
	def getFileSize(self):
		statinfo = os.stat(self.FileName)
		return statinfo.st_size
			
	def getFeatureVector(self, classifiedInstructions, interval):
		packetInterval = int(round(interval/1.0 * CDG_PACKETS_PER_SECOND))
		featureVector = np.zeros((self.getFileSize()/24)/packetInterval, dtype=np.int)
		for i in range(len(featureVector)):
			packetIndex = i * packetInterval
			closestValue = self.findClosestValue(classifiedInstructions, packetIndex)
			if(np.abs(packetIndex - closestValue) <= packetInterval / 2):
				featureVector[i] = 1
		return featureVector
	
	def findClosestValue(self, myList, myNumber):
		"""
		Assumes myList is sorted. Returns closest value to myNumber.

		If two numbers are equally close, return the smallest number.
		"""
		pos = bisect_left(myList, myNumber)
		if pos == 0:
			return myList[0]
		if pos == len(myList):
			return myList[-1]
		before = myList[pos - 1]
		after = myList[pos]
		if after - myNumber < myNumber - before:
		   return after
		else:
		   return before
	
	def instructionsClose(self, instruction1, instruction2):
		if(math.fabs(instruction1.x - instruction2.x) <=1 and fabs(instruction1.y - instruction2.y) <= 2 and instruction1.color == instruction2.color):
			return True
		else:
			return False
			
	def instructionNextLine(self, instruction1, instruction2):
		if(instruction1.x - instruction2.x > 15 and instruction2.y - instruction1.y >= 0 and instruction2.y - instruction1.y < 3 and instruction1.color == instruction2.color): 
		# y difference is often 1, less like 0 or 2, not sure if ever 3, x difference is often quite high, 15 seems too low but is also safe to not throw away correct ones
			return True
		else:
			return False
		
	# Decode the CDG commands read from the CDG file
	def packetIsInstruction (self, packd):
		if (packd['command'] & CDG_MASK) == CDG_COMMAND:
			return True
		else: 
			return False
		
	def skipNonColoring(self): #returns the first packet with coloring
		while 1:
			currentPackd = self.cdgGetNextPacket()
			if(currentPackd):
				if(self.packetIsInstruction(currentPackd)):
					inst_code = (currentPackd['instruction'] & CDG_MASK)
					if(inst_code == CDG_INST_TILE_BLOCK_XOR or inst_code == CDG_INST_TILE_BLOCK):
						return currentPackd
			else:
				return
		
	def getData(self, packd):
		data_block = packd['data']
		data_block = packd['data']
		on_color = data_block[1] & 0x0F
		x_index = ((data_block[3] & 0x3F)) # The blocks are 6x12 pixels, this information could be used to improve detection
		y_index = ((data_block[2] & 0x1F)) 
		
		return [x_index, y_index, on_color]
		
	# Reads the next sequence of CDG instructions until memory block is found and returns the instructions
	def cdgGetNextSequence(self):
		sequence = Sequence()
		currentPackd = self.skipNonColoring()

		while 1:
			if(currentPackd):
				if(self.packetIsInstruction(currentPackd)):
					inst_code = (currentPackd['instruction'] & CDG_MASK)
					if inst_code == CDG_INST_TILE_BLOCK_XOR:
						xor = 1
					elif inst_code == CDG_INST_TILE_BLOCK:
						xor = 0
					else:
						return sequence				
						
					[x, y, color] = self.getData(currentPackd)
					
					coord = (x, y)
					if(coord in sequence.coordDict):
						sequence.coordDict[coord] += 1
					else:
						sequence.coordDict[coord] = 1
						
					if(not color in sequence.colorList):
						sequence.colorList.append(color)
					
					sequence.instructionList.append(Instruction(self.packetCount, x, y, color))
					self.amountOfInstructions = self.amountOfInstructions + 1
			else:
				if len(sequence.instructionList) == 0:
					return None
				else:
					return sequence
			
			currentPackd = self.cdgGetNextPacket()

	# Read the next CDG command from the file (24 bytes each)
	def cdgGetNextPacket (self):
		packd={}
		packet = self.cdgFile.read(24)
		if (len(packet) == 24):
			self.packetCount = self.packetCount + 1
			packd['command']=struct.unpack('B', packet[0])[0]
			packd['instruction']=struct.unpack('B', packet[1])[0]
			packd['parityQ']=struct.unpack('2B', packet[2:4])[0:2]
			packd['data']=struct.unpack('16B', packet[4:20])[0:16]
			packd['parity']=struct.unpack('4B', packet[20:24])[0:4]
			return packd
		elif (len(packet) > 0):
			print ("Didnt read 24 bytes")
			return None

	def getSeconds(self, packetCount):
		return self.packetCount/CDG_PACKETS_PER_SECOND - (1.0/CDG_PACKETS_PER_SECOND)
		
	def secondsToPacketCount(self, seconds):
		return (seconds+(1.0/CDG_PACKETS_PER_SECOND))*CDG_PACKETS_PER_SECOND
			
def main():
	args = sys.argv[1:]
	if (len(sys.argv) != 2):
		sys.exit(2)
	player = cdgPlayer(sys.argv[1])

if __name__ == "__main__":
    main()