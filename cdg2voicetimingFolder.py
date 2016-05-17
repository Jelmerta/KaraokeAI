import glob
import os
import sys
import subprocess
	
def main():
	args = sys.argv[1:]
	if (len(sys.argv) != 3):
		sys.exit(2)
	folderPath = sys.argv[1]
	interval = sys.argv[2]
	
	os.chdir(folderPath)
	correctCount = 0
	fileCount = 0
	for file in glob.glob("*.cdg"):
		call = "python ../../../KaraokeAI/cdg2voicetiming.py \"" + file + "\"" + " " + interval
		if(subprocess.call(call, shell=True) == 0):
			correctCount += 1
		fileCount += 1
		
	print "Created "+ str(correctCount) + " label files, out of " + str(fileCount) + " total cdg files."

if __name__ == "__main__":
    main()