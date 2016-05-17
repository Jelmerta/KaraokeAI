import glob
import os
import sys
import subprocess
	
def main():
	args = sys.argv[1:]
	if (len(sys.argv) != 2):
		sys.exit(2)
	folderPath = sys.argv[1]
	
	os.chdir(folderPath)
	for file in glob.glob("*.cdg"):
		call = "python cdg2voicetiming.py \"" + file + "\""
		subprocess.call(call, shell=True)

if __name__ == "__main__":
    main()