#!/usr/bin/python

import sys
import subprocess
import re
import random
import time

exepath = "../project/list_intersection"

def test_one(srand):
	print "testing..... %d" % (srand)
	time.sleep(0.05);
	proc = subprocess.Popen(["cuda-memcheck",exepath,str(srand)],stdout=subprocess.PIPE)
	return_code = proc.wait()
	text = " ".join(proc.stdout)

	#print text

	pattern ='ERROR:'

	result = re.findall(pattern,text)
	if result:
		print text
		return False
	
	return True


def work():
	for i in range(100000):
		s = random.randint(10,1000000)
		#s = 601431
		if not test_one(s):
			break


work()
