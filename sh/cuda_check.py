#!/usr/bin/python

import sys
import subprocess
import re
import random
import time

exepath = "/home/delta1/xchen/programming/workspace/list_intersection/project/list_intersection"
home = "/home/delta1/xchen"
checkexe = home+"/sh/my.cudacheck.sh"

def test_one(srand):
	print "testing..... %d" % (srand)
	time.sleep(0.05);
	exe = "~/sh/my.cudacheck.sh";
	proc = subprocess.Popen([exe,exepath,str(srand)],stdout=subprocess.PIPE,shell=True)
	return_code = proc.wait()
	text = " ".join(proc.stdout)

	print text

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


print checkexe,exepath
p1 = subprocess.call(["/home/delta1/xchen/sh/my.cudacheck.sh",exepath],shell=True)

#work()
