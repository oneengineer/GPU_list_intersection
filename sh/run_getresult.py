#!/usr/bin/python

import sys
import subprocess
import re

exepath = "../project/list_intersection"

proc = subprocess.Popen([exepath],stdout = subprocess.PIPE)


text = "\n".join(proc.stdout)

return_code = proc.wait()
pattern ='(?P<value>\d+) was found at'

result = re.findall(pattern,text,re.IGNORECASE)
result = map(lambda x:int(x), result )
result = sorted(result)
print result


m = re.search('cpuresultsize',text,re.IGNORECASE)
cpu_result = text[m.end():]
cpu_pattern = '\[\d+\]:\s*(?P<value>\d+)'
cpu_result = re.findall(cpu_pattern,cpu_result,re.IGNORECASE)
cpu_result = map(lambda x:int(x), cpu_result )
cpu_result = sorted(cpu_result)
print cpu_result
print "Your result length: %d  correct length:%d" % (len(result),len(cpu_result))

a0 = "".join(map( lambda x:str(x),cpu_result));
a1 = "".join(map( lambda x:str(x),result));
if a0 != a1:
	print "WRONG!"
	for i in range(len(result)):
		if cpu_result[i] != result[i]:
			print "wrong at [%d] %d  cpu:%d" % (i,cpu_result[i],result[i])
