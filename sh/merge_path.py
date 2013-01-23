import sys
import random


def generate(n,mod,mod2):
	return [random.randint(mod,mod2) for i in range(0,n) ]
	
	
n = 16

l1 = sorted(generate(n,50,70))
l2 = sorted(generate(n,1,100))

result = [ [""]*(n+1) for i in range(0,n+1) ];

for i in range(1,n+1):
	result[i][0] = str(l1[i-1])

for i in range(1,n+1):
	result[0][i] = str(l2[i-1])

def work():
	for i in range(0,n):
		for j in range(0,n):
			if l1[i] == l2[j]:
				result[i+1][j+1] = str(2)
			elif l1[i] > l2[j]:
				result[i+1][j+1] = str(1)
			else :
				result[i+1][j+1] = str(0)

def print_result():
	x = map( lambda x: "\t".join(x) ,result)
	print "\n".join(x)

work()
print_result()


