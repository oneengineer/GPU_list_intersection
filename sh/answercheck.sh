#!/bin/bash

test_times=1000

echo "check $test_times times"

for i in `seq 1 $test_times`;do
	rand=$RANDOM
	echo "result of rand: $rand"
	../project/list_intersection $rand | grep -e "ERROR" -e "Wrong"
	echo "----------------------------"
	sleep 0.5

done


