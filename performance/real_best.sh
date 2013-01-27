#!/bin/bash
compile_dir="/home/delta1/xchen/programming/workspace/list_intersection/compile_v1"
code_dir="/home/delta1/xchen/programming/workspace/list_intersection/v1"
exe="$compile_dir/list_intersection"
temp_log_name="exe_temp_log"

current_dir=`pwd`


cal_times=10

globalp1="haha"
cases=50

compile_once(){
	cd $compile_dir
	$current_dir/v1_compile.sh
	cd $current_dir
}

calculate_average(){
	avg=`python -c 'import sys;a=sys.argv[1].split();print "%.7lf" % (reduce(lambda x,y:x+y,map(lambda x:float(x),a))/len(a))' "$globalp1"`
	echo $avg
}

seed1=0
seed2=1

generate_rand_id_list(){
	local my_seed=$1
	max_id=200
	r=`python -c 'import sys;import random;random.seed(int(sys.argv[3]));a=int(sys.argv[1]);b=int(sys.argv[2]);l1=[ str(random.randint(0,a)) for i in range(0,b) ];print " ".join(l1);' $max_id $cases $my_seed`
	echo $r
}

set_config(){
num_blocks=32
for i_blocks in `seq 1 5` ; do

	sed -i -r "s/#define DEF_D1 [0-9]+/#define DEF_D1 $num_blocks/g"  $code_dir/common_defines.h
	for num_warps in `seq 1 16` ;do
		num_threads=$(($num_warps * 32 * 4))
		sed -i -r "s/#define DEF_D2 [0-9]+/#define DEF_D2 $num_threads/g" $code_dir/common_defines.h
		compile_once
		work 
	done
	num_blocks=`expr $num_blocks \* 2`
done

}

work(){
	for i in `seq 0 $(($cases - 1))`;do
		l1_id=${l1_list[ $i ]}
		l2_id=${l2_list[ $i ]}
		local len1=`cat data_len_100000 | sed -n $(( $l1_id+ 1))p | cut -d ' ' -f 2`
		local len2=`cat data_len_100000 | sed -n $(( $l2_id+ 1))p | cut -d ' ' -f 2`
		result=""
		for times in `seq 1 $cal_times` ; do
			$exe $l1_id $l2_id > $temp_log_name
			temp=`cat $temp_log_name| grep "MY Algo:[0-9\.]" | cut -d ":" -f 2`
			result_len=`cat $temp_log_name| grep "Lresult:[0-9]" | cut -d ":" -f 2 `
			temp_error=`cat $temp_log_name| grep "ERROR:"`
			if [ ${#temp_error} -ne 0 ];then
				echo "ERROR"
				exit
			fi
			result="$result $temp"
			sleep 0.1
		done
		globalp1=$result
		myalgo_avg=`calculate_average`
		echo $l1_id $l2_id $len1 $len2 $(($num_blocks * 2)) $num_threads $myalgo_avg $result_len
	done
}

#compile_once
echo generate seed1 $seed1 seed2 $seed2
echo id1 id2 len1 len2 GridDim BlockDim time result_len
#test_on_n 1 4 2 10480000
l1_list=(`generate_rand_id_list $seed1`)
l2_list=(`generate_rand_id_list $seed2`)

set_config

