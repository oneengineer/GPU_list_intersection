compile_dir="/home/delta1/xchen/programming/workspace/list_intersection/compile_v1"
code_dir="/home/delta1/xchen/programming/workspace/list_intersection/v1"
exe="$compile_dir/list_intersection"
temp_log_name="exe_temp_log"

current_dir=`pwd`


cal_times=10

globalp1="haha"

compile_once(){
	cd $compile_dir
	$current_dir/v1_compile.sh
	cd $current_dir
}

calculate_average(){
	avg=`python -c 'import sys;a=sys.argv[1].split();print "%.7lf" % (reduce(lambda x,y:x+y,map(lambda x:float(x),a))/len(a))' "$globalp1"`
	echo $avg
}

test_on_n(){
num_blocks=32
for i_blocks in `seq 1 5` ; do

	local l1_id=$1
	local l2_id=$2
	#echo $alpha $scala1 $scala2 $size_n
	local len1=`cat data_len_100000 | sed -n $(( $l1_id+ 1))p | cut -d ' ' -f 2`
	local len2=`cat data_len_100000 | sed -n $(( $l2_id+ 1))p | cut -d ' ' -f 2`

	sed -i -r "s/#define DEF_D1 [0-9]+/#define DEF_D1 $num_blocks/g"  $code_dir/common_defines.h
	for num_warps in `seq 1 16` ;do
		num_threads=$(($num_warps * 32 * 4))
		sed -i -r "s/#define DEF_D2 [0-9]+/#define DEF_D2 $num_threads/g" $code_dir/common_defines.h
		compile_once
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
			sleep 0.5
		done
		globalp1=$result
		myalgo_avg=`calculate_average`
		echo $l1_id $l2_id $len1 $len2 $(($num_blocks * 2)) $num_threads $myalgo_avg $result_len
	done
	num_blocks=`expr $num_blocks \* 2`
done

}

work(){
	test_times=20
	max_id=340
	for i in `seq 1 $test_times`;do
		r=`python -c 'import sys;import random;a=int(sys.argv[1]);l1=random.randint(0,a);l2=random.randint(0,a);print l1,l2' $max_id`
		test_on_n $r
	done
}

#compile_once
echo id1 id2 len1 len2 GridDim BlockDim time result_len
#test_on_n 1 4 2 10480000
work

