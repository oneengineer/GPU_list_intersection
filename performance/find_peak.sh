compile_dir="/home/delta1/xchen/programming/workspace/list_intersection/project"
code_dir="/home/delta1/xchen/programming/workspace/list_intersection/src"
exe="$compile_dir/list_intersection"

current_dir=`pwd`

num_blocks=32

cal_times=10


test_on_n(){
for i_blocks in `seq 1 5` ; do

	local alpha=$1
	local scala1=$2
	local scala2=$3
	local size_n=$4
	#echo $alpha $scala1 $scala2 $size_n

	sed -i -r "s/#define DEF_D1 [0-9]+/#define DEF_D1 $num_blocks/g"  $code_dir/common_defines.h
	for num_warps in `seq 1 16` ;do
		num_threads=$(($num_warps * 32 * 4))
		sed -i -r "s/#define DEF_D2 [0-9]+/#define DEF_D2 $num_threads/g" $code_dir/common_defines.h
		cd $compile_dir
		$current_dir/compile.sh
		cd $current_dir
		result=""
		for times in `seq 1 $cal_times` ; do
			temp=`$exe $alpha $scala1 $scala2 $size_n| grep "MY Algo:[0-9\.]" | cut -d ":" -f 2`
			result="$result $temp"
			sleep 0.5
		done
		avg=`python -c 'import sys;a=sys.argv[1].split();print "%.7lf" % (reduce(lambda x,y:x+y,map(lambda x:float(x),a))/len(a))' "$result"`
		echo $alpha $scala1 $scala2 $size_n $(($num_blocks * 2)) $num_threads $avg
	done
	num_blocks=`expr $num_blocks \* 2`
done

}

work(){
	alpha=`seq 1 10 && seq 20 10 40`
	for i in $alpha ; do
		scala1=`python -c 'import sys;a=int(sys.argv[1]);print "%d %d %.3lf %d" % (2*a,a,float(a)/2.0,1)' $i`
		for j in $scala1 ; do
			scala2="1 2"
			for k in $scala2 ; do
				ns=`python -c 'import sys;a=int(sys.argv[1]);n2=1024**2;
if a > 10:
	n2 /=10
n3 = 4*1024**2/a
print n2,n3' $i`
				for nx in $ns ; do
					test_on_n $i $j $k $nx
					#echo $i $j $k $nx
				done
			done
		done
	done
}

echo alpha scala1 scala2 N GridDim BlockDim time
#test_on_n 1 4 2 10480000
work

