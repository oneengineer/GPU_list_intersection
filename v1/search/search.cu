#include "../common_defines.h"
#include "search.cuh"
#include <cstdio>


	//extern __device__ void scan_a_block_neat(int * src_data,volatile int shared[][WARP_SIZE*2]);

	template<int loops,bool exclusive>
	inline __device__ void scan_warp_neet(volatile int * shared,int value,int id){
		shared[id] = 0;
		id += WARP_SIZE;
		shared[id] = value;

		if ( 1 <= loops ) shared[id] += shared[id - 1];
		if ( 2 <= loops ) shared[id] += shared[id - 2];
		if ( 3 <= loops ) shared[id] += shared[id - 4];
		if ( 4 <= loops ) shared[id] += shared[id - 8];
		if ( 5 <= loops ) shared[id] += shared[id - 16];

		if ( exclusive )
			shared[id] -= value;
	}

	template< int id_range >
	inline __device__ void scan_a_block_neat(volatile int * src_data,volatile int  shared[][WARP_SIZE*2]){
		int id = threadIdx.x;
		int local_id = id&(WARP_SIZE-1);
		int warp_num = id >> LOG_WARP_SIZE ;
		volatile __shared__ int shared_level2[WARP_SIZE*2];

		scan_warp_neet< LOG_WARP_SIZE,false>(shared[warp_num],src_data[id],local_id );//basic level
		syncthreads();
		if ( id < WARP_SIZE){
			int myvalue;
			if ( id < id_range  )
				myvalue = shared[id][WARP_SIZE+WARP_SIZE-1];
			scan_warp_neet<LOG_WARP_SIZE ,true>(shared_level2,myvalue,id);//second level, use calculated 32,in one block
		}
		syncthreads();
		//uniform update
		int a = shared_level2[ warp_num+WARP_SIZE ];
		src_data[id] = shared[warp_num][ local_id + WARP_SIZE ] + a;
	}

	__global__ void scan_buffer_large(int save_indx){

#if DEF_D1*2/WARP_SIZE < 1
		__shared__ volatile int  shared[ 2 ][WARP_SIZE*2];
#else
		__shared__ volatile int  shared[ DEF_D1*2/WARP_SIZE ][WARP_SIZE*2];
#endif

		scan_a_block_neat<2*DEF_D1/WARP_SIZE>(_scan_buffers[save_indx],shared);
		//printf("scaned: <%d> %d\n",threadIdx.x,_scan_buffers[save_indx][threadIdx.x]);
	}



	/**
	 *
	 * To write the program use minimum command
	 * Three things decide the next position:
	 * 1. position out of boundary?
	 * 2. get value, direction? change or not?
	 * 3. length? how l
	 * L: position = what
	 *
	 *
	 * value: the value to search
	 * pos2: starting place, extrapolated position, it need to be return!
	 * return:result 0/1
	 *
	 */

	inline __device__ int gallop_search (int value,int & pos2,volatile int *list,int left,int right){

		int pos = pos2;
		int result = 0;
		int len;
		int value2;

		char last_direction = 3;
		char direction = 0;
		char start_decrease = 0;

		len = 1;
		while ( len > 0){

			if (!( left <= pos2 && pos2 <= right )){
				start_decrease = 1;
			}
			else{
				value2 = list[pos2] - value;
				if ( 0 == value2 ){
					result = 1;break;
				}
				else direction = value2 < 0;
				start_decrease |= (direction + last_direction) ==1;// 0 + 1 or 1 + 0
				pos = pos2;
				last_direction = direction;
			}
			if ( start_decrease )
				len >>=1;
			else len <<=1;

			if ( direction )
				pos2 = pos + len;
			else pos2 = pos - len;
		}
		return result;
	}

	inline __device__ int binary_search(int value,int & pos,volatile int *list,int left,int right){
		int result = 0;
		while ( left <= right ){
			if ( value == list[ pos ] ){
				//printf("block:%d thread: %d INTERSECTION %d\n",blockIdx.x,threadIdx.x,value);
				result = 1;
				break;
			}
			else if ( list[pos] < value )
				left = pos + 1;
			else right = pos - 1;
			pos = (left + right)/2;
		}
		return result;
	}

	/*
	 * because the search_2 is done in one SM in GPU, no divergence occur
	 * begin and end are assigned from blockIdx, one block only have on value => no divergence
	 *
	 */
	inline __device__ int search_one_value(int value,int & pos,volatile int *opposite_addr,const int &left,const int &right){

		//printf("[<%d,%d>] has value  %d search in %d %d\n",blockIdx.x,threadIdx.x,value,opposite_addr[left],opposite_addr[right]);
		int result = binary_search(value,pos,opposite_addr,left,right);

//
//		if (result){
//			int write_pos = atomicAdd(&gpu_result_size,1);
//			printf("[<%d,%d>] found %d\n",blockIdx.x,threadIdx.x,value);
//			_result[write_pos] = value;
//		}
		return result;
	}

	/*
	 * search 4 integer stored in uint4, a,b,c,d
	 * use divide-and-conquer way a,c,b,d order search
	 *
	 */
	inline __device__ void search_uint4(const int & id,const uint4 &myvalue,volatile int *opposite_list,
			volatile int *shared_range,const struct partition_info &info,
			int &mask,int &uint4_sum){
		int pos_A,pos_X,pos_C,pos_D;
		pos_A = (info.left + info.right)/2;
		//pos_A = id<<2;//check search times
		if (search_one_value(myvalue.x,pos_A,opposite_list,info.left,info.right)){
			mask |=1,uint4_sum ++;
		}
		shared_range[id] = pos_A;
		syncthreads();
		int id_addone = id + 1;
		if ( id_addone != info.len )
			pos_D = shared_range[id_addone];
		else pos_D = info.right;

		pos_C = (pos_A + pos_D) /2 ;
		if (search_one_value(myvalue.z,pos_C,opposite_list,pos_A,pos_D)){
			mask |=4,uint4_sum ++;
		}
		pos_X = (pos_A + pos_C) /2 ;
		if (search_one_value(myvalue.y,pos_X,opposite_list,pos_A,pos_C)){
			mask |=2,uint4_sum ++;
		}
		pos_X = (pos_C + pos_D) /2 ;
		if (search_one_value(myvalue.w,pos_X,opposite_list,pos_C,pos_D)){
			mask |=8,uint4_sum ++;
		}
	}

#if 1
	/*
	 * Simple method, not use shared
	 * */
	inline __device__ void search_uint4_2(const int & id,const uint4 &myvalue,volatile int *opposite_list,
			volatile int *shared_range,const struct partition_info &info,
			int &mask,int &uint4_sum){
		int pos_A,pos_X,pos_D;
		//pos_A = id;
		pos_A = (info.left + info.right)/2;
		mask = 0,uint4_sum=0;
		if (search_one_value(myvalue.x,pos_A,opposite_list,info.left,info.right)){
			mask |=1,uint4_sum ++;
		}
		pos_D =  ( pos_A+ info.right)/2;
		if (search_one_value(myvalue.w,pos_D,opposite_list,pos_A,info.right)){
			mask |=8,uint4_sum ++;
		}
		pos_X = (pos_A + pos_D) /2 ;
		if (search_one_value(myvalue.y,pos_X,opposite_list,pos_A,pos_D)){
			mask |=2,uint4_sum ++;
		}
		if (search_one_value(myvalue.z,pos_X,opposite_list,pos_X,pos_D)){
			mask |=4,uint4_sum ++;
		}
	}
#endif



	inline __device__ void scan_and_save_buffer(volatile int *scan_array,volatile int *scan_shared,int * result_buffer,
			const int & id,uint4 &myvalue,int &mask,int &uint4_sum){
		scan_array[id] = uint4_sum;
		syncthreads();
		scan_a_block_neat<DEF_D2/4/WARP_SIZE>((int *)scan_array,(int (*)[WARP_SIZE*2])scan_shared);
		syncthreads();

		int position = scan_array[id] - uint4_sum; // exclusive scan
		//int position_debug = position;

		if ( mask & 1 )
			result_buffer[position++] = myvalue.x;
		if ( mask & 2 )
			result_buffer[position++] = myvalue.y;
		if ( mask & 4 )
			result_buffer[position++] = myvalue.z;
		if ( mask & 8 )
			result_buffer[position++] = myvalue.w;

		//debug bad save method
//		FOR_I(position_debug,position){
//			//int write_pos = atomicAdd(&gpu_result_size,1);
//			if ( result_buffer[i] == 3836)
//			printf("%d is saved at %d by <%d,%d>\n",result_buffer[i],i,blockIdx.x,threadIdx.x);
//		}



	}

#if DEPLETED

	inline __device__ void brute_force(uint4 & myvalue,volatile int * list,const struct partition_info &info){
		FOR_I(info.left,info.right+1){
			if ( myvalue.x == list[i] ){
				int write_pos = atomicAdd(&gpu_result_size,1);
				_result[write_pos] = myvalue.x;
			}
			if ( myvalue.y == list[i] ){
				int write_pos = atomicAdd(&gpu_result_size,1);
				_result[write_pos] = myvalue.y;
			}
			if ( myvalue.z == list[i] ){
				int write_pos = atomicAdd(&gpu_result_size,1);
				_result[write_pos] = myvalue.z;
			}
			if ( myvalue.w == list[i] ){
				int write_pos = atomicAdd(&gpu_result_size,1);
				_result[write_pos] = myvalue.w;
			}
		}
	}
#endif

	inline __device__ void search_in_block(int * V,int search_now,const struct partition_info &info){
		int id = threadIdx.x;
		// begin and end is closed interval [begin , end]

		volatile __shared__ int opposite_list[DEF_D2];//scan use 2/4 times space
		volatile __shared__ int shared_range[DEF_D2/4];// only use once, the right most thread does not use it
		int mask,uint4_sum;
		uint4 myvalue ;
		bool flag1 = false;
		/*
		 * shared_range __shared__ use as src_data for scan
		 * opposite_list __shared__ use as
		 *  */

		// the thread which should copy opposited list elements
		if (id < info.len_opposite){

			myvalue = ((uint4 *)info.opposite_addr)[id];
			((uint4 *)opposite_list)[id] = myvalue;
			if ( id < info.len ){
				myvalue = ((uint4 *)info.addr)[id];
				flag1 = true;

			}
		}
		if ( id < info.warp_len ){
			mask= 0;uint4_sum=0;
			syncthreads();
			if (flag1){
				search_uint4(id,myvalue,opposite_list,shared_range,info,mask,uint4_sum);
			}
			syncthreads();
			scan_and_save_buffer(shared_range,opposite_list,V+DEF_D2*blockIdx.x,id,myvalue,mask,uint4_sum);
			if ( info.warp_len-1 ==id ){
				//printf("<%d %d>'s last  is %d\n",blockIdx.x,threadIdx.x,shared_range[id]);
				_scan_buffers[search_now][blockIdx.x] = shared_range[id];//scan_array
			}
		}
	}

	__global__ void algo2_search(int * V,int search_now,int offset){
		if (partitions_info[search_now][blockIdx.x].len>0){
			search_in_block(V,search_now,partitions_info[search_now][ blockIdx.x ]);
		}
	}



