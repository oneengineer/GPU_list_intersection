

#include "cal_indices.cuh"
#include "../common_defines.h"
#include <cstdio>

	__device__ int next_queue_pos(int value){
			return (value+1) % QUEUE_SIZE;
	}

	// config should be <1,4>
	__device__ void cal_indx_1(int block_size,int block_2_size,int indices_now){
		int id = threadIdx.x;
		int group_id = id >> 1;
		int sub_id = id & 1;
		int opposite_sub_id = sub_id ^ 1; //!sub_id

		int indices_next = (indices_now + 1) % QUEUE_SIZE;
		int temp_blocks[] = {block_size, block_2_size};

		int *myList = list_p[indices_now][ group_id ^ sub_id ];
		int *oppositeList = list_p[indices_now][ group_id ^ opposite_sub_id ];
		int myLen = temp_blocks[sub_id];
		int oppositeLen = temp_blocks[ opposite_sub_id ];
		int value = myList[ myLen - 1 ];
		int left = 0, right = oppositeLen -1;

		while ( left < right ){
			int mid = (left + right + 1)/2;
			if ( value < oppositeList[mid] )
				right = mid - 1;
			else left = mid;
		}

		//printf("id:%d value %d get to %d\n",id,value,oppositeList[left]);//debug

		int next_opposite_offset = left + ( oppositeList[left] <= value );

		int *save_indices0 = calculated_indices_len[indices_now];
		int *save_indices = save_indices0 + (group_id << 1);
		int *opposite_indices = save_indices0 + ( (group_id^1) << 1);
		save_indices[sub_id] = next_opposite_offset ;

		int mysum = save_indices[ sub_id ] + save_indices[ opposite_sub_id ];
		int oppositeSum = opposite_indices[ sub_id ] + opposite_indices[ opposite_sub_id ];

		//decide opposite list offset and length
		if ( mysum + group_id > oppositeSum){

			//printf("cal TURN: id %d len: %d  \n",id,next_opposite_offset);//debug
			save_indices0[ opposite_sub_id ] = next_opposite_offset - 1;
			list_p[indices_now][sub_id] = myList;

			list_p[indices_next][ opposite_sub_id ] = oppositeList + next_opposite_offset;//set opposite pointer
			swapped[ indices_next ] = swapped[ indices_now ] ^ group_id;
		}
	}

	//config should be dim3 ths(16,2)
	__device__ void cal_indx_2(){

	}

	__global__ void cal_indx (int block_size,int block_2_size,int indices_now){
		cal_indx_1(block_size,block_2_size,indices_now);
	}

	//special case for cal_indices
	__global__ void move_indices(int len1,int len2,int block_size,int indices_now){
		int indices_next = (indices_now + 1) % QUEUE_SIZE;
		int swapflag = 0;
		//make sure that len1 is always the shorter one
		if ( len1 > len2 ){
			int *temp = list_p[indices_now][0];
			list_p[indices_now][0] = list_p[indices_now][1];
			list_p[indices_now][1] = temp;
			int temp2 = len1;
			len1 = len2;
			len2 = temp2;
			swapflag = 1;
		}
		int b1 = min(len1,block_size);
		list_p[ indices_next ][ 0 ] = list_p[ indices_now ][ 0 ] + b1;
		list_p[ indices_next ][ 1 ] = list_p[ indices_now ][ 1 ] ;//keep the same
		swapped[ indices_next ] = swapped[ indices_now ] ^ swapflag;
		calculated_indices_len[ indices_now ][ 0 ] = b1-1;// or b1
		calculated_indices_len[ indices_now ][ 1 ] = len2-1;// or b1
	}

