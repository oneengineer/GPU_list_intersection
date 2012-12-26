

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

	__device__ void bitonic_merge(bool up, volatile int * data,int id, int n){
		if ( n <1 ) return ;
		int n2 = n >> 1;

		syncthreads();
		if  ((data[id] > data[id + n]) ^ up ){
			int t = data[id];
			data[id] = data[id+n];
			data[id+n] = t;
		}
		//syncthreads();
		// change part of the thread
		if ( id >= n2 )
			id -= n2,data += n;
		bitonic_merge(up, data,id, n2);
	}

	//config should be dim3 ths(16,2)
	__device__ void cal_indx_2(int parts,int part_size,int block_size,int block2_size,int indices_now){
		int id = threadIdx.x;
		int myside = threadIdx.y;
		int opposite_side = !myside;
		int idx = (id+1)*part_size-1;
		int opposite_idx;
		int *myList = list_p[indices_now][ myside ];
		int *oppositeList = list_p[indices_now][ opposite_side ];


		int myValue = myList[ idx ];

		//printf("%d %d parts:%d\n",idx,myside,opposite_side);


		int temp_len[] ={ block_size,block2_size };
		//------ bsearch upper bound part----------
		int left = 0,right = temp_len[opposite_side];
		while ( left < right ){
			int mid = (left + right + 1)/2;
			if ( myValue < oppositeList[mid] )
				right = mid - 1;
			else left = mid;
		}
		//------ END bsearch upper bound part----------
		opposite_idx = left + ( oppositeList[left] <= myValue ) - 1;

		__shared__ volatile int shared[2][64];
		__shared__ volatile int shared2[64];
		shared[myside][ parts - id - 1 ] = idx; //reverse save
		shared[opposite_side][ parts + id ] = opposite_idx;

		syncthreads();
		bitonic_merge(false,shared[myside],id,parts);
		syncthreads();

		partitions[indices_now][ id + 1 ][ myside ] = shared[myside][ id ];
		partitions[indices_now][ id + parts + 1 ][ myside ] = shared[myside][ id + parts ];

		//print out
//		if ( !myside && !id ){
//			FOR_I(0,parts*2){
//				int a = shared[0][ i ];
//				int b = shared[1][ i ];
//				printf("[%d]:%d --- [%d]:%d\n",a,myList[a],b,oppositeList[b]);
//			}
//		}
	}

	__global__ void cal_indx (int block_size,int block_2_size,int indices_now){
		//cal_indx_1(block_size,block_2_size,indices_now);
		int n = blockDim.x;
		cal_indx_2(n,block_size/n,block_size,block_2_size,indices_now);
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

