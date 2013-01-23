#include "../common_defines.h"
#include "save_result.h"
#include "scan.cuh"
#include <cstdio>

extern int *scan_buffer;
extern cudaStream_t *scan_stream;

	inline __device__ void one_block_compact(const int &compact_blocks,int *buffer,int *result,const int & save_idx){
		int id = threadIdx.x;
		int bsize = blockDim.x;
		int end;
		int *save_addr = _result_addr[save_idx];

		FOR_I( 0 ,compact_blocks){
			int j = buffer[i-1];
			end = buffer[i];
			int id2 = id;
			j += id;

			while ( j < end ){
				save_addr[ j ] = result[id2];
				j += bsize;
				id2 += bsize;
			}
			result += DEF_D2;
		}
		if ( 0 == id && blockIdx.x == gridDim.x -1 ){
			int next_idx = (save_idx + 1)%QUEUE_SIZE;
			//printf("End move: %d\n",end);//debug
			_result_addr[next_idx] = end + save_addr;
			_result = _result_addr[next_idx];//atmoic
		}
	}

	__global__ void compact(int compact_blocks,int *result,int save_idx){
		int offset1 = blockIdx.x * compact_blocks;
		int offset2 = offset1 * DEF_D2;
		one_block_compact(compact_blocks,_scan_buffers[save_idx] + offset1,result+offset2,save_idx);
	}




