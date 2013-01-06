#include "scan.cuh"
#include "../common_defines.h"
#include <assert.h>
#include <iostream>
#include <cstdio>

int *scan_buffer;
int SCAN_SMALL_SIZE = 2048;

cudaStream_t *scan_stream;

	__device__ void scan_one(int *array,int n){
		int id = threadIdx.x;
		int id2 = id;
		__shared__ volatile int shared[ 2048*2  ];
		shared[ id ] = 0;
		id += n;
		shared[ id ] = array[ id2 ];
		for (int len = 1; len < n;len <<= 1){
			syncthreads();
			int a = shared[ id ] + shared[ id-len ];
			syncthreads();
			shared[ id ] = a;
		}
		array[ id2 ] = shared[id];
	}

	template<int loops,bool exclusive,bool directly>
	inline __device__ void scan_warp(volatile int * shared,int * src_data,int id,int idx){
		int val0;
		if ( !directly ){
			shared[idx] = 0;
			idx += WARP_SIZE;
			shared[idx] = src_data[ id ];
		}
		val0 = shared[idx];

		if ( 1 <= loops ) shared[idx] += shared[idx - 1];
		if ( 2 <= loops ) shared[idx] += shared[idx - 2];
		if ( 3 <= loops ) shared[idx] += shared[idx - 4];
		if ( 4 <= loops ) shared[idx] += shared[idx - 8];
		if ( 5 <= loops ) shared[idx] += shared[idx - 16];

		if ( exclusive )
			shared[idx] -= val0;
	}

	inline __device__ int scan_offset_share(int x){
		return (x<<1) - (x & (WARP_SIZE-1));
	}

	inline __device__ void scan_a_block2(int * src_data,int offset = 0){
		src_data += offset;
		int id = threadIdx.x;
		int id2 = threadIdx.x + blockDim.x;
		int idx = scan_offset_share(id);
		int idx2 = scan_offset_share(id2);

		volatile __shared__ int shared[SCAN_BLOCK_SIZE*2 + WARP_SIZE*2];
		syncthreads();
		scan_warp< LOG_WARP_SIZE,false,false>(shared,src_data,id,idx);//basic level
		syncthreads();
		scan_warp< LOG_WARP_SIZE,false,false>(shared,src_data,id2,idx2);//basic level part 2
		syncthreads();

		//move data
		if ( id < WARP_SIZE ){
			int id2 = scan_offset_share((id << LOG_WARP_SIZE) + WARP_SIZE-1) + WARP_SIZE;
			shared[ SCAN_BLOCK_SIZE*2 + id ] = 0;
			shared[ SCAN_BLOCK_SIZE*2 + WARP_SIZE+id ] = shared[ id2 ];
			scan_warp<LOG_SCAN_BLOCK_SIZE - LOG_WARP_SIZE ,true,true>(shared + SCAN_BLOCK_SIZE*2 + WARP_SIZE ,0,0,id);//second level, use calculated 32,in one block
		}
		syncthreads();
		//uniform update
		int a = shared[ (id >> LOG_WARP_SIZE) + SCAN_BLOCK_SIZE*2 + WARP_SIZE ];
		//printf("id: %d   += [%d] %d\n",id,(id >> LOG_WARP_SIZE) + SCAN_BLOCK_SIZE*2 + WARP_SIZE,a);
		shared[ idx+WARP_SIZE ] += a;
		a =  shared[ (id2 >> LOG_WARP_SIZE) + SCAN_BLOCK_SIZE*2 + WARP_SIZE ];
		shared[ idx2 + WARP_SIZE ] += a;
		//write back
		src_data[ id ] = shared [ idx + WARP_SIZE ];
		src_data[ id2 ] = shared [ idx2 + WARP_SIZE ];
	}

	__global__ void scan_small(int *array,int small_size){
		array += small_size * blockIdx.x;
		scan_one(array,small_size);
	}

	__global__ void scan_large(int *array,int *buffer,int blocks,int block_n){
		int id = threadIdx.x;
		int start = block_n-1;
		buffer[id] = array[start + id*block_n];
		scan_one(buffer,blocks);
	}

	__global__ void uniform_update(int *array,int *buffer){
		int id = threadIdx.x + blockIdx.x * blockDim.x;
		int addon = buffer[ blockIdx.x ];
		array[id] += addon;
	}

	__global__ void scan_x(int *array,int small_size){
		scan_a_block2(array,blockIdx.x*small_size);
	}


	void init_scan(cudaStream_t * in_scan_stream = 0,int scan_small_size = 512){
		SCAN_SMALL_SIZE = scan_small_size;
		scan_stream = in_scan_stream;
		cudaMalloc((void **)&scan_buffer,sizeof(int)*1024*8);
		int temp[] = {0,0,0,0};
		cudaMemcpy(scan_buffer,temp,sizeof(temp),H_T_D);
	}

	void scan(int *devL,int n){
		int small_size = SCAN_SMALL_SIZE;
		int blocks = n / small_size;
		scan_small<<<blocks,small_size,0,*scan_stream>>>(devL,small_size);
		scan_large<<<1,blocks,0,*scan_stream>>>(devL,scan_buffer+4,blocks,small_size);
		uniform_update<<<blocks,small_size,0,*scan_stream>>>(devL,scan_buffer+3);
	}

	void scan2(int *devL,int n){
		int small_size = 1024;
		int blocks = n / small_size;
		scan_x<<<blocks,small_size/2,0,*scan_stream>>>(devL,small_size);
		scan_large<<<1,blocks,0,*scan_stream>>>(devL,scan_buffer+4,blocks,small_size);
		uniform_update<<<blocks,small_size,0,*scan_stream>>>(devL,scan_buffer+3);
	}

	void ScanSequence::run_scan(int step){
		assert(step < steps);
		int offset = step * step_size;
		scan_x<<< D1,D2,0,*scan_stream >>>( devL + offset , small_size);
	}

	__global__ void cu_printx(int *list,int size){
		printf("CUDA SHOWx:\n");
		if ( 0 == threadIdx.x && 0 == blockIdx.x ){
			FOR_I(0,size)
				printf("[%d]:%d\t",i,list[i]);
			printf("\n");
		}
	}

	void ScanSequence::run_large(){
		int blocks = n / small_size;
		scan_large<<<1,blocks,0,*scan_stream>>>(devL,scan_buffer+4,blocks,small_size);
	}






