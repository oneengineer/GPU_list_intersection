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

	__global__ void scan_large(int *array,int *buffer,int blocks,int block_n){
		int id = threadIdx.x;
		int start = block_n-1;
		buffer[id] = array[start + id*block_n];
		scan_one(buffer,blocks);
	}









