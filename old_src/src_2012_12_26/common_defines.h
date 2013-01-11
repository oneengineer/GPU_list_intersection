/*
 * common_defines.h
 *
 *  Created on: Nov 27, 2012
 *      Author: xchen
 *
 *  define the variables that used in common
 *  most for data and cuda definations
 */


#ifndef COMMON_DEFINES_H_
#define COMMON_DEFINES_H_

#define PI acos(-1)
#define eps 1e-9

#define out(x) (cout<<#x<<":"<<x<<" ")
#define outln(x) (cout<<#x<<":"<<x<<endl)
#define outs(x) (cout<<x)
#define outline (cout<<endl)
#define HERE (printf("GET HERE\n"))
#define THERE (printf("GET THERE\n"))
#define mssleep(time) usleep((time)*(10*1000))

#define FOR_I(begin,end) for (int i=begin;i<end;i++)
#define FOR_J(begin,end) for (int j=begin;j<end;j++)
#define FOR_K(begin,end) for (int k=begin;k<end;k++)
#define FOR_I_J(B1,E1,B2,E2) FOR_I(B1,E1) FOR_J(B2,E2)
#define FOR_I_J_K(B1,E1,B2,E2,B3,E3) FOR_I_J(B1,E1,B2,E2) FOR_K(B3,E3)
#define FOR(begin,end) FOR_I(begin,end)
#define FORN(end) FOR_I(0,end)


const int QUEUE_SIZE= 4;
const int N = 50*1024*1024;

#define H_T_D cudaMemcpyHostToDevice
#define D_T_H cudaMemcpyDeviceToHost
#define D_T_D cudaMemcpyDeviceToDevice
#define CUID (threadIdx.x + blockIdx.x*blockDim.x)
#define CUSIZE (blockDim.x*gridDim.x)

extern __device__ int *list_p0[2];// the origional   address
extern __device__ int *list_p[QUEUE_SIZE][2]; // save the starting position of either list
extern __device__ int calculated_indices_len[QUEUE_SIZE][4];
extern __device__ int swapped[QUEUE_SIZE];   // save swapped stage for each status
extern __device__ int *_result;
extern __device__ int _nm[2];
extern __device__ int partitions[QUEUE_SIZE][128+4][2];

#endif
