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

//#define out(x) (cout<<#x<<":"<<x<<" ")
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

#define DEPLETED 0


const int QUEUE_SIZE= 4;
const int N = 50*1024*1024;

#define H_T_D cudaMemcpyHostToDevice
#define D_T_H cudaMemcpyDeviceToHost
#define D_T_D cudaMemcpyDeviceToDevice
#define CUID (threadIdx.x + blockIdx.x*blockDim.x)
#define CUSIZE (blockDim.x*gridDim.x)
#define IS_LAST ((threadIdx.x + blockIdx.x*blockDim.x) == CUSIZE -1)
#define IS_FIRST (!threadIdx.x && !blockIdx.x)

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5
#define SCAN_BLOCK_SIZE 1024
#define LOG_SCAN_BLOCK_SIZE 10

#define DEF_D1 512
#define DEF_D2 2048

struct partition_info{
	int *addr,*opposite_addr;
	int len,len_opposite,warp_len;//length of elements in addr
	int left,right;// range of opposite_addr
	bool B2A;
};


struct list_info{
	int len;
	int *start_addr;
};

struct dataset_info{
	int len;
};

extern __device__ list_info *list_infos;
extern __device__ list_info L1,L2;

extern __device__ int *list_p0[2];// the origional   address
extern __device__ int *list_p[QUEUE_SIZE][2]; // save the starting position of either list
extern __device__ int calculated_indices_len[QUEUE_SIZE][2];
extern __device__ int swapped[QUEUE_SIZE];   // save swapped stage for each status
extern __device__ int *_result;
extern __device__ int gpu_result_size;
extern __device__ int _nm[2];
extern __device__ partition_info partitions_info[QUEUE_SIZE][DEF_D1*2 + 4];
extern __device__ int *_result_addr[QUEUE_SIZE];

extern __device__ int o_scan_buffers[QUEUE_SIZE][4+DEF_D1*2+32*2];
extern __device__ int *_scan_buffers[QUEUE_SIZE];

extern int *data_pool;
extern int *gpu_data_pool;
extern struct list_info *gpuData;
extern struct list_info *cpuData;

struct debug_structure{
	int num_loop;
	int wrong_1;
	int wrong_2;
};

extern __device__ debug_structure debug1;

typedef long long LL;

#endif




