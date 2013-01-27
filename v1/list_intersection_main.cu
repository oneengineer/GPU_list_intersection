/*New feature:
 pre calculate indices, smart adaptive, fully pipeline, not fully hidden
 cudpp have streams, but number of device block/threads can not be controlled
 cudpp prefix sum for my algo
 pipeline ( cudpp pipelined) is employed
 pipeline ( myscan pipelined) is employed
 */

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include <unistd.h>
#include <sys/time.h>

//------ for CUDA ------
#include <cuda.h>
#include <cuda_runtime.h>

#include "common_defines.h"
#include "common_functions.h"
#include "others/bsearch.cuh"
#include "others/memory_data.cuh"
#include "indices/cal_indices.cuh"
#include "search/search.cuh"
#include "save/scan.cuh"
#include "save/save_result.h"
#include "others/read_data.cuh"

#include "others/generate_data.h"

//#include "cudpp.h"

using namespace std;

template<typename T>
void debug_a(T * data, int begin, int end) {
	for (int i = begin; i < end; i++)
		cout << "[" << i << "]: " << data[i] << "\t";
	cout << endl;
}
template<typename T>
void debug_a(T * data, int end) {
	debug_a(data, 0, end);
}
template<typename T>
void debug_a2(T * data, int end1, int end2) {
	for (int i = 0; i < end1; i++) {
		cout << "row " << i << endl;
		for (int j = 0; j < end2; j++)
			cout << "[" << i << "," << j << "] " << data[i][j] << "\t";
		cout << endl;
	}
}
double get_sec(const struct timeval & tval) {
	return ((double) (tval.tv_sec * 1000 * 1000 + tval.tv_usec)) / 1000000.0;
}

template<typename T>
T checkmin(T & data, T value) {
	data = min(data, value);
	return data;
}
struct Watch {
	timeval begin, end;
	void start() {
		gettimeofday(&begin, NULL);
	}
	double time() {
		return get_sec(end) - get_sec(begin);
	}
	double stop() {
		gettimeofday(&end, NULL);
		return time();
	}
};

inline double rand_double(double range) {
	return ((double) rand() / (double) RAND_MAX) * range;
}
//----------- cuda template ------------

struct CudaWatch {
	cudaEvent_t start_event, stop_event;
	void start() {
		cudaEventCreate(&start_event);
		cudaEventCreate(&stop_event);
		cudaEventRecord(start_event, 0);
	}
	float stop() {
		float time_cost = -1.0;
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&time_cost, start_event, stop_event);
		return time_cost;
	}
};
void cu_checkError() {
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
}

__global__ void cu_print(int *list, int size) {
	printf("CUDA SHOW:\n");
	if (0 == threadIdx.x && 0 == blockIdx.x) {
		FOR_I(0,size)
			printf("[%d]:%d\t", i, list[i]);
		printf("\n");
	}
}

void cu_host_print(int *list, int size) {
	cudaDeviceSynchronize();
	cu_print<<<1, 1>>>(list, size);
	cudaDeviceSynchronize();
}

//-------------------WRITE YOUR PROGRAM FROM HERE

int *host_lists[2];

int *devL1, *devL2;
bool *v;
int *devV[QUEUE_SIZE], *devResult;
int * devIndices;
int n, m;

int Lresult = 0;
int *resultList;
int *cpuResult;

struct list_info *cpuData;
struct list_info *gpuData;
int *data_pool;
int *gpu_data_pool;

__device__ int *list_p0[2]; // the origional   address
__device__ int *list_p[QUEUE_SIZE][2]; // save the starting position of either list
__device__ int calculated_indices_len[QUEUE_SIZE][2];
__device__ int swapped[QUEUE_SIZE]; // save swapped stage for each status
__device__ int *_result;
__device__ int gpu_result_size;
__device__ int _nm[2];
__device__ partition_info partitions_info[QUEUE_SIZE][DEF_D1*2 + 4];
__device__ int *_result_addr[QUEUE_SIZE];
__device__ debug_structure debug1;

__device__ int o_scan_buffers[QUEUE_SIZE][4+DEF_D1*2+32*2];
__device__ int *_scan_buffers[QUEUE_SIZE];

inline void move_pos(int &pos) {
	pos = (pos + 1) % QUEUE_SIZE;
}

__global__ void help_show_partation(int indices_now, int num_part,
		bool printout = false) {
	FOR_I(0,num_part)
	{
		struct partition_info *info = &partitions_info[indices_now][i];
		if (info->len <=0) continue;

		printf("Partation: %d\n", i);
		printf("[%llx]: %d --- [%llx]:%d\n", info->addr, *(info->addr),
				info->opposite_addr, *(info->opposite_addr));
		printf("[%lld]: --- [%lld]:\n", info->addr - list_p0[info->B2A],
				info->opposite_addr - list_p0[!info->B2A]);
		if (printout) {
			FOR_I(0,info->len*4)
				printf("%d \t", info->addr[i]);
			printf("\n |||||||| \n");
			int xx = info->right + (3 - info->right % 4) % 4 + 1;
			FOR_I(0, xx)
				printf("%d \t", info->opposite_addr[i]);
			//FOR_I(info->left,info->right+1) printf("%d \t",info->opposite_addr[i]);
			printf("\n");
		}
		printf("len:%d \t oppt_len:%d left:%d\t right:%d \t B2A:%d\n", info->len,info->len_opposite,
				info->left, info->right, info->B2A);
	}
}

__global__ void help_show_partation2(int indices_now, int num_part) {
	FOR_I(0,num_part)
	{
		struct partition_info *info = &partitions_info[indices_now][i];
		printf("Partation: %d\n", i);
		printf("len:%d \t left:%d\t right:%d \t B2A:%d warp_len:%d\n",
				info->len, info->left, info->right, info->B2A, info->warp_len);
		long long offset1, offset2;
		if (info->B2A) {
			offset1 = info->opposite_addr - list_p0[0];
			offset2 = info->addr - list_p0[1];
		} else {
			offset1 = info->addr - list_p0[0];
			offset2 = info->opposite_addr - list_p0[1];
		}
		printf("offset : < %lld --- %lld >", offset1, offset2);
		printf("\n---------------------\n");
	}
}



__global__ void show_list2(){
	FOR_I(0,10)
			printf("L1[%d] = %d\n",i,list_p0[0][i]);
	FOR_I(0,10)
			printf("L1[%d] = %d\n",i,list_p0[1][i]);

}

	void work(int l1_id,int l2_id) {

		int numStream;
		numStream = 4;
		cudaStream_t *streams = (cudaStream_t *) malloc(
				numStream * sizeof(cudaStream_t));
		for (int i = 0; i < numStream; i++) {
			cudaStreamCreate(&(streams[i]));
		}

		int D1, D2 ,D_save1,D_save2;
		int block_size;
		int block_2_size;
		//------ some settings ----------
		D1 = DEF_D1;
		D2 = DEF_D2;

		printf("CASE config:<%d %d>\n",D1,D2);

		D_save1 = max(1,2*D1/16); // one block deal 4 parts
		D_save2 = 64; // 64 threads per block for compacting

		//D1 = 16;D2 = 64;
		//D1 = 8;D2 = 64*4;//D2 has to be the multiply of WARP_SIZE, because

		cudaStream_t &cal_index_stream= streams[0];
		cudaStream_t &search_stream= streams[0];
		cudaStream_t &save_stream= streams[0];

		block_size = D1 * D2;
		block_2_size = block_size;
		//block_2_size = block_size*1000;



		//init_data(block_size);
		init_real_data(block_size,l1_id,l2_id);
		init_device_variables();

		dim3 cal_indx_setting(D1, 2);

		CudaWatch cudawatch;
		Watch cpuWatch;
		cpuWatch.start();
		cudawatch.start();
		cu_checkError();

		//printf("addr : %llx %llx",devL1,devL2);

		int len1, len2;
		int cal_pos = 0, search_pos = 0, save_pos = 0; // they are for L1, L2 position
		while (true) {
			back_next_relative_len(len1, len2, cal_pos);
			//cu_checkError();

			if (len1 <= 0 || len2 <= 0) break;
			int loops = min(len1, len2) / block_size;
			if ( min(len1,len2) % block_size != 0 ) loops ++;

			printf(" left n:%d  left m: %d loops:%d\n",len1,len2,loops);
			if (loops > 0) {
				//-- first stage
				cudaDeviceSynchronize();

				cal_indx<<<1, cal_indx_setting,0,cal_index_stream>>>(block_size, block_2_size,cal_pos);
				move_pos(cal_pos);
				//-- second stage
				if (loops >= 2){
					cal_indx<<<1, cal_indx_setting,0,cal_index_stream>>>(block_size, block_2_size,cal_pos);
					move_pos(cal_pos);
				}
				else cudaStreamSynchronize(cal_index_stream);
				algo2_search<<<2 * D1, D2 / 4,0,search_stream>>>(devV[search_pos], search_pos, 0);
				move_pos(search_pos);
				cudaStreamSynchronize(search_stream);

				//-- stage middle
				for (int i=3;i<loops;i++) {
					//help_debug<<<1,1>>>(loops);
					//outline;outline;outln(i);
					cal_indx<<<1, cal_indx_setting,0,cal_index_stream>>>(block_size, block_2_size,cal_pos);
					algo2_search<<<2 * D1, D2 / 4,0,search_stream>>>(devV[search_pos], search_pos, 0);
					scan_buffer_large<<<1, max( 2*D1,32),0,save_stream>>>(save_pos);
					compact<<<D_save1, D_save2,0,save_stream>>>(2*D1/D_save1,devV[save_pos],save_pos);
					move_pos(cal_pos);move_pos(search_pos);move_pos(save_pos);
					//cudaDeviceSynchronize();
				}
				cudaStreamSynchronize(save_stream);
				if (loops >=2 ){
					algo2_search<<<2 * D1, D2 / 4,0,search_stream>>>(devV[search_pos], search_pos, 0);
					scan_buffer_large<<<1, max( 2*D1,32),0,save_stream>>>(save_pos);
					compact<<<D_save1, D_save2,0,save_stream>>>(2*D1/D_save1,devV[save_pos],save_pos);
					move_pos(search_pos);move_pos(save_pos);
				}
				cudaStreamSynchronize(search_stream);
				scan_buffer_large<<<1, max( 2*D1,32),0,save_stream>>>(save_pos);
				compact<<<D_save1, D_save2,0,save_stream>>>(2*D1/D_save1,devV[save_pos],save_pos);
				move_pos(save_pos);
			}
			//return;
		}
		cudaDeviceSynchronize();
		//back__result<<<1,1>>>();
		cout << "MY Algo:" << cudawatch.stop() / 1000.0 << endl;
		cout << "MY Algo cpu test: " << cpuWatch.stop() << endl;
	}

int cpuResultSize = 0;
int merge_algo(int *array1, int *array2, int begin1, int end1, int begin2,
		int end2) {
	cpuResultSize = 0;
	int i = begin1, j = begin2;
	int lasti, lastj;

	lasti = array1[i];
	lastj = array2[j];
	while (i < end1 && j < end2) {
		if (lasti == lastj) {
			cpuResult[cpuResultSize++] = lasti;
			i++;
			j++;
			lasti = array1[i];
			lastj = array2[j];
		} else if (lasti < lastj) {
			i++;
			lasti = array1[i];
		} else {
			j++;
			lastj = array2[j];
		}
	}
	return cpuResultSize;
}

int check_correctness(int cpuResultSize) {
	FOR_I(0,cpuResultSize)
		if (cpuResult[i] != resultList[i]) {
			printf("Wrong at [%d] R:%d Yours:%d\n", i, cpuResult[i],
					resultList[i]);
			debug_a(cpuResult + i, -10, 10);
			debug_a(resultList + i, -10, 10);
			return i;
		}
	return -1;
}

void prepare_data(int n) {
	host_lists[0] = new int[n];
	resultList = new int[n];
	host_lists[1] = new int[n];
	v = new bool[20 * n];
	resultList = new int[n];
	cpuResult = new int[n];
}

__global__ void help_show_cal3(int indices_now, int num_part,bool printout = false) {
	FOR_I(0,num_part)
	{
		struct partition_info *info = &partitions_info[indices_now][i];
		printf("Partation: %d %d\n", i,info->len);
		if (printout) {
			FOR_I(0,info->len)
				printf("%d \t", info->addr[i]);
		}
		printf("len:%d \t left:%d\t right:%d \t B2A:%d\n", info->len,info->left, info->right, info->B2A);
	}
}

	void test_cal_idx3(){
		n = 16;
		generate_random(1.0, 0.5, 0.5);
		//generate_case_cal3();
		debug_a(host_lists[0],n);debug_a(host_lists[1],m);//debug
		int block_size = 8;
		init_data(block_size);
		init_device_variables();
		dim3 config(2,2);

		FOR_I(0,2){
//			cal_indx<<<1,config>>>(16,block_size,i);
//			//help_show_cal3<<<1,1>>>(0,4,true);
//			help_show_partation<<<1,1>>>(i,3,true);
//			algo2_search<<<2 * 2, 2>>>(devV[0], i, 0);

		}
		cudaDeviceSynchronize();
	}

	void performance_read(char **args,int &l1,int &l2){
		sscanf(args[1],"%d",&l1);
		sscanf(args[2],"%d",&l2);
	}

	void test_read(){
		read_gov2();
		exit(0);
	}

	void cpu_work(int l1_id,int l2_id){

		list_info a,b;
		a = cpuData[l1_id];
		b = cpuData[l2_id];

		printf("L1 use [%d] L2 use[%d]\n",l1_id,l2_id);
		printf("List 1 ( %d --- %d --- %d ) %d\n",a.start_addr[0],a.start_addr[ a.len / 2], a.start_addr[a.len-1],a.len);
		printf("List 1 ( %d --- %d --- %d ) %d\n",b.start_addr[0],b.start_addr[ b.len / 2], b.start_addr[b.len-1],b.len);

		Watch watch;
		watch.start();
		cpuResultSize = merge_algo(a.start_addr, b.start_addr, 0, a.len, 0,b.len);
		cout << "CPU ALGO time: " << watch.stop() << endl;
		cout << "CPU find " << cpuResultSize<< endl;
	}

int main(int arg_num, char ** args) {

	prepare_data(1024 * 1024 * 90);

	read_gov2();
	int l1,l2;
	performance_read(args,l1,l2);
	//l1 = 171,l2=164;

//	test_cal_idx3();
//	return 0;

	FOR_I(155,10000)
	{
		srand(time(0) % 1234567);


//	work3();//test bsearch
//	return 0;

		cpu_work(l1,l2);

		work(l1,l2);
		cuda_copyResult();cout<<"copied back"<<endl;
		cu_checkError();
		free_device_memory();
		printf(" results / elements = %d / %d %lf\n", Lresult, n,(1.0 * Lresult) / (1.0 * n));

		//sort(resultList, resultList + Lresult);
		outln(Lresult);
		outln(cpuResultSize);
//		debug_a(resultList,Lresult+20);
//		debug_a(cpuResult,cpuResultSize);

		//continue;// DO NOT CHECK correctness

		int error_index;
		if ((error_index = check_correctness(cpuResultSize)) >= 0) {
			outline;
			outln(i);
			printf("all length: cpu:%d gpu:%d  ration: %.5lf\n", cpuResultSize,Lresult, 100.0 * error_index / (1.0 * Lresult));
			break;
		}
		outline;
		outline;
		break; //only execute once
	} //end FOR
	cudaDeviceReset();
	return 0;
}

