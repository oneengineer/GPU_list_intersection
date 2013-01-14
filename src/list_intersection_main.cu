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

#include "others/generate_data.h"

//#include "cudpp.h"


using namespace std;

	template <typename T>
	void debug_a(T * data,int begin,int end){
		for (int i=begin;i<end;i++) cout<<"["<<i<<"]: "<<data[i]<<"\t";cout<<endl;
	}
	template <typename T>
	void debug_a(T * data,int end){
		debug_a(data,0,end);
	}
	template <typename T>
	void debug_a2(T * data,int end1,int end2){
		for (int i=0;i<end1;i++){cout<<"row "<<i<<endl; for (int j=0;j<end2;j++) cout<<"["<<i<<","<<j<<"] "<<data[i][j]<<"\t";cout<<endl;}
	}
	double get_sec(const struct timeval & tval){
		return ((double)(tval.tv_sec*1000*1000 + tval.tv_usec))/1000000.0;
	}

	template <typename T>
	T checkmin(T & data,T value){
		data = min(data,value);
		return data;
	}
struct Watch{
	timeval begin,end;
	void start(){gettimeofday(&begin,NULL);}
	double time(){return get_sec(end)-get_sec(begin);}
	double stop(){gettimeofday(&end,NULL);return time();}
};

	inline double rand_double(double range){
		return  ((double)rand()/(double)RAND_MAX)*range;
	}
//----------- cuda template ------------

struct CudaWatch{
	cudaEvent_t start_event, stop_event;
	void start(){
	cudaEventCreate(&start_event) ;
	cudaEventCreate(&stop_event) ;
	cudaEventRecord(start_event,0);
	}
	float stop(){
		float time_cost = -1.0;
		cudaEventRecord(stop_event,0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&time_cost,start_event,stop_event);
		return time_cost;
	}
};
	void cu_checkError(){
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess){
			printf("ERROR: %s\n",cudaGetErrorString(error));
			exit(-1);
		}
	}

	__global__ void cu_print(int *list,int size){
		printf("CUDA SHOW:\n");
		if ( 0 == threadIdx.x && 0 == blockIdx.x ){
			FOR_I(0,size)
				printf("[%d]:%d\t",i,list[i]);
			printf("\n");
		}
	}

	void cu_host_print(int *list,int size){
		cudaDeviceSynchronize();
		cu_print<<<1,1>>>(list,size);
		cudaDeviceSynchronize();
	}


//-------------------WRITE YOUR PROGRAM FROM HERE



int *host_lists[2];

int *devL1,*devL2;
bool *v;
int *devV[3],*devResult;
int * devIndices;
int n,m;

int Lresult;
int *resultList;
int *cpuResult;

__device__ int *list_p0[2];// the origional   address
__device__ int *list_p[QUEUE_SIZE][2]; // save the starting position of either list
__device__ int calculated_indices_len[QUEUE_SIZE][4];
__device__ int swapped[QUEUE_SIZE];   // save swapped stage for each status
__device__ int *_result;
__device__ int gpu_result_size;
__device__ int _nm[2];
__device__ int partitions[QUEUE_SIZE][512+4][2];
__device__ partition_info partitions_info[QUEUE_SIZE][512+4];

__device__ debug_structure debug1;

	inline void move_pos(int &pos){
		pos = (pos + 1) % QUEUE_SIZE;
	}

//	CUDPPHandle prepare_prefixsum(int size, cudaStream_t * stream){
//		CUDPPConfiguration config;
//		config.op = CUDPP_ADD;
//		config.datatype = CUDPP_INT;
//		config.algorithm = CUDPP_SCAN;
//		config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
//		config.mystream = stream;
//
//		CUDPPHandle theCudpp;
//		cudppCreate(&theCudpp);
//		CUDPPHandle scanplan = 0;
//		CUDPPResult res = cudppPlan(theCudpp,&scanplan,config,size,1,0);
//		if ( CUDPP_SUCCESS != res ){
//			printf("ERROR in prepare_prefixsum\n");
//			exit(-1);
//		}
//		return scanplan;
//	}

	__global__ void save_one_core(int search_now,int *devV){
		int num = 0;
		int len = calculated_indices_len[search_now][0];
		FOR_I(0,len+1)
			if ( devV[i] ){
				_result[ num++ ] = list_p[search_now][0][i];
				//printf("in save : saved :%d\n",list_p[search_now][0][i]);//debug
			}
		_result += num;
	}

	__global__ void help_show_partation(int indices_now,int num_part,bool printout=false){
		FOR_I(0,num_part){
			struct partition_info *info = &partitions_info[indices_now][i];
			printf("Partation: %d\n",i);
			printf("[%llx]: %d --- [%llx]:%d\n",info->addr,*(info->addr),info->opposite_addr,*(info->opposite_addr));
			printf("[%lld]: --- [%lld]:\n",info->addr - list_p0[ info->B2A ],info->opposite_addr - list_p0[!info->B2A]);
			if (printout){
				FOR_I(0,info->len*4) printf("%d \t",info->addr[i]);
				printf("\n |||||||| \n");
				int xx= info->right + (3- info->right%4)%4 +1;
				FOR_I(0, xx) printf("%d \t",info->opposite_addr[i]);
				//FOR_I(info->left,info->right+1) printf("%d \t",info->opposite_addr[i]);
				printf("\n");
			}
			printf("len:%d \t left:%d\t right:%d \t B2A:%d\n",info->len,info->left,info->right,info->B2A);
		}
	}

	__global__ void help_show_partation2(int indices_now,int num_part){
		FOR_I(0,num_part){
			struct partition_info *info = &partitions_info[indices_now][i];
			printf("Partation: %d\n",i);
			printf("len:%d \t left:%d\t right:%d \t B2A:%d\n",info->len,info->left,info->right,info->B2A);
			long long offset1,offset2;
			if (info->B2A){
				offset1 = info->opposite_addr - list_p0[0];
				offset2 = info->addr - list_p0[1];
			}
			else{
				offset1 = info->addr - list_p0[0];
				offset2 = info->opposite_addr - list_p0[1];
			}
			printf("offset : < %lld --- %lld >",offset1,offset2);
			printf("\n---------------------\n");
		}
	}

	__global__ void back__result(){
		_result += gpu_result_size;
	}

	__global__ void show_addr_value(int *devL,int n){
		FOR_I(0,n){
			printf("[%llx]: %d \t",devL+i,devL[i]);
		}
		printf("\n");
	}

	__global__ void help_debug(int loop){
		debug1.num_loop = loop;
	}

	void work(){

		int numStream;
		numStream = 4;
		cudaStream_t *streams = (cudaStream_t *) malloc(numStream * sizeof(cudaStream_t));
		for (int i = 0; i < numStream; i++)
		{
			cudaStreamCreate(&(streams[i]));
		}

		int D1,D2,D1save;
		int block_size;
		int block_2_size;
		//------ some settings ----------
		D1save = 16;
		D1 = 128;D2 = 8*32*4;

		//D1 = 16;D2 = 64;
		//D1 = 8;D2 = 64*4;//D2 has to be the multiply of WARP_SIZE, because

		int save_stream = 2;
		int search_stream = 1;

		block_size = D1 * D2;
		block_2_size = block_size;
		//block_2_size = block_size*1000;

		init_data(block_size);
		init_device_variables();
		init_scan(&streams[save_stream],1024);

		int num_small_block = D1;
		dim3 cal_indx_setting(num_small_block,2);

		CudaWatch cudawatch;
		Watch cpuWatch;cpuWatch.start();
		cudawatch.start();
		cudaFuncSetCacheConfig(algo2_search,cudaFuncCachePreferShared);
		cu_checkError();

		//CUDPPHandle prefixsum_plan = prepare_prefixsum(block_size,streams+2);
		//show_addr_value<<<1,1>>>(devL1,n);
		//show_addr_value<<<1,1>>>(devL2,m);

		int len1,len2;
		int cal_pos = 0, search_pos = 0, save_pos  =0; // they are for L1, L2 position
		while (true){
			int devVinc = 0; // use which devV to store data
			back_next_relative_len(len1,len2,cal_pos);
			//cu_checkError();
			outln(len1);outln(len2);//debug
			if ( len1<=0 || len2 <= 0 ) break;
			int loops = min(len1,len2)/block_size;
			if ( min(len1,len2) % block_size != 0 )
				loops ++;

			if ( loops >0 ){
				outln(loops);
				int *saveV; //pointer of saving result

				//-- stage middle
				for ( ;loops> 0 ;loops -- ){
					help_debug<<<1,1>>>(loops);
					outline;outline;outln(loops);
					saveV = devV[ devVinc];
					cal_indx<<<1,cal_indx_setting>>>( block_size,block_2_size,cal_pos);
					//cudaDeviceSynchronize();
					if (1 == loops){
						//help_show_partation2<<<1,1>>>(cal_pos,128);//debug
						//return;
					}
					//cudaDeviceSynchronize();
					algo2_search<<< 2 * D1 , D2/4 >>>(saveV,search_pos,0);

					move_pos(cal_pos);move_pos(search_pos);move_pos(save_pos);

					//mssleep(10);
					cudaDeviceSynchronize();
					//back__result<<<1,1>>>();return;
				}
				//break;//debug
				//cudaDeviceSynchronize();
			}

			cu_checkError();
		}
		cudaDeviceSynchronize();
		back__result<<<1,1>>>();
		cout<<"MY Algo:"<<cudawatch.stop()/1000.0<<endl;
		cout<<"MY Algo cpu test: "<<cpuWatch.stop()<<endl;
	}

int cpuResultSize = 0;
	int merge_algo(int *array1,int *array2, int begin1,int end1,int begin2,int end2){
		cpuResultSize = 0;
		int i=begin1,j=begin2;
		int lasti,lastj;

		lasti=array1[i];
		lastj=array2[j];
		while ( i<end1 && j<end2){
			if ( lasti == lastj ){
				cpuResult[cpuResultSize++] = lasti;
				i++;j++;
				lasti=array1[i];
				lastj=array2[j];
			}
			else if ( lasti < lastj){
				i++;
				lasti = array1[i];
			}
			else {
				j++;
				lastj = array2[j];
			}
		}
		return cpuResultSize;
	}

	int check_correctness(int cpuResultSize){
			FOR_I(0,cpuResultSize)
				if (cpuResult[i] != resultList[i]){
					printf("Wrong at [%d] R:%d Yours:%d\n",i,cpuResult[i],resultList[i]);
					debug_a( cpuResult+i,-10,10 );
					debug_a( resultList+i,-10,10 );
					return i;
				}
		return -1;
	}

	void prepare_data(int n){
		host_lists[0] = new int [n];
		resultList = new int[n];
		host_lists[1] = new int [n];
		v = new bool [10*n];
		resultList = new int[n];
		cpuResult = new int [n];
	}

#if 0

	extern int * scan_buffer;

	__global__ void set_somethings(int *L,int n){
		calculated_indices_len[0][0] = n;
	}

	void test_scan_save(){
		n = 1024*128;
		generate_random(1.0,2.0,2.0);
		init_data(1024*256);
		init_device_variables();
		int *temp = new int [n];
		int *tempR = new int [n];
		FOR_I(0,n) temp[i] = (rand() % 50 )== 1;
		cudaMemcpy(devV[0],temp,sizeof(int)*n,H_T_D);
		int num = 0;
		int count = 0;

		FOR_I(0,n) if ( temp[i]) tempR[num++] = host_lists[0][i];

		set_somethings<<<1,1>>>(devV[0],n);
		//cu_host_print(devV[0],n);
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		init_scan(&stream,1024);
		ScanSequence seq(1,devV[0],n);
		seq.save_indices_now = 0;
		seq.init();
		seq.run_all();

//		seq.run_scan(0);
//		seq.run_large();
//		seq.run_save(0);

		seq.run_saveLarge();
		cudaMemcpy(resultList,devResult,sizeof(int)*num,D_T_H);


		FOR_I(0,num){
			if ( tempR[i] != resultList[i] )
				cout<<"ERROR AT "<<i<<" "<<tempR[i]<<" "<<resultList[i]<<endl;
		}
//		debug_a(tempR,num);
//		debug_a(resultList,num);

		//cu_host_print(devResult,n);
		cudaDeviceSynchronize();
		cout<<"Test over "<<endl;
	}



	__global__ void help_test_cal_indices(){
		FOR_I(0,16){
			printf("i: %d (%d %d)\n",i,partitions[0][i][0],partitions[0][i][1]);
		}
	}

	__global__ void show_part(){
		FOR_I(0,16)
			printf("%d  [%d] -- [%d]\n",i,partitions[0][i][0],partitions[0][i][1]);

	}

	void test_cal_indices(){
		n = 4 * 128;
		generate_random(1.2,1.5,1.5);
		//FOR_I(0,n) host_lists[0][i] =2*i ,host_lists[1][i] = i;
		init_data(128);
		init_device_variables();

		int block_size = 128;
//		debug_a(host_lists[0],block_size+10);
//		debug_a(host_lists[1],block_size+10);


		dim3 d(8,2);
		cal_indx<<<1,d>>>(block_size,block_size,0);
		cu_checkError();
		//help_test_cal_indices<<<1,1>>>();
		cu_checkError();
		cudaDeviceSynchronize();

		algo2_search<<<16,16>>>(devV[0],0,0);
		cu_checkError();
		cudaDeviceSynchronize();

		cpuResultSize = 0;
		merge_algo(host_lists[0],host_lists[1],0,128,0,128);
//		cout<<"Cpu result"<<endl;debug_a(cpuResult,cpuResultSize);//debug
		resultList = new int [200];

		cudaMemcpy(resultList,devV[0],sizeof(int)*128,D_T_H);
		int num_dev = 0;
		FOR_I(0,128)
			if (resultList[i]){
//				printf("[%d] %d\t",i,host_lists[0][i]);
				resultList[num_dev++] = host_lists[0][i];
			}printf("\n");

		cout<<"Check Correctness"<<endl;

		if (num_dev != cpuResultSize) { printf("Wrong num!");exit(1);}

		FOR_I(0,cpuResultSize)
			if (  resultList[i] != cpuResult[i] ){
				printf("Wrong at [%d] %d %d",i,resultList[i],cpuResult[i]);
				exit(1);
			}
		cout<<"It is correct"<<endl;
	}

#endif


	__global__ void show_debug_1(){
		printf("wrong 1 times: %d\n",debug1.wrong_1);
		printf("wrong 2 times: %d\n",debug1.wrong_2);
	}

int main(int arg_num,char ** args){
	//cudaDeviceReset();

	prepare_data(1024*1024*90);

	FOR_I(155,10000){
		srand(i);
		FOR_J(0,arg_num){
			char ch = args[j][0];
			if ( '0'<= ch && ch<='9'){
				int seed;
				sscanf(args[j],"%d",&seed);
				srand(seed);
				printf("SEED: %d\n",seed);
			}
		}
		//srand(710852);
	n = 1024*1024*40;
	//n = 1024*1024*10;
	//n = 1024*1024;
	//n = 1024*102;
	//n = 5000;
	//n = 660;
	//n = 66;
	//n = 20;

	//generate_case5();
	//generate_random(1.0,2.0,2.0);
	//n = m = 70;
	//host_lists[1][16] = 65;
	//generate_same(2.0);
	generate_shift(2.0,1);

	cout<<"generate data over srand("<<i<<") n="<<n<<" m="<<m<<endl;
	//debug_a(host_lists[0],n);debug_a(host_lists[1],m);//debug
	printf("List 1 ( %d --- %d --- %d )\n",host_lists[0][0],host_lists[0][n/2],host_lists[0][n-1]);
	printf("List 2 ( %d --- %d --- %d )\n",host_lists[1][0],host_lists[1][m/2],host_lists[1][m-1]);

	int sum1=0,sum2=0;
	FOR_K(0,n) if (k%4 ==3) { sum1 += host_lists[0][k],sum2 += host_lists[1][k]; }

	Watch watch;watch.start();
	int cpuResultSize = merge_algo(host_lists[0],host_lists[1],0,n,0,m);
	cout<<"CPU ALGO time: "<<watch.stop()<<endl;

//	work3();//test bsearch
//	return 0;


	work();
	show_debug_1<<<1,1>>>();
	outln(sum1);outln(sum2);
	//cuda_copyResult();cout<<"copied back"<<endl;
	cu_checkError();
	free_device_memory();
	printf(" results / elements = %d / %d %lf\n",Lresult,n,(1.0*Lresult)/(1.0*n));

	sort(resultList,resultList + Lresult);
	outln(Lresult);outln(cpuResultSize);
	//debug_a(resultList,Lresult);
	//debug_a(cpuResult,cpuResultSize);
	//break;

	//continue;// DO NOT CHECK correctness

	int error_index;
	if  ( (error_index = check_correctness(cpuResultSize)) >=0 ){
		outline;
		outln(i);
		printf("all length: cpu:%d gpu:%d  ration: %.5lf\n",cpuResultSize,Lresult,100.0*error_index/(1.0*Lresult));
		break;
	}
		outline;
		outline;
		break; //only execute once
	}//end FOR
	return 0;
}

