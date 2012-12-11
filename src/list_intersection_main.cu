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
int *devV[3],*devResult, *devMark;
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
__device__ int _nm[2];

	inline void move_pos(int &pos){
		pos = (pos + 1) % QUEUE_SIZE;
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
		D1 = 256;
		D1save = 16;
		D2 = 512;
		//D1 = 32;
		//D2 = 32;
		//D1save = 8;

		int save_stream = 2;
		int search_stream = 1;

		block_size = D1 * D2;
		block_2_size = block_size;
		//block_2_size = block_size*1000;

		init_data(block_size);
		init_device_variables();
		init_scan(&streams[save_stream],1024);
		ScanSequence scanSeq(2,devV[0],block_size);
		scanSeq.init();

		SearchSettingQueue searchQue(2,&(streams[search_stream]));
		int searchConfig[][2] = { {64,512},{64,512},{64,512},{64,512} };
		int searchConfig8[][2] = { {32,512},{32,512},{32,512},{32,512},{32,512},{32,512},{32,512},{32,512} };
		int searchConfig1[][2] = { {256,512}};
		int searchConfig2[][2] = { {128,512},{128,512}};
		searchQue.setSettings(searchConfig2);
		searchQue.init();
		outln(searchQue.length());


		CudaWatch cudawatch;
		Watch cpuWatch;cpuWatch.start();
		cudawatch.start();

		int len1,len2;
		int cal_pos = 0, search_pos = 0, save_pos  =0; // they are for L1, L2 position
		while (true){
			int devVinc = 0; // use which devV to store data
			back_next_relative_len(len1,len2,cal_pos);
			cu_checkError();
			outln(len1);outln(len2);//debug
			if ( len1<=0 || len2 <= 0 ) break;
			int loops = min(len1,len2)/block_2_size;
			if ( loops >0 ){
				outln(loops);
				bool lastButOne = loops > 1;
				int *saveV; //pointer of saving result

				//-- stage 1
				cal_indx<<<1,4,0,streams[0]>>>(block_size,block_2_size,cal_pos);
				move_pos( cal_pos );

				//-- stage 2
				if ( lastButOne ){
					cal_indx<<<1,4,0,streams[0]>>>(block_size,block_2_size,cal_pos);
					searchQue.set(devV[devVinc],search_pos);
					searchQue.run_all();
					move_pos( cal_pos );move_pos( search_pos );
				}
				cudaDeviceSynchronize();

				//-- stage middle
				for ( loops -= 2; loops> 0 ;loops -- ){
					//outln(loops);
					saveV = devV[ devVinc];devVinc = (devVinc+1) % 3;
					scanSeq.set(saveV,save_pos);searchQue.set(devV[devVinc],search_pos);


					scanSeq.run_scan(0);
					searchQue.run_search(0);
					scanSeq.run_scan(1);
					scanSeq.run_large();
					cal_indx<<<1,4,0,streams[0]>>>(block_size,block_2_size,cal_pos);
					scanSeq.run_save(0);
					searchQue.run_search(1);
					scanSeq.run_save(1);
					scanSeq.run_saveLarge();


//					scanSeq.run_all();
//					searchQue.run_all();


					move_pos( cal_pos );move_pos( search_pos );move_pos( save_pos );
					cu_checkError();
					//mssleep(10);
				}
				cudaDeviceSynchronize();
				//-- stage last but one
				if (lastButOne){
					saveV = devV[ devVinc];devVinc = (devVinc+1) % 3;
					scanSeq.set(saveV,save_pos);
					scanSeq.run_all();
					move_pos( save_pos );
				}
				searchQue.set(devV[devVinc],search_pos);
				searchQue.run_all();
				move_pos( search_pos );
				cudaDeviceSynchronize();
				//----- stage last --------
				saveV = devV[ devVinc];devVinc = (devVinc+1) % 3;
				scanSeq.set(saveV,save_pos);
				scanSeq.run_all();
				move_pos( save_pos );
			}
			else {
				cout<<"small seg "<<len1<<" "<<len2<<endl;
				move_indices<<<1,1,0,streams[save_stream]>>>(len1,len2,block_size,cal_pos);
				searchQue.set(devV[0],search_pos);
				searchQue.run_all();
				scanSeq.set( devV[0] ,save_pos);
				scanSeq.run_all();
				move_pos(cal_pos);move_pos(search_pos);move_pos(save_pos);
			}
			cu_checkError();
		}
		cudaDeviceSynchronize();
		cout<<"MY Algo:"<<cudawatch.stop()/1000.0<<endl;
		cout<<"MY Algo cpu test: "<<cpuWatch.stop()<<endl;
	}

	int merge_algo(int *array1,int *array2, int begin1,int end1,int begin2,int end2){
		//return ;
		int i=begin1,j=begin2;
		int lasti,lastj;
		int cpuResultSize = 0;
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
#endif

int main(){

	prepare_data(1024*1024*40);

//	test_scan_save();
//	return 0;

	FOR_I(103,10000){
	//r =1344532745 ;
	srand(i);
	n = 1024*1024*40;
	//n = 1024*1024;
	//n = 50;

	//generate_case5();
	generate_random(1.0,2.0,2.0);

//	FOR_I(0,n){ if ( 7592 == host_lists[1][i] ) printf("7592 = [%d]\n",i);
//	if ( 962329 == host_lists[1][i] ) printf("962329 = [%d]\n",i);
//	}

	cout<<"generate data over srand("<<i<<") n="<<n<<" m="<<m<<endl;
	printf("List 1 ( %d --- %d --- %d )\n",host_lists[0][0],host_lists[0][n/2],host_lists[0][n-1]);
	printf("List 2 ( %d --- %d --- %d )\n",host_lists[1][0],host_lists[1][m/2],host_lists[1][m-1]);


	Watch watch;watch.start();
	int cpuResultSize = merge_algo(host_lists[0],host_lists[1],0,n,0,m);
	cout<<"CPU ALGO time: "<<watch.stop()<<endl;

//	work3();//test bsearch
//	return 0;


	work();
	cuda_copyResult();cout<<"copied back"<<endl;
	cu_checkError();
	free_device_memory();
	printf(" results / elements = %d / %d %lf\n",Lresult,n,(1.0*Lresult)/(1.0*n));
	outln(Lresult);outln(cpuResultSize);

	//outln(Lresult);outln(cpuResultSize);
	//debug_a(resultList,Lresult);
	//debug_a(cpuResult,cpuResultSize);
	//break;

	//continue;// DO NOT CHECK correctness

	int error_index;
	if  ( (error_index = check_correctness(cpuResultSize)) >=0 ){
		outline;
		outln(i);
		printf("all length: cpu:%d gpu:%d  ration: %.5lf\n",cpuResultSize,Lresult,100.0*i/(1.0*Lresult));
		break;
	}
		outline;
		outline;
		break; //only execute once
	}//end FOR
	return 0;
}

