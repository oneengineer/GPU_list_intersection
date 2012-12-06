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
//#include <string>
//#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include <unistd.h>
#include <sys/time.h>
#include <cudpp.h>

#include "t_scan.h"

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

#define SWAP(a,b) { a=(a)-(b);b=(b)+(a);a=(b)-(a);}

//------ for CUDA ------
#include <cuda.h>
//#include <cutil.h>
#include <cuda_runtime.h>
#define H_T_D cudaMemcpyHostToDevice
#define D_T_H cudaMemcpyDeviceToHost
#define D_T_D cudaMemcpyDeviceToDevice
#define CUID (threadIdx.x + blockIdx.x*blockDim.x)
#define CUSIZE (blockDim.x*gridDim.x)

using namespace std;

//for mpi
#define MCW MPI_COMM_WORLD

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


const int N = 55000100;
const int M = 1024*1024;

int list[2][N];
int *devL1,*devL2;

bool v[N*10];

int n,m;

int V[M];
int resultList[N];
int cpuResult[N],cpuResultSize;

int *devV[2],*devResult, *devMark;

int L2index,L1index,Lresult;
int *devL2index,*devL1index,*devResultIndex;

int block_size;

const int QUEUE_SIZE= 4;

__device__ int *list_p0[2];
__device__ int *list_p[QUEUE_SIZE][2]; // save the starting position of either list
__device__ int calculated_length[QUEUE_SIZE];
__device__ int calculated_indices_len[QUEUE_SIZE][4];
__device__ int swapped[QUEUE_SIZE];   // save swapped stage for each status
__device__ int _nm[2];
__device__ int *_result;

__device__ double ratio_avg;
__device__ int cal_times;

int * devIndices;

__shared__ int share[2048*2];

	void generate_different(int * array,int n,int mod){
		FOR_I(0,mod) v[i] = false;
		for (int i=0;i<n;i++){
			int j=-1;
			do
			{
				j = rand() % mod;
			}while ( v[j] );
			v[j] = true;
			array[i] = j;
		}
	}

	void generate_random(double alpha=1.0,double scala1=1.0,double scala2=1.0){
		int mod = n*4;
		m = (int)((double)n*alpha + eps);
		int mod1 = (int)((double)mod*scala1 + eps);
		int mod2 = (int)((double)mod*scala2 + eps);
		generate_different(list[0],n,mod1);
		sort(list[0],list[0]+n);
		generate_different(list[1],m,mod2);
		sort(list[1],list[1]+m);
	}

	void generate_case2(){
		printf("a general test. mainly for the search bound\nPlease use D1 = D2 = 2,block2size = 5\n\n");
		int temp1[] = {1,2,3,4,           20,            29, 34,36,37,38 ,39,30,66
					,100,110,120,130,140,200,211,230,231,540 };
		int temp2[] = {1,2,3,4,6,7,10,12, 20,25,26,27,28 ,29                   ,60
				,100,110,120,130,140,200,210,220,230,540};

		n = sizeof(temp1)/sizeof(int);
		m = sizeof(temp2)/sizeof(int);
		FOR_I(0,n) list[0][i] = temp1[i];
		FOR_I(0,m) list[1][i] = temp2[i];
	}

	void generate_case3(){
		printf("a generate test. mainly for the search swap\nPlease use D1 = 1; D2 = 3,block2size = 5\n\n");
		int temp1[] = {1,  2,  9,   11,15,16,17,26,27
					,100,110,120,130,140,200,211,230,231,540 };
		int temp2[] = {1,2,5,6,9,          21,25,  27
					,100,110,120,130,140,200,210,220,230,540 };

		n = sizeof(temp1)/sizeof(int);
		m = sizeof(temp2)/sizeof(int);
		FOR_I(0,n) list[0][i] = temp1[i];
		FOR_I(0,m) list[1][i] = temp2[i];
	}

	void generate_case4(){
		printf("a end point test. Please use D1 = 2; D2 = 2,block2size = 6\n\n");
		int temp1[] = {196};
		int temp2[] = {195,196,200};

		n = sizeof(temp1)/sizeof(int);
		m = sizeof(temp2)/sizeof(int);
		FOR_I(0,n) list[0][i] = temp1[i];
		FOR_I(0,m) list[1][i] = temp2[i];
	}

	void generate_case5(){
		printf(" a total equal test \n\n");
		n = 64;
		m = 64;
		FOR_I(0,n) list[0][i] = list[1][i] = i*10 + rand() % 5;
	}



	void init_data(){
		int sizeV = block_size + 10;
		int *temp = new int [sizeV];
		FOR_I(0,sizeV) temp[i] = 0;

		cudaMalloc( (void **)&devL1, sizeof(int)*n );
		cudaMalloc( (void **)&devL2, sizeof(int)*m );
		cudaMalloc( (void **)&(devV[0]), sizeof(int)*sizeV );
		cudaMalloc( (void **)&(devV[1]), sizeof(int)*sizeV );
		cudaMalloc( (void **)&devResult, sizeof(int)*n );
		cudaMalloc( (void **)&devMark, sizeof(int)*(n+1) );
		cu_checkError();

		cudaMalloc( (void **)&devL2index, sizeof(int) );
		cudaMalloc( (void **)&devL1index, sizeof(int) );
		cudaMalloc( (void **)&devResultIndex, sizeof(int) );
		cu_checkError();

		cudaMemcpy( devL1,list[0],sizeof(int)*n,H_T_D );
		cudaMemcpy( devL2,list[1],sizeof(int)*m,H_T_D );
		cudaMemcpy( devMark,temp,sizeof(int),H_T_D );
		cudaMemcpy( devV[0],temp,sizeof(int)*sizeV,H_T_D );
		cudaMemcpy( devV[1],temp,sizeof(int)*sizeV,H_T_D );
		devV[0] +=4; // memeory allign for cudpp
		devV[1] +=4; // memeory allign for cudpp

		cu_checkError();
	}

	void show_lists(int index1,int index2,int len1=n,int len2=n){
		printf("Two lists:\n");
		printf("L1 :\t");debug_a(list[0]+index1,len1);
		printf("L2 :\t");debug_a(list[1]+index2,len2);
		outline;
	}

	void cuda_copyResult(){
		int *end;
		cudaMemcpyFromSymbol(&end,_result,sizeof(int *),0,D_T_H );
		cu_checkError();
		Lresult = (int)(end - devResult );
		cudaMemcpy( resultList,devResult, sizeof(int)*Lresult,D_T_H );
		cu_checkError();
	}

	__device__ int next_queue_pos(int value){
			return (value+1) % QUEUE_SIZE;
	}

	__global__ void prefix_sum_oneCore(int *V,int block_size){
		FOR_I(1,block_size)
			V[i] += V[i-1];
	}

//------------------ prefix sum over ---------------

	//binary search lower bound in list2 for each element in list1
	//algo2 should not care about swap or not
	__global__ void algo2_search(int * V,int search_now,int offset = 0){
		int id = CUID;
		id += offset;
		if (id > calculated_indices_len[search_now][0]){
//			printf("id %d quit!\n",id);
			return;
		}
		int left = 0;
		int right = calculated_indices_len[search_now][1];

		int *list1 = list_p[ search_now ][ 0 ];
		int *list2 = list_p[ search_now ][ 1 ];
		int value = list1[ id ];
//		if(!id) { printf("R:%d\n",right) ;FOR_I(0,right+1) printf("!%d\t",list2[i]); printf("\n"); }
		int result = 0;
		while ( left <= right ){
			int mid = (left + right)/2;
			if ( value == list2[ mid ] ){
//				printf("id: %d INTERSECTION %d\n",id,value);
				result = 1;
				break;
			}
			else if ( list2[mid] < value )
				left = mid + 1;
			else right = mid - 1;
		}
		V[id] = result;
//		printf("  algo2 search id: %d  %d get [%d] %d\n",id, value,right,list2[right]);
	}

	__global__ void save_result(int *V,int indices_now){
		int _size = calculated_indices_len[indices_now][0];
		int id = CUID;
		int cusize = CUSIZE;
		int *list1 = list_p[indices_now][0];


		while (id <= _size ){
			int diff = V[id] - V[id-1];
			if ( diff >0){
				//if (diff !=1) printf("ERROR !=1\n");
				if ( V[id-1] > 256*512 ) {
					printf("ERROR > D1*D2 %d  id:[%d]  says save to V:%llx \n",V[id-1],id,V);
				}
				//if ( V[id-1] <0 ) printf("ERROR < 0\n");
				//printf("save result %d\n",list1[ id ]);
				_result[ V[id-1] ] = list1[ id ];
			}
			id += cusize;
		}
	}

	//can only be executed by one core
	__global__ void move_result_addr(int *V,int indices_now){
		int _size = calculated_indices_len[indices_now][0];
		_result += V[_size];
	}

	__global__ void init_device_var_kernel(){

		int id = CUID;
		if (id) return;
		list_p[0][0] = list_p0[0];
		list_p[0][1] = list_p0[1];
		swapped[0] = 0;//false
		ratio_avg = 0.0;//for debug
		cal_times = 0;//for debug
	}


	void init_device_variables(){
		cudaMemcpyToSymbol(list_p0,&devL1,sizeof(int *),0,H_T_D);
		cu_checkError();
		cudaMemcpyToSymbol(list_p0,&devL2,sizeof(int *),0+sizeof(int *),H_T_D);
		int temp[] = {n,m};
		cudaMemcpyToSymbol(_nm,temp,sizeof(int)*2,0,H_T_D);

		cudaMemcpyToSymbol( _result,&devResult,sizeof(int*),0,H_T_D);
		cu_checkError();
		init_device_var_kernel<<<1,1>>>();
		cu_checkError();
		cudaDeviceSynchronize();
		cudaMalloc( (void **)&devIndices , sizeof(int )*2);
	}

	void free_device_memory(){
		cudaFree(devL1);
		cudaFree(devL2);
		cudaFree(devResult);
		cudaFree(devMark);
		cu_checkError();
	}

	__global__ void cal_indx (int block_size,int block_2_size,int indices_now){
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

	//special case for cal_indices
	__global__ void move_indices(int len1,int len2,int block_size,int indices_now){
		int indices_next = (indices_now + 1) % QUEUE_SIZE;
		int swapflag = 0;
		//make sure that len1 is always the shorter one
		if ( len1 > len2 ){
			int *temp = list_p[indices_now][0];
			list_p[indices_now][0] = list_p[indices_now][1];
			list_p[indices_now][1] = temp;
			SWAP(len1,len2);
			swapflag = 1;
		}
		int b1 = min(len1,block_size);
		list_p[ indices_next ][ 0 ] = list_p[ indices_now ][ 0 ] + b1;
		list_p[ indices_next ][ 1 ] = list_p[ indices_now ][ 1 ] ;//keep the same
		swapped[ indices_next ] = swapped[ indices_now ] ^ swapflag;
		calculated_indices_len[ indices_now ][ 0 ] = b1-1;// or b1
		calculated_indices_len[ indices_now ][ 1 ] = len2-1;// or b1
	}


	// configure : <<<1,2>>> only and but only 2 threads
	__global__ void helpLen_relative(int *memory,int indices_now){
			int temp = indices_now; //next_queue_pos(indices_now);
			int id = threadIdx.x;
			int isSwapped = swapped[ temp ];
			int myindex = id ^ isSwapped;// id:1  !swapped
			memory[id] = list_p[temp][ id ]-list_p0[ myindex ];
			memory[id] = _nm[myindex] - memory[id];
		}

	void back_next_relative_len(int & len1,int & len2,int indices_now){
		int temp[2];
		helpLen_relative<<<1,2>>>(devIndices, indices_now);
		cudaMemcpy(temp,devIndices,sizeof(int)*2, D_T_H);
		len1 = temp[0];len2 = temp[1];
	}

	__global__ void show_cur_lists(int len1,int len2,int indices_now){
		FOR_I(0, calculated_indices_len[indices_now][0] )
				printf("{%d} %d\t",i,list_p[indices_now][0][i]);printf("\n");
		FOR_I(0, calculated_indices_len[indices_now][1])
				printf("{%d} %d\t",i,list_p[indices_now][1][i]);printf("\n");
	}

	CUDPPHandle prepare_prefixsum(int size, cudaStream_t * stream){
		CUDPPConfiguration config;
		config.op = CUDPP_ADD;
		config.datatype = CUDPP_INT;
		config.algorithm = CUDPP_SCAN;
		config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
		config.mystream = stream;

		CUDPPHandle theCudpp;
		cudppCreate(&theCudpp);
		CUDPPHandle scanplan = 0;
		CUDPPResult res = cudppPlan(theCudpp,&scanplan,config,size,1,0);
		if ( CUDPP_SUCCESS != res ){
			printf("ERROR in prepare_prefixsum\n");
			exit(-1);
		}
		return scanplan;
	}
	inline void move_pos(int &pos){
		pos = (pos + 1) % QUEUE_SIZE;
	}

struct SearchSetting{
	int D1,D2;
	int size(){
		return D1*D2;
	}
	void set(int D1,int D2){
		this->D1 = D1,this->D2 = D2;
	}
};

struct SearchSettingQueue{
	int parts;
	SearchSetting * settings;

	SearchSettingQueue(int parts){
		settings = new SearchSetting[parts];
	}
};

	void work(){

		int numStream;
		numStream = 4;
		cudaStream_t *streams = (cudaStream_t *) malloc(numStream * sizeof(cudaStream_t));
		for (int i = 0; i < numStream; i++)
		{
			cudaStreamCreate(&(streams[i]));
		}

		int D1,D2,D1save;
		int block_2_size;
		//------ some settings ----------
		D1 = 256;
		D1save = 8;
		D2 = 512;

		//D1 = 32;
		//D2 = 32;
		//D1save = 8;

		int save_stream = 2;

		block_size = D1 * D2;
		block_2_size = block_size;
		//block_2_size = block_size*1000;

		init_data();
		init_device_variables();
		CudaWatch cudawatch;
		Watch cpuWatch;cpuWatch.start();
		cudawatch.start();

		CUDPPHandle prefixsum_plan = prepare_prefixsum(block_size,&streams[save_stream]);
		//prefixsum_plan = prepare_prefixsum(block_size,0);

		init_scan(&streams[save_stream],1024);

		int len1,len2;
		int cal_pos = 0, search_pos = 0, save_pos  =0; // they are for L1, L2 position
		while (true){
			back_next_relative_len(len1,len2,cal_pos);
			cu_checkError();
			outln(len1);outln(len2);//debug
			if ( len1<=0 || len2 <= 0 ) break;
			int loops = min(len1,len2)/block_2_size;
			if ( loops >0 ){
				outln(loops);
				bool lastButOne = loops > 1;
				int *saveV; //pointer of saving result
				int devVinc = 0; // use which devV to store data

				//-- stage 1
				cal_indx<<<1,4,0,streams[0]>>>(block_size,block_2_size,cal_pos);
				move_pos( cal_pos );

				//-- stage 2
				if ( lastButOne ){
					cal_indx<<<1,4,0,streams[0]>>>(block_size,block_2_size,cal_pos);
					move_pos( cal_pos );
					algo2_search<<<D1,D2,0,streams[1]>>>( devV[devVinc],search_pos );
					move_pos( search_pos );
				}
				cudaDeviceSynchronize();

				//-- stage middle
				for ( loops -= 2; loops> 0 ;loops -- ){
					//outln(loops);
					saveV = devV[ devVinc];
					devVinc = 1- devVinc;
					//scan(saveV,block_size);
					scan2(saveV,block_size);
					save_result<<<D1save,D2,0,streams[save_stream]>>>(saveV,save_pos);/*---- save part -----*/
					algo2_search<<<D1,D2,0,streams[1]>>>( devV[devVinc],search_pos );
					//cudaDeviceSynchronize();
					cal_indx<<<1,4,0,streams[0]>>>(block_size,block_2_size,cal_pos);
					//cudppScan(prefixsum_plan,saveV,saveV,block_size);/*---- save part -----*/
					move_result_addr<<<1,1,0,streams[save_stream]>>>(saveV,save_pos);/*---- save part -----*/
					move_pos( cal_pos );move_pos( search_pos );move_pos( save_pos );
					cu_checkError();
				}
				cudaDeviceSynchronize();
				//-- stage last but one
				if (lastButOne){
					saveV = devV[ devVinc];
					//scan(saveV,block_size);/*---- save part -----*/
					cudppScan(prefixsum_plan,saveV,saveV,block_size);/*---- save part -----*/
					devVinc = 1- devVinc;
					save_result<<<D1save,D2,0,streams[save_stream]>>>(saveV,save_pos);/*---- save part -----*/
					move_result_addr<<<1,1,0,streams[save_stream]>>>(saveV,save_pos);/*---- save part -----*/
					move_pos( save_pos );
				}
				algo2_search<<<D1,D2,0,streams[1]>>>( devV[devVinc],search_pos );
				move_pos( search_pos );
				cudaDeviceSynchronize();
				//----- stage last --------
				saveV = devV[ devVinc];
				//scan(saveV,block_size);/*---- save part -----*/
				cudppScan(prefixsum_plan,saveV,saveV,block_size);/*---- save part -----*/
				devVinc = 1- devVinc;
				save_result<<<D1,D2,0,streams[save_stream]>>>(saveV,save_pos);/*---- save part -----*/
				move_result_addr<<<1,1,0,streams[save_stream]>>>(saveV,save_pos);/*---- save part -----*/
				move_pos( save_pos );
			}
			else {
				cout<<"small seg "<<len1<<" "<<len2<<endl;
				move_indices<<<1,1>>>(len1,len2,block_size,cal_pos);
				algo2_search<<<D1,D2>>>(devV[0],search_pos);
				cudppScan(prefixsum_plan,devV[0],devV[0],block_size);
				save_result<<<D1,D2>>>(devV[0],save_pos);
				move_result_addr<<<1,1>>>(devV[0],save_pos);
				move_pos(cal_pos);move_pos(search_pos);move_pos(save_pos);
			}
			//printf(" poss: %d %d %d\n",cal_pos,search_pos,save_pos);
		}
		cu_checkError();
		cudaDeviceSynchronize();
		cout<<"MY Algo:"<<cudawatch.stop()/1000.0<<endl;
		cout<<"MY Algo cpu test: "<<cpuWatch.stop()<<endl;
	}

	void merge_algo(int *array1,int *array2, int begin1,int end1,int begin2,int end2){
		//return ;
		int i=begin1,j=begin2;
		int lasti,lastj;
		cpuResultSize = 0;
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
	}

	int check_correctness(){
			FOR_I(0,cpuResultSize)
				if (cpuResult[i] != resultList[i]){
					printf("Wrong at [%d] R:%d Yours:%d\n",i,cpuResult[i],resultList[i]);
					debug_a( cpuResult+i,-10,10 );
					debug_a( resultList+i,-10,10 );
					return i;
				}
		return -1;
	}

	__global__ void algo_bsearch(
			int *list1,int *list2
			,int *mark
			,int n,int m){
		int i = CUID;
		int size = CUSIZE;
		while ( i < n){
			int value = list1[i];
			int left = 0,right = m-1;
			mark[i] = 0;
			while (left <= right){
				int mid = (left+right)/2;
				int difference = value-list2[mid];
				if ( difference <0 )
					right = mid-1;
				else if ( difference >0 )
					left = mid +1;
				else {
					mark[i] = 1;
					break;
				}
			}
			i += size;
		}
	}

	__global__ void bsearch_copy_result(int * list1,int *mark, int *result,int n){
		int id = CUID;
		int size = CUSIZE;
		while ( id < n ){
			if (mark[id] - mark[id-1] >0){
				result[ mark[id-1] ] = list1[ id ];
			}
			id += size;
		}
	}

	void work3(){
		init_data();
		cudaStream_t oneStream;
		cudaStreamCreate(&oneStream);
		//outln(n);
		CUDPPHandle prefixsum_plan = prepare_prefixsum(n+1,&oneStream);
		CudaWatch cudawatch,w2;cudawatch.start();
		int * markFrom = devMark+1;

		w2.start();
		algo_bsearch<<<32,512,0,oneStream>>>(devL1,devL2,markFrom,n,m);
		cout<<"naive bsearch: part1 :"<<w2.stop()/1000.0<<endl;

		cudppScan(prefixsum_plan,devMark,devMark,n+1);
		bsearch_copy_result<<<32,512,0,oneStream>>>(devL1,markFrom,devResult,n+1);
		cout<<"NAIVE Bsearch:"<<cudawatch.stop()/1000.0<<endl;
		cudaMemcpy(&Lresult,devMark+n,sizeof(int),D_T_H);cu_checkError();
		cudaMemcpy(resultList,devResult,sizeof(int)*Lresult,D_T_H);cu_checkError();
		free_device_memory();
	}

int main(){
	int r;
	FOR_I(103,10000){
	r = rand() % 10013123 ;
	//r =1344532745 ;
	srand(i);
	n = 1024*1024*40;
	//n = 1024*1024*5;
	//n = 50;

	//generate_case5();
	generate_random(1.0,2.0,2.0);

	cout<<"generate data over srand("<<i<<") n="<<n<<" m="<<m<<endl;
	printf("List 1 ( %d --- %d --- %d )\n",list[0][0],list[0][n/2],list[0][n-1]);
	printf("List 2 ( %d --- %d --- %d )\n",list[1][0],list[1][m/2],list[1][m-1]);

//		debug_a(list[0],n);outline;debug_a(list[1],m);

	memset(V,0,sizeof(V));

	Watch watch;watch.start();
	merge_algo(list[0],list[1],0,n,0,m);
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
	if  ( (error_index = check_correctness()) >=0 ){
		outline;
		outln(i);
		printf("all length: cpu:%d gpu:%d  ration: %.5lf\n",cpuResultSize,Lresult,100.0*i/(1.0*Lresult));
		break;
	}
		outline;
		outline;
		outline;
		outline;

		break; //only execute once
	}//end FOR
	return 0;
}
