/*New feature:
	pre calculate indices is employed
	adaptive search is tried.( not pipeline yet )
	foolish adaptive
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
//#include <memory.h>

#define PI acos(-1)
#define eps 1e-9

#define out(x) (cout<<#x<<":"<<x<<" ")
#define outln(x) (cout<<#x<<":"<<x<<endl)
#define outs(x) (cout<<x)
#define outline (cout<<endl)
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

#define BANK_SIZE 32
#define BANK_SIZE_LOG 5

#define SHARE_OFFSET(id) ( (id) >> (BANK_SIZE_LOG))
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


const int N = 54100100;
const int M = 1024*1024;

int list[2][N];
int *devL1,*devL2;

bool v[N*10];

int n,m;

int P[M],V[M],V2[M];
int resultList[N];
int cpuResult[N],cpuResultSize;

int *devP,*devV,*devResult;

int L2index,L1index,Lresult;
int *devL2index,*devL1index,*devResultIndex;

int block_size;

const int QUEUE_SIZE= 5;

__device__ int *list_p0[2];
__device__ int *list_p[QUEUE_SIZE][2]; // save the starting position of either list
__device__ int calculated_length[QUEUE_SIZE];
__device__ int swapped[QUEUE_SIZE];   // save swapped stage for each status
__device__ int queue_size = QUEUE_SIZE;
__device__ int indices_now=0, search_now=0, prefix_now=0;
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
		int temp1[] = {1,7,8,9,12};
		int temp2[] = {3,5,7,9,11};
		n = m = 5;
		FOR_I(0,n) list[0][i] = temp1[i];
		FOR_I(0,n) list[1][i] = temp2[i];
	}

	void generate_case3(){
		int temp1[] = {1,7,8,9,12};
		int temp2[] = {13,15,17,19,21};
		n = m = 5;
			FOR_I(0,n) list[0][i] = temp1[i];
			FOR_I(0,n) list[1][i] = temp2[i];
	}

	void generate_case4(){
		int temp1[] = {1,7,8,11,12,  25,29,30,55,90};
		int temp2[] = {13,15,17,19,21, 22,25,30,33,34};
		n = m = 10;
		FOR_I(0,n) list[0][i] = temp1[i];
		FOR_I(0,n) list[1][i] = temp2[i];
	}

	void generate_case5(){
		int temp1[] = {1,7,8,19,20, 21,29,30,35,40};
		int temp2[] = {13,15,17,19,21, 24,25,30,33,40};
		n = m = 10;
		FOR_I(0,n) list[0][i] = temp1[i];
		FOR_I(0,n) list[1][i] = temp2[i];
	}

	void generate_case6(){
		int temp1[] = {13,15,17,19,21, 24,25,30,33,40};
		int temp2[] = {71,77,78,79,80, 91,92,93,94,95};
		n = m = 10;
		FOR_I(0,n) list[0][i] = temp1[i];
		FOR_I(0,n) list[1][i] = temp2[i];
	}

	void init_data(){
		int sizeV = 2*block_size + 32;
		int *temp = new int [sizeV];
		FOR_I(0,sizeV) temp[i] = 0;

		cudaMalloc( (void **)&devL1, sizeof(int)*n );
		cudaMalloc( (void **)&devL2, sizeof(int)*m );
		cudaMalloc( (void **)&devV, sizeof(int)*sizeV );
		cudaMalloc( (void **)&devP, sizeof(int)*block_size );
		cudaMalloc( (void **)&devResult, sizeof(int)*n );

		cudaMalloc( (void **)&devL2index, sizeof(int) );
		cudaMalloc( (void **)&devL1index, sizeof(int) );
		cudaMalloc( (void **)&devResultIndex, sizeof(int) );

		cudaMemcpy( devL1,list[0],sizeof(int)*n,H_T_D );
		cudaMemcpy( devL2,list[1],sizeof(int)*m,H_T_D );
		cudaMemcpy( devV,temp,sizeof(int)*sizeV,H_T_D );
		cudaMemcpy( devP,temp,sizeof(int)*block_size,H_T_D );
		devV = devV + 32 + block_size;
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
		Lresult = (int)(end - devResult );
		outln(Lresult);
		cudaMemcpy( resultList,devResult, sizeof(int)*Lresult,D_T_H );
	}

	__device__ int next_queue_pos(int value){
			return (value+1) % queue_size;
	}

//----------------- prefix sum part ------------------
	__global__ void prefix_sum_cuda(int *list){
		int id = threadIdx.x;
		int block_size = blockDim.x << 1;
		int offset = blockIdx.x * block_size;
		list += offset;

		share[ 2*id + SHARE_OFFSET(2*id) ] = list[ 2*id ];
		share[ 2*id + 1 + SHARE_OFFSET(2*id+1) ] = list[ 2*id+1 ];
		//printf("%d --> %d\n",2*id,2*id + SHARE_OFFSET(2*id));

		//reduce stage
		int len = 1;
		int threads = 1;
		for (threads = blockDim.x ; threads >0  ;len <<=1,threads >>= 1 ){
			syncthreads();
			if ( id < threads ){
				int from = len -1 + 2*len*id;
				int addTo =  from + len;
				from += SHARE_OFFSET(from);
				addTo += SHARE_OFFSET(addTo);
				//list[ addTo ] += list[from];
				share[addTo] += share[from];
				//printf("id %d to[%d] %d\n",id,addTo,share[addTo]);
			}
		}
		//map stage
		len >>= 2;
		for ( threads = 2;len >=1 ;len >>=1,threads <<= 1 ){
			syncthreads();
			if ( id < threads -1 ){
				int from = 2*len -1 + 2*len*id;
				int addTo = from + len;
				from += SHARE_OFFSET(from);
				addTo += SHARE_OFFSET(addTo);
				//list[ addTo ] += list[from];
				share[addTo] += share[from];
			}
		}

			syncthreads();
		list[ 2*id ] = share[ 2*id + SHARE_OFFSET(2*id) ] ;
		list[ 2*id+1 ] = share[ 2*id + 1 + SHARE_OFFSET(2*id+1) ] ;
	}


	//add the value to all elements in list
	__global__ void addup(int *list,int *valueList){
		int id = threadIdx.x;
		int block_size = blockDim.x << 1;
		int offset = (blockIdx.x+1) * block_size;
		list += offset;
		// the length of list should be double of CUSIZE

		int value = *(valueList + blockIdx.x);
		int temp = id << 1;
		list[ temp ] += value;
		list[ temp+1 ] += value;
	}

	inline int block_multiple(int value,int block_size){
		if ( value % block_size != 0 )
			return block_size - value % block_size + value;
		return value;
	}

	//do prefix sum for inputed list
	void prefix_sum_algo(int *devList,int size, int block_size,int deep){
		int num_blocks = size / block_size;
		prefix_sum_cuda<<< num_blocks ,block_size/2>>>( devList );
		if ( num_blocks <=1 ) return;
		int size2 = block_multiple(num_blocks,block_size);
		//prepare data stage
			//ATTENTION this part is sequential part
		int i=0;
		int *nextList = devList - size2 - 1;
		for (i=0;i < num_blocks ;i++){
			int *dst = nextList + i;/// NOTATION from the end and a zero between
			int *src = devList + block_size*i + block_size - 1;
			cudaMemcpy( dst, src, sizeof(int) , D_T_D );//only copy last element
		}
		for (;i < size2 ;i++){
			int *dst = nextList + i;
			int temp = 0;
			cudaMemcpy(dst,&temp,sizeof(int),H_T_D);
		}
		//cal higher level block
		prefix_sum_algo(nextList,size2,block_size,deep+1);
		addup<<< num_blocks-1 ,block_size/2>>>(devList,nextList);
	}

	__global__ void prefix_sum_oneCore(int *V,int block_size){
		FOR_I(1,block_size)
			V[i] += V[i-1];
	}

//------------------ prefix sum over ---------------

	//binary search lower bound in list2 for each element in list1
	//algo2 should not care about swap or not
	__global__ void algo2_search(int * V){
		bool swap_now = swapped[indices_now];
		swap_now = false;
		bool swap_next = swapped[next_queue_pos(indices_now)];
		int *list1 = list_p[ indices_now ][ swap_now ];
		int *list2 = list_p[ indices_now ][ !swap_now ];
		//int *list2_end = list_p[ next_queue_pos(indices_now) ][ !swap_next ];

		int id = CUID;
		//if (!id) printf("now %d\n",indices_now);

		int block_size = CUSIZE;
		int left = 0;
		int right = calculated_length[indices_now] ;
		int value = list1[ id ];

			//if (!id)
		//printf("search %d in (%d <-- %d --> %d)\n",list1[id],list2[0],right+1,list2[right]);//debug
		V[id] = 0;
		while ( left <= right ){
			int mid = (left + right)/2;
			if ( value == list2[ mid ] ){
				V[ id ] = 1;
				return ;
			}
			else if ( list2[mid] < value )
				left = mid + 1;
			else right = mid - 1;
		}

		//printf("id: %d  %d get [%d] %d\n",id, value,left,list2[left]);
	}

	__global__ void save_result(int *V){
		int block_size = CUSIZE;
		int id = CUID;
		int *list1 = list_p[indices_now][0];

		if (V[id] - V[id-1] >0){
			_result[ V[id-1] ] = list1[ id ];
		}
	}

	__global__ void move_result(int block_size,int * V){
		//muse wait for others
		int id = CUID;
		if ( id == 0){
			_result += V[  block_size - 1 ];
		}
	}

	__global__ void init_device_var_kernel(){
		int id = CUID;
		if (id) return;
		list_p[0][0] = list_p0[0];
		list_p[0][1] = list_p0[1];
		swapped[0] = 0;//false
		ratio_avg = 0.0;//for debug
		cal_times = 0;//for debug
		indices_now = search_now = prefix_now= -1;
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


	//calculate the start point of next block ( pointer might overflow )
	// two threads calculate and write opposite lowerbound (each other)
	// id my index !id opposite index
	__global__ void calculate_indices(int block_size,int block_2_size,int isSwap){

		int id = CUID;
		if ( id > 1 ) return;

		int opposite = !id;

		if (0 == id)
			indices_now = (indices_now + 1) % queue_size;

		syncthreads();

		int temp_arr[] = {block_2_size,block_size};
		int left = 0;
		int right = temp_arr[id] - 1;

		int *list2 = list_p[indices_now][ opposite ];
		int value = list_p[indices_now][ id ][ temp_arr[opposite] -1];

		//printf("CI: id %d| [ %d ] %d search (%d %d)\n",id, (int)(list_p[indices_now][0] - list_p0[0]) , value,list2[left],list2[right]);//debug

		while ( left < right ){
			int mid = (left + right + 1)/2;
			if ( value < list_p[indices_now][ opposite ][mid] )
				right = mid - 1;
			else left = mid;
		}

		//printf("CI: id %d|   %d search to [%d] %d\n",id,value,left,list2[left]);//debug
		int next = next_queue_pos(indices_now);
		//list_p[next][ (0+isSwap) %2 ] = list_p[indices_now][0]+block_size;

		int next_opposite_index = (opposite)^isSwap;

		int next_offset = left+ ( list_p[indices_now][ opposite ][left] <= value );
		//Attention <= is the lower bound condition for edge condition
		list_p[next][ next_opposite_index ] = list_p[indices_now][ opposite ] + next_offset;

		if (0 == id){
			calculated_length[ indices_now ] = next_offset - 1;
			swapped[next] = swapped[indices_now] ^ isSwap;
		}
	}

	__global__ void helpIndex(int *indices,bool isNext){
		int temp = indices_now;
		if (isNext) temp = next_queue_pos(temp);
		indices[0] = list_p[temp][swapped[temp] ]-list_p0[0];
		indices[1] = list_p[temp][!swapped[temp] ]-list_p0[1];
	}

	__global__ void helpIndex_relative(int *indices,bool isNext){
		int temp = indices_now;
		if (isNext) temp = next_queue_pos(temp);
		indices[0] = list_p[temp][ 0 ]-list_p0[0];
		indices[1] = list_p[temp][ 1 ]-list_p0[1];
	}

	void back_next_relative_indices(int & index1,int & index2){
		int temp[2];
		helpIndex_relative<<<1,1>>>(devIndices,true);
		cudaMemcpy(temp,devIndices,sizeof(int)*2, D_T_H);
		index1 = temp[0];index2 = temp[1];
	}

	// configure : <<<1,2>>> only and but only 2 threads
	__global__ void helpLen_relative(int *memory){
			int temp = next_queue_pos(indices_now);
			int id = threadIdx.x;
			int isSwapped = swapped[ temp ];
			int myindex = id ^ isSwapped;// id:1  !swapped

			memory[id] = (int)(list_p[temp][ id ]-list_p0[ myindex ]);
			memory[id] = _nm[myindex] - memory[id];

		}

	void back_next_relative_len(int & len1,int & len2){
		int temp[2];
		helpLen_relative<<<1,2>>>(devIndices);
		cudaMemcpy(temp,devIndices,sizeof(int)*2, D_T_H);
		len1 = temp[0];len2 = temp[1];
	}

	void back_next_indices(int & index1,int & index2){
		int temp[2];
		helpIndex<<<1,1>>>(devIndices,true);
		cudaMemcpy(temp,devIndices,sizeof(int)*2, D_T_H);
		index1 = temp[0];index2 = temp[1];
	}

	void back_indices(int & index1,int & index2){
		int temp[2];
		helpIndex<<<1,1>>>(devIndices,false);
		cudaMemcpy(temp,devIndices,sizeof(int)*2, D_T_H);
		index1 = temp[0];index2 = temp[1];
	}

	void work(){
		int D1,D2,Dprefix;// Dprefix is the cuda dimension of prefix sum
		int block_2_size;

		//------ some settings ----------
		D1 = 64*4;
		D2 = 64*2;
		Dprefix = 4;

		block_size = D1 * D2;
		//block_size is only the length of looking list
		block_2_size = block_size*4;
		int block_sum = block_size + block_2_size;

		int switch_mode = 1;// foolish adaptive mode

		//------ END some settings --------
//		list[0][n++] = 1 <<10;list[1][m++] = 1<< 10;

		init_data();
		init_device_variables();

		L1index = L2index = 0;
//		n--;m--;

		int times=0;
		CudaWatch cudawatch;
		cudawatch.start();

		bool DEB_VP = false;

		int len1 = n,len2 = m;
		while (true){

			back_next_relative_len(len1,len2);
			//outln(len1);outln(len2);//debug
			if ( len1<=0 || len2 <= 0 ) break;

			int loops = 2*(min(len1,len2)-block_2_size)/block_sum;

			if ( loops >0 ){
				outln(loops);//debug
				FOR_I(0,loops){
					calculate_indices<<<1,2>>>(block_size,block_2_size,switch_mode);
					algo2_search<<<D1,D2>>>(devV);
					//prefix_sum_oneCore<<<1,1>>>(devV,block_size);
					//save_result<<<D1,D2>>>(devV);
					//move_result<<<1,1>>>(block_size,devV);

					//back_next_indices(L1index,L2index);//debug
					//printf("Next [%d] %d --> [%d] %d\n",L1index,list[0][L1index],L2index,list[1][L2index]);

					//back_indices(L1index,L2index);//debug
					//show_lists(L1index,L2index,10,10);//debug
					//getchar();//debug
					//TODO prefix sum
				}//end for
			}
			else {

				if ( len1 < 1024 ) {
					cout<<"Small segment:";out(len1);outln(len2);out(L1index);outln(L2index);
					calculate_indices<<<1,2>>>(len1,len2,switch_mode);
					algo2_search<<<1,len1>>>(devV);
					//prefix_sum_oneCore<<<1,1>>>(devV,len1);
					//save_result<<<1,len1>>>(devV);
					//move_result<<<1,1>>>(len1,devV);


				}
				else {
					//cout<<"go to 1024"<<endl;
					len2 = min(block_2_size,len2);
					calculate_indices<<<1,2>>>(32*32,len2,switch_mode);
					algo2_search<<<32,32>>>(devV);
					//prefix_sum_oneCore<<<1,1>>>(devV,32*32);
					//save_result<<<32,32>>>(devV);
					//move_result<<<1,1>>>(32*32,devV);

				}
			}

		}
		outln(cudawatch.stop()/1000.0);
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
				//V2[i] = 1;
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
					double x = (double)i / (double)cpuResultSize;
					printf("Wrong at [%d] R:%d Yours:%d   len:%lf\n",i,cpuResult[i],resultList[i],x);

					FOR_J(-10,10)
					printf("(%d,%d)",cpuResult[i+j],resultList[i+j]);

					return i;
				}
		return -1;
	}

	__global__ void algo_bsearch(
			int *list1,int *list2
			,int *V
			,int n,int m){
		int i = threadIdx.x + blockIdx.x*blockDim.x;
		while ( i < n){
			int value = list1[i];
			int left = 0,right = m-1;
			while (left <= right){
				int mid = (left+right)/2;
				int difference = value-list2[mid];
				if ( difference <0 )
					right = mid-1;
				else if ( difference >0 )
					left = mid +1;
				else {
					V[i] = 8;break;
				}
			}
			i += blockDim.x * gridDim.x;
		}
	}

	void work3(){
		init_data();
		CudaWatch cudawatch;
		cudawatch.start();
		//best achieved at 128,128 64, 32, 16
		// which is strange for 16 !
		algo_bsearch<<<32,512>>>(devL1,devL2,devResult,n,m);
		cout<<"NAIVE Bsearch:"<<cudawatch.stop()<<endl;
	}


	void test_indices(){
		int block_size = 64;
		int block_2_size = 128;
		n = 1024*1024;
		srand(45);
		generate_random();
		init_data();
		init_device_variables();

		//show_lists(50,50,70,70);

		int i;
		int t1,t2;
		for ( i=1;;i++ ){
			calculate_indices<<<1,1>>>(block_size,block_2_size,true);
			back_indices(t1,t2);
			printf("%d %d | %d %d\n",t1,t2, t1-i*block_size,t2-i*block_size);
			if ( n - max(t1,t2) < block_2_size )
				break;
		}
		outln(t1);outln(t2);
		outln(i);//debug

		cudaDeviceSynchronize();
		double temp;
		int times;
		//cudaMemcpyFromSymbol( &temp, "ratio_avg", sizeof(double),0,D_T_H );
		cudaDeviceSynchronize();
		//cudaMemcpyFromSymbol( &times, "cal_times", sizeof(int),0,D_T_H );
		cu_checkError();
		//printf(" %.1lf / %d = %lf\n",temp,times,temp/(1.0*times));
	}

	void test_prefix_sum(){
		n = 1024*2;
		generate_random();
		int m = n*2 +32;
		cudaMalloc((void **)& devL1, sizeof(int)*m);
		cudaMemcpy( devL1+n+32 ,list[0],sizeof(int)*n,H_T_D);
		cudaMemcpy( devL1,list[0]+n,sizeof(int)*n,H_T_D);
		prefix_sum_algo(devL1+n+32,n,2,0);
		cu_checkError();
		cudaDeviceSynchronize();
		cu_host_print(devL1,2*n);//debug
		cudaDeviceSynchronize();
		cudaMemcpy( list[1],devL1+n+32,sizeof(int)*n,D_T_H);
		FOR_I(1,n) {
			list[0][i] += list[0][i-1];
			if (list[0][i] != list[1][i]){
				cout<<"ERROR at"<<i<<endl;
				exit(123);
			}
		}

		cu_checkError();
		exit(0);
	}

int main(){
	//test_indices();return 0;
	//test_prefix_sum();

	//test_bsearch();return 0;

	int r;
	FOR_I(6,10000){
	r = rand() % 10013123 ;
	//r =1344532745 ;
	srand(i);
	n = 1024*1024*20;
	//n = 1024*16;

	//generate_case2();
	generate_random(2.0,2.0,1.0);


	cout<<"generate data over srand("<<i<<") n="<<n<<" m="<<m<<endl;
	printf("List 1 ( %d --- %d --- %d )\n",list[0][0],list[0][n/2],list[0][n-1]);
	printf("List 2 ( %d --- %d --- %d )\n",list[1][0],list[1][m/2],list[1][m-1]);

		//debug_a(list[0],n);
		//debug_a(list[1],n);

	memset(V2,0,sizeof(V2));
	memset(V,0,sizeof(V));

	Watch watch;watch.start();
	merge_algo(list[0],list[1],0,n,0,m);
	cout<<"CPU ALGO time: "<<watch.stop()<<endl;

	work();
	cu_checkError();

	work3();//test bsearch

	//cuda_copyResult();
	//outln(Lresult);outln(cpuResultSize);
	//debug_a(resultList,Lresult);
	cout<<"copied back"<<endl;
	//debug_a(cpuResult,10);
	//break;

	continue;

	int error_index;
	if  ( (error_index = check_correctness()) >=0 ){
		outline;
		outln(i);
		break;
	}
		outline;
		outline;
		outline;
		outline;

		//break; //debug
	}//end FOR
	return 0;
}
