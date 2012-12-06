/*New feature:
	different block_size for each list is employed.
	prefix sum algorithm is employed

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


const int N = 50000100;
const int M = 1024*1024;

int list[2][N];
int *devL1,*devL2;
int length[2];

bool v[N*10];

int n,m;

int P[M],V[M],V2[M];
int resultList[N];
int cpuResult[N],cpuResultSize;

int *devP,*devV,*devResult;

int L2index,L1index,Lresult;
int *devL2index,*devL1index,*devResultIndex;

int block_size;


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

	void generate_random(){
		int mod = n*7;
		generate_different(list[0],n,mod);
		sort(list[0],list[0]+n);
		generate_different(list[1],n,mod);
		sort(list[1],list[1]+n);
		m = n;
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

	void testCuda(){
		cudaMemcpy( list[1],devL1,sizeof(int)*n,D_T_H);
		FOR_I(0,n)
			if (list[0][i] != list[1][i]){
				cout<<"CUDA ERROR"<<endl;
				exit(0);
			}
		cout<<"CUDA OK"<<endl;
		exit(0);
	}


	void cuda_copyback(){
		cudaMemcpy( V,devV,sizeof(int)*block_size,D_T_H );
		cudaMemcpy( P,devP,sizeof(int)*block_size,D_T_H );
	}

	void cuda_copyResult(){

		cudaMemcpy( &Lresult,devResultIndex,sizeof(int),D_T_H );
		cudaMemcpy( resultList,devResult, sizeof(int)*Lresult,D_T_H );
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

//------------------ prefix sum over ---------------


	/*find k where  L2[k]<= L1[i] < L2[k+1] for each i P[i] = k
		if  L2[k] == L1[i] then V[i] = 8;
		if  L2[k] < L1[i] then V[i] = 4;

		--- rull 3 and 4 ---
		if  any k L1[i] < L2[k] (no L2[k] <= L1[i])  Left edge
			then V[i] = 2;
		if  any k L2[k] < L1[i]  (no L1[i] < L2[k+1]) Right edge
			then V[i] = 16;

		---- because L2[k] < L2[k+1] ----
		L1[i] < L2[left] Left edge
		L2[i] > L2[right]   Right edge
	*/

	//binary search lower bound in list2 for each element in list1
	__global__ void algo2_search(int * list1, int * list2,
					int *index1,int * index2,
					int m,
					int *P,int *V,
					int block_2_size){
		list1 += *index1;
		list2 += *index2;
		//V += *index1;
		int i = threadIdx.x + blockIdx.x*blockDim.x;
		int block_size = blockDim.x * gridDim.x;
		int endIdx = block_size - 1;
		int left = 0;
		int right_edge = block_2_size - 1;
		int right = right_edge ;
		int value = list1[ i + left];
		while ( left < right ){
			int mid = (left + right + 1)/2;
			if ( value < list2[mid] )
				right = mid - 1;
			else left = mid;
		}


		P[i] = left ;
		if ( value == list2[left] ){
			V[i] = 8;
		}
		else if ( value < list2[left] ){
			//no lower bound
			V[i] = 2;P[i] = left -1;
		}
		else if ( value > list2[right_edge]){
			//bigger than all variables
			V[i] = 16;
		}
		else V[i] = 4;

		//decide index2
		if (i == endIdx){
			*index2 += P[i]+1;
		}
	}

	__global__ void prepare_prefix(int *V){
		int id = CUID;
		if (V[id] == 8){
			V[id] = 1;
		}
		else V[id] = 0;
	}

	__global__ void save_result(int *list1,int index1,int *result,int *index_r,int *V){
		int block_size = CUSIZE;
		int id = CUID;
		list1 += index1;
		result += (*index_r) - 1;
		if (V[id] - V[id-1] >0){
			result[ V[id] ] = list1[ id ];
		}
		syncthreads();
		//muse wait for others
		if ( id == block_size - 1 ){
			*index_r += V[id];
		}
	}

	//decide index1
	__global__ void cal_index(int *index1,int *index2,
				int *P,int *V){
		//V += *index1;
		int block_size = blockDim.x * gridDim.x;
		int i = threadIdx.x + blockIdx.x*blockDim.x;
		if ( i == block_size - 1 ){
			if (V[i] != 16 ){
				*index1 += block_size;
			}
		}
		else if ( V[i+1] - V[i] >=8 ){
			*index1 += i+1;
		}
	}

	__global__ void copyto(int * V,int *l){
		V[threadIdx.x] = l[threadIdx.x];
	}

	__global__ void cuda_index(int * devIndex,int index){
		*devIndex = index;
	}

	void back_index(){
		cudaMemcpy(&L2index,devL2index,sizeof(int),D_T_H);
		cudaMemcpy(&L1index,devL1index,sizeof(int),D_T_H);
	}

	__global__ void test(int *testv){
		*testv += 1;
	}

	void show_lists(int index1,int index2,int len1,int len2){
		printf("Two lists:\n");
		printf("L1 :\t");debug_a(list[0]+index1,len1);
		printf("L2 :\t");debug_a(list[1]+index2,len2);
		outline;
	}

	void show_vp(int len){
		cuda_copyback();
		cout<<"P\t";debug_a(P,len);
		cout<<"V\t";debug_a(V,len);
	}

	void show_vp(){
		show_vp(block_size);
	}

	void work(){
		int D1,D2,Dpre;// Dpre is for prefix sum
		int block_2_size;
		D1 = 64*2;
		D2 = 64*2;
		Dpre = 1024;

		block_size = D1 * D2;
		//block_size is only the length of looking list
		block_2_size = D1*D2*2;
		int block_sum = block_size + block_2_size;
		block_sum = block_size;

		outln(block_2_size);
		init_data();

		L1index = L2index = 0;
		cuda_index<<<1,1>>>(devL1index,0);
		cuda_index<<<1,1>>>(devL2index,0);
		cuda_index<<<1,1>>>(devResultIndex,0);

		int times=0;
		CudaWatch cudawatch;
		cudawatch.start();
		Watch watch;
		watch.start();

		bool DEB_VP = false;

		while (true){
			int len1,len2;
			len1 = n - L1index;
			len2 = m - L2index;

			//show_lists(L1index,L2index,len1,len2);//debug
			//out(len1);outln(len2);getchar();
			if ( len1<=0 || len2 <= 0 ) break;
			else if ( len1 < 1024 && ( len1 < block_sum || len2 < block_2_size) ) {
				cout<<"Small segment:";out(len1);outln(len2);out(L1index);outln(L2index);
				//getchar();
				len1 = min(block_size,len1);
				len2 = min(block_2_size,len2);
				//show_lists(L1index,L2index,len1,len2);//debug
				//algo1_each<<<len1,len2>>>(
				algo2_search<<<len1,1>>>(
					devL1,devL2,
					devL1index,devL2index,m,
					devP,devV,len2 );

				back_index();
				cal_index<<<1,len1>>>(devL1index,devL2index,devP,devV);
				int prefix_len = block_multiple(len1,Dpre);
				cudaMemset(devV+len1,0,sizeof(int)*(prefix_len - len1));
				prepare_prefix<<<D1,D2>>>(devV);
				prefix_sum_algo(devV,block_size,Dpre,0);
				save_result<<<D1,D2>>>(devL1,L1index,devResult,devResultIndex,devV );


				if (DEB_VP){
					show_vp(len1);//debug
					back_index();
					out(L1index);outln(L2index);
					getchar();
				}



			}
			else if (len1 < block_sum || len2 < block_2_size){
				cout<<"go to 1024"<<endl;
				len2 = min(block_2_size,len2);
				algo2_search<<<32,32>>>(
					devL1,devL2,
					devL1index,devL2index,m,
					devP,devV,len2 );

				back_index();
				cal_index<<<32,32>>>(devL1index,devL2index,devP,devV);
				int prefix_len = block_multiple(len1,Dpre);
				prepare_prefix<<<32,32>>>(devV);
				prefix_sum_algo(devV,block_size,Dpre,0);
				save_result<<<32,32>>>(devL1,L1index,devResult,devResultIndex,devV );


			}
			else {
				int loops = min(len1/block_sum,len2/block_2_size);
				outln(loops);//debug
				if ( loops == 0) getchar();
				FOR_I(0,loops){
					//algo1_each<<<block_size,block_size>>>(
					algo2_search<<<D1,D2>>>(
						devL1,devL2,
						devL1index,devL2index,m,
						devP,devV,block_2_size);
					//cu_host_print(devV,block_size);

					back_index();
					cal_index<<<D1,D2>>>(devL1index,devL2index,devP,devV);

					prepare_prefix<<<D1,D2>>>(devV);
					prefix_sum_algo(devV,block_size,Dpre,0);

					save_result<<<D1,D2>>>(devL1,L1index,devResult,devResultIndex,devV );

					//getchar();

					if (DEB_VP){
						show_vp();//debug
						back_index();
						out(L1index);outln(L2index);
						getchar();
					}
					//back_index();//will be slow two times
				}//end for
			}//end else
			back_index();
			//cout<<"("<<L1index<<","<<L2index<<")"<<endl;//debug
			//printf("L1[%d] : %d   L2[%d] : %d\n",L1index,list[0][L1index],L2index,list[1][L2index]);

			//outline;
			//outln(times++);
		}
		outln(cudawatch.stop());
		outln(watch.stop());
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

	__global__ void GPU_merge(int *array1,int *array2,int *arrayV,int end1,int end2){
		int i=0,j=0;
		int lasti,lastj;
		lasti=array1[i];
		lastj=array2[j];
		while ( i<end1 && j<end2){
			if ( lasti == lastj ){
				arrayV[i] = 8;
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
	void work2(){
		init_data();
		Watch watch;watch.start();
		GPU_merge<<<1,1>>>(devL1,devL2,devV,n,m);
		outln(watch.stop());
	}

	int check_correctness(){
			FOR_I(0,cpuResultSize)
				if (cpuResult[i] != resultList[i]){
					printf("Wrong at [%d] R:%d Yours:%d\n",i,cpuResult[i],resultList[i]);
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
			//syncthreads();
			i += blockDim.x * gridDim.x;
		}
	}

	void work3(){
		init_data();
		Watch watch;CudaWatch cudawatch;
		watch.start();cudawatch.start();
		//best achieved at 128,128 64, 32, 16 
		// which is strange for 16 !
		algo_bsearch<<<32,512>>>(devL1,devL2,devV,n,m);
		outln(cudawatch.stop());
		outln(watch.stop());
	}

	void test_bsearch(){
		int value = 57;
		int left = 0;
		int right = 1;
		int list2[] = {16,16,30 };
		while ( left < right ){
			int mid = (left + right + 1)/2;
			outln(mid);
			if ( value < list2[mid] )
				right = mid - 1;
			else left = mid;
		}
		outln(left);
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
	//test_prefix_sum();

	//test_bsearch();return 0;

	int r;
	FOR_I(41,1000){
	r = rand() % 10013123 ;
	//r =1344532745 ;
	srand(i);
	n = 1024*1024*50;
	//n = 1000*1000*10;
	//n = 1000001;
	//n = 3412312;
	//n = 1024*50;
	//generate_case2();
	generate_random();
	cout<<"generate data over srand("<<i<<") n="<<n<<endl;

		//debug_a(list[0],n);
		//debug_a(list[1],n);

	memset(V2,0,sizeof(V2));
	memset(V,0,sizeof(V));

	Watch watch;watch.start();
	merge_algo(list[0],list[1],0,n,0,m);
	outln(watch.stop());

	work();
	cu_checkError();

	//work2();//test single merge
	//work3();//test bsearch
	//cuda_copyback();
	cuda_copyResult();
	outln(Lresult);outln(cpuResultSize);
	//debug_a(resultList,Lresult);
	cout<<"copied back"<<endl;
	//break;

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

