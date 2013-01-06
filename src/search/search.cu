#include "../common_defines.h"
#include "search.cuh"
#include <cstdio>


#if DEPLETED

	//binary search lower bound in list2 for each element in list1
	//algo2 should not care about swap or not


	__device__ void search_1(int * V,int search_now,int offset){
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

	//Too many command
	__global__ void gallop_search_stage2(int *devL1,int *devL,int n){
		int id = threadIdx.x;
		int value = devL1[id];
		int pos = id;
		int left = 0,right = n;
		int bound;
		int result = 0;
		int len;
		//printf("id:%d get value:%d\n",id,value);
		int value2;

		int debugid = 20;
		int add_to_len;

		bool direction;
		//decide direction
		if ( devL[pos] <= value ) // right -->> direction
			direction = true,bound = right,len=1,add_to_len=0;
		else direction = false,bound = left,len=-1,add_to_len=1;
		printf(" (%d) value: %d  direction:%d  len:%d\n",id,value,direction,len);

		//len increase only one direction
		int pos2;
		while ( true ){
			value2 = devL[pos];
			if (id == debugid) printf(" compared with [%d]:%d,it move to %d for next time %d\n",pos,value2,pos,len);
			if ( value2 == value ) break;
			else if ( direction == ( value2 < value ) ){
				pos2 = pos + len;
			}
			else {
				len = (len+add_to_len)>>1;
				break;}
			if (!( left <= pos2 && pos2 <=right )) break;
			pos = pos2;
			len <<= 1;
		}

		if (!id) printf(" stage 1 over\n");
		printf(" (%d) value: %d  pos:%d  len:%d\n",id,value,pos,len);

		//len = 1;
		//len decrease, possible two directions
		while ( len!=0 ){
			len = (len+add_to_len)>>1;
			if (id == debugid) printf(" compared with %d,it move to %d for next time %d\n",value2,pos,len);
			if ( value2 == value ){
				result = 1;
				//add a break?
				break;
			}
			else if ( direction == (value2 < value )){
				pos += len;
			}
			else pos -= len;

			value2 = devL[pos];
		}

		if ( result )
			printf("(%d)  found %d at [%d]\n",id,value,pos);
		syncthreads();
		devL1[id] = result;
	}

	void test_gallop_search(){
		n = 1001;
		generate_random(1.0,1,1);
		//FOR_I(4,n)
//		host_lists[0][i] *=10;
		//host_lists[0][1] = 33;
		//host_lists[0][0] = 17;
		//generate_same(2);
		//generate_case3();
		debug_a(host_lists[0],n);outline;debug_a(host_lists[1],m);//debug
		init_data(128);
		init_device_variables();
		cudaDeviceSynchronize();
		HERE;
		//n = 32;

		int * cpuresult = new int[n];
		FOR_I(0,n){
			cpuresult[i] = 0;
			FOR_J(0,n)
			if ( host_lists[0][i] == host_lists[1][j] ){
				cpuresult[i] = 1;
				break;
			}
		}
		gallop_search<<<1,n>>>(devL1,devL2,n-1);
		cudaMemcpy(resultList,devL1,sizeof(int)*n,D_T_H);

		cudaDeviceSynchronize();

		debug_a(resultList,n);
		debug_a(cpuresult,n);

		cu_checkError();

		FOR_I(0,n)
			if ( resultList[i] != cpuresult[i] ){
				printf("WRONG AT [%d]:%d\n",i,host_lists[0][i]);
				exit(0);
			}
	}

#endif


	/**
	 *
	 * To write the program use minimum command
	 * Three things to decide the next position:
	 * 1. position? out of boundary?
	 * 2. get value, direction? change or not?
	 * 3. length? how l
	 * L: position = what
	 *
	 *
	 * value: the value to search
	 * pos2: starting place, extrapolated position, it need to be return!
	 * return:result 0/1
	 *
	 */

	__device__ int gallop_search (int value,int & pos2,int *list,int left,int right){

		int pos = pos2;
		int result = 0;
		int len;
		int value2;

		char last_direction = 3;
		char direction = 0;
		char start_decrease = 0;

		len = 1;
		while ( len > 0){

			if (!( left <= pos2 && pos2 <= right )){
				start_decrease = 1;
			}
			else{
				value2 = list[pos2] - value;
				if ( 0 == value2 ){
					result = 1;break;
				}
				else direction = value2 < 0;
				start_decrease |= (direction + last_direction) ==1;// 0 + 1 or 1 + 0
				pos = pos2;
				last_direction = direction;
			}
			if ( start_decrease )
				len >>=1;
			else len <<=1;

			if ( direction )
				pos2 = pos + len;
			else pos2 = pos - len;
		}
		return result;
	}

	/*
	 * because the search_2 is done in one SM in GPU, no divergence occur
	 * begin and end are assigned from blockIdx, one block only have on value => no divergence
	 *
	 */

	template<bool use1>
	__device__ void search_2(int * V,int search_now,int offset,int begin1,int end1,int begin2,int end2){
		int id = threadIdx.x;
		// begin and end is closed interval [begin , end]

		int *list1 ;
		int *list2 ;
		int value;

		__shared__ int opposite_list[2048];

		int left(0),right;
		if ( use1 ){
			list2 = list_p[ search_now ][ 1 ];
			opposite_list[ id ] = list2[ begin2 + id ];
			if ( begin1 + id > end1 ) return;
			list1 = list_p[ search_now ][ 0 ];
			//left = begin2;
			//right = end2;
			right = end2 - begin2;
			value = list1[ begin1 + id ];
		}
		if ( !use1 ){
			list2 = list_p[ search_now ][ 0 ];
			if ( begin1 + id <= end1 ){
				opposite_list[ id ] = list2[begin1 + id];
				V[ begin1 + id ] = 0;
			}
			if ( begin2 + id > end2 ) return;
			list1 = list_p[ search_now ][ 1 ];
			//left = begin1;
			//right = end1;
			right = end1 - begin1;
			value = list1[ begin2 + id ];
		}
		syncthreads();

//		if ( !id )printf("block:%d flag:%d { %d  -> %d } [%d] -- [%d]\n",
//					blockIdx.x,use1,end1-begin1,end2 - begin2,list_p[ search_now ][ 0 ][end1],list_p[ search_now ][ 1 ][end2]);//debug



		int result = 0;
//		int mid = left;
//
//		while ( left <= right ){
//			mid = (left + right)/2;
//			if ( value == opposite_list[ mid ] ){
//				//printf("block:%d thread: %d INTERSECTION %d\n",blockIdx.x,threadIdx.x,value);
//				result = 1;
//				break;
//			}
//			else if ( opposite_list[mid] < value )
//				left = mid + 1;
//			else right = mid - 1;
//		}
		int pos = id;
		result = gallop_search(value,pos,opposite_list,0,right);


		if (use1){
			//printf("(%d,%d) set V[%d] %d to zero \n",blockIdx.x,threadIdx.x,begin1 + id,V[ begin1 + id ]);
			V[ begin1 + id ] = result;
			if (result){
				//printf("(%d,%d) set V[%d] %d to 1\n",blockIdx.x,threadIdx.x,begin1 + id,value);
			}
		}
		if (!use1){
			//printf("(%d,%d) set V[%d] %d to real_zero \n",blockIdx.x,threadIdx.x,begin1 + id,V[ begin1 + id ]);
			//V[ begin1 + id ] = 0;
			if (result){
				V[ begin1 + pos ] = 1;
				//printf("(%d,%d) set oV[%d] %d to 1\n",blockIdx.x,threadIdx.x,begin1 + mid,value);
			}
		}

	}

	__global__ void algo2_search(int * V,int search_now,int offset){
		//search_1(V,search_now,offset);
		int begin1 = partitions[search_now][blockIdx.x][0]+1;
		int end1 = partitions[search_now][blockIdx.x+1][0];
		int begin2 = partitions[search_now][blockIdx.x][1]+1;
		int end2 = partitions[search_now][blockIdx.x+1][1];
		if (end1 - begin1 <= end2 - begin2){
			search_2<true>(V,search_now,offset,begin1,end1,begin2,end2);
		}
		else {
			search_2<false>(V,search_now,offset,begin1,end1,begin2,end2);
		}
	}




