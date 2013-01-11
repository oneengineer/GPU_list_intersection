#include "../common_defines.h"
#include "search.cuh"
#include <cstdio>

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

	__device__ void use_shared(){}

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

		__shared__ int opposite_list[128];

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
			opposite_list[ id ] = list2[begin1 + id];
			if ( begin2 + id > end2 ) return;
			list1 = list_p[ search_now ][ 1 ];
			//left = begin1;
			//right = end1;
			right = end1 - begin1;
			value = list1[ begin2 + id ];
		}

//		if ( !id )printf("block:%d flag:%d { %d  -> %d } [%d] -- [%d]\n",
//					blockIdx.x,use1,end1-begin1,end2 - begin2,list_p[ search_now ][ 0 ][end1],list_p[ search_now ][ 1 ][end2]);//debug


		//TODO replace list2 as shared!
		int result = 0;
		int mid = left;

		while ( left <= right ){
			mid = (left + right)/2;
			if ( value == opposite_list[ mid ] ){
				//printf("block:%d thread: %d INTERSECTION %d\n",blockIdx.x,threadIdx.x,value);
				result = 1;
				break;
			}
			else if ( opposite_list[mid] < value )
				left = mid + 1;
			else right = mid - 1;
		}

		if ( result ){
			if (use1){
				V[ begin1 + id ] = 1;
//				printf("(%d,%d) set V[%d] %d to 1\n",blockIdx.x,threadIdx.x,begin1 + id,value);
			}
			if ( !use1 ){
				V[ begin1 + mid ] = 1;
//				printf("(%d,%d) set oV[%d] %d to 1\n",blockIdx.x,threadIdx.x,begin1 + mid,value);
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
			if ( end1 < 128 ) search_2<true>(V,search_now,offset,begin1,end1,begin2,end2);
		}
		else {
			if (end2 < 128)
			search_2<false>(V,search_now,offset,begin1,end1,begin2,end2);
		}
	}




