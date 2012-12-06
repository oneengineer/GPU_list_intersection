#include "memory_data.cuh"
#include "../common_defines.h"


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


