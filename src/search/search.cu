#include "../common_defines.h"
#include "search.cuh"

	//binary search lower bound in list2 for each element in list1
	//algo2 should not care about swap or not
	__global__ void algo2_search(int * V,int search_now,int offset){
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
