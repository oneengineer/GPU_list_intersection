/*
 * search.cuh
 *
 *  Created on: Nov 27, 2012
 *      Author: xchen
 */
#include <assert.h>
#include "../common_defines.h"

#ifndef SEARCH_CUH_
#define SEARCH_CUH_


__global__ void algo2_search(int * V,int search_now,int offset = 0);

__device__ void scan_buffer_large2(volatile int  shared[][WARP_SIZE*2]);

__global__ void scan_buffer_large(int save_indx);

//template<int num_warp>
//__global__ void scan_buffer_large(){
//	__shared__ volatile int  shared[num_warp][WARP_SIZE*2];
//	scan_buffer_large2(shared);
//}

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
	int *lens;
	int *V;
	int search_now;
	cudaStream_t *stream;

	SearchSettingQueue(int parts,cudaStream_t *stream):parts(parts),stream(stream){
		settings = new SearchSetting[parts];
	}
	void setSettings( int config_settings[][2] ){
		FOR_I(0,parts)
				settings[i].set(config_settings[i][0],config_settings[i][1]);
	}
	void set(int *V,const int &search_now){
		this->V = V;
		this->search_now = search_now;
	}

	int length(){
		return lens[parts];
	}

	void init(){
		lens = new int [parts+1];
		lens[0] = 0;
		FOR_I(1,parts+1){
			lens[ i ] = lens[i-1] + settings[i].size();
		}

	}

	void run_search( int step ){
		assert( step < parts );
		algo2_search<<< settings[step].D1,settings[step].D2,0,*stream>>>(V,search_now,lens[step]);
	}

	void run_all(){
		FOR_I(0,parts)
		run_search(i);
	}

};


#endif /* SEARCH_CUH_ */
