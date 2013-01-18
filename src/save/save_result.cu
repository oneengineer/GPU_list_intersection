#include "../common_defines.h"
#include "save_result.h"
#include "scan.cuh"
#include <cstdio>

extern int *scan_buffer;
extern cudaStream_t *scan_stream;

	__device__ void save_result1(int *V,int indices_now){
			int _size = calculated_indices_len[indices_now][0];
			int id = CUID;
			int cusize = CUSIZE;
			int *list1 = list_p[indices_now][0];

			while (id <= _size ){
				int diff = V[id] - V[id-1];
				if ( diff >0){
					_result[ V[id-1] ] = list1[ id ];
				}
				id += cusize;
			}
		}

	__global__ void save_result(int *V,int indices_now){
		save_result1(V,indices_now);
	}


	inline __device__ void block_update_and_save(const int &id, int *V, int partial_addup,int indices_now){
		int *L1 = list_p[indices_now][0];

		if ( V[id] - V[id-1] > 0 ){
			_result[ V[id-1]+partial_addup ] = L1[ id ];
			//printf("in block_update_and_save (id:%d) %d is saved at [%d]\n",id,L1[ id ],V[id-1]+partial_addup);//debug
		}
	}

	__device__ void inner_move_result_addr(int *V,const int & indices_now){
		int _size = calculated_indices_len[indices_now][0];
		_result += V[_size];
	}

	__global__ void block_update_level2(int *buffer, int *V, int small_size,int indices_now){
		int *L1 = list_p[indices_now][0];
		int id = threadIdx.x;
		int idx = id * small_size ;
		int _size = calculated_indices_len[indices_now][0];

		if ( idx > _size ) return ;

		int a = buffer[ id];
		if ( V[idx] >0 ){
			_result[ a ] = L1[ idx ];
		}
		int part_id = _size / small_size;
		syncthreads();
		if ( part_id == id ){ // relative last part
			V[ _size ] += a;
			_result += V[ _size ];
		}
	}


	__global__ void update_and_save(int *buffer,int *V,int indices_now,int offset = 0){
		int _size = calculated_indices_len[indices_now][0];
		int partial_addup = buffer[ blockIdx.x ];
		int id = CUID + offset;

		if (id <= _size){
			block_update_and_save( id,V,partial_addup,indices_now);
		}
	}

	__global__ void move_result_addr(int *V,int indices_now){
		//inner_move_result_addr(V,indices_now);
	}

	void ScanSequence::run_save(int step){
		//dim3 threads(32,32);
		dim3 threads(small_size);
		int offset = step*step_size;
		int b_offset = offset / small_size;
		update_and_save<<< D1,small_size,0,*scan_stream>>>(scan_buffer+3 + b_offset,devL,save_indices_now,offset);
	}

	void ScanSequence::run_saveLarge(){
		block_update_level2<<<1,n/small_size,0,*scan_stream>>>(scan_buffer+3,devL,small_size,save_indices_now);
	}

	inline __device__ void one_block_compact(const int &compact_blocks,int *buffer,int *result,const int & save_idx){
		int id = threadIdx.x;
		int bsize = blockDim.x;
		int end;
		int *save_addr = _result_addr[save_idx];

		FOR_I( 0 ,compact_blocks){
			int j = buffer[i-1];
			end = buffer[i];
			int id2 = id;
			j += id;

			while ( j < end ){
				save_addr[ j ] = result[id2];
				j += bsize;
				id2 += bsize;
			}
			result += DEF_D2;
		}
		if ( 0 == id && blockIdx.x == gridDim.x -1 ){
			int next_idx = (save_idx + 1)%QUEUE_SIZE;
			//printf("End move: %d\n",end);//debug
			_result_addr[next_idx] = end + save_addr;
			_result = _result_addr[next_idx];//atmoic
		}
	}

	__global__ void compact(int compact_blocks,int *result,int save_idx){
		int offset1 = blockIdx.x * compact_blocks;
		int offset2 = offset1 * DEF_D2;
		one_block_compact(compact_blocks,_scan_buffers[save_idx] + offset1,result+offset2,save_idx);
	}




