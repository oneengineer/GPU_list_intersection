
#include "../common_defines.h"
#include "memory_data.cuh"
#include <cstdio>

	void init_data(int block_size){
		block_size += 4 + 1 + WARP_SIZE*4;
		block_size *= 2;
		int sizeV = block_size +4;
		int *temp = new int [sizeV];
		FOR_I(0,sizeV) temp[i] = 0;

		cudaMalloc( (void **)&devL1, sizeof(int)*(n+block_size) );
		cudaMalloc( (void **)&devL2, sizeof(int)*(m+block_size) );
		FOR_I(0,QUEUE_SIZE){
			cudaMalloc( (void **)&(devV[i]), sizeof(int)*sizeV );
			cudaMemcpy( devV[i],temp,sizeof(int)*sizeV,H_T_D );
		}
		cudaMalloc( (void **)&devResult, sizeof(int)*(n+block_size) );
		cudaMemcpy( devL1,host_lists[0],sizeof(int)*(n),H_T_D );
		cudaMemcpy( devL2,host_lists[1],sizeof(int)*(m),H_T_D );

		int max1 =  (1<<31-1) - 2 - block_size;
		FOR_I(0,block_size) temp[i] = max1+i;
		cudaMemcpy((int *)(devL1+n),temp,sizeof(int)*block_size,H_T_D);
		FOR_I(0,block_size) temp[i] = max1 -1 -block_size +i;
		cudaMemcpy((int *)(devL2+m),temp,sizeof(int)*block_size,H_T_D);

	}


	__global__ void init_device_var_kernel(){
		int id = CUID;
		if (id) return;
		list_p[0][0] = list_p0[0];
		list_p[0][1] = list_p0[1];
		FOR_I(0,QUEUE_SIZE){
			_result_addr[i] = _result;
			FOR_J(0,4+32) o_scan_buffers[i][j] = 0;
			_scan_buffers[i] = o_scan_buffers[i]+4;
		}
		swapped[0] = 0;//false
		gpu_result_size = 0;
		debug1.wrong_1 = debug1.wrong_2 = 0;
	}


	void init_device_variables(){
		cudaMemcpyToSymbol(list_p0,&devL1,sizeof(int *),0,H_T_D);
		cudaMemcpyToSymbol(list_p0,&devL2,sizeof(int *),0+sizeof(int *),H_T_D);
		int temp[] = {n,m};
		cudaMemcpyToSymbol(_nm,temp,sizeof(int)*2,0,H_T_D);
		cudaMemcpyToSymbol( _result,&devResult,sizeof(int*),0,H_T_D);
		init_device_var_kernel<<<1,1>>>();
		cudaDeviceSynchronize();
		cudaMalloc( (void **)&devIndices , sizeof(int )*2);
	}

	void free_device_memory(){
		cudaFree(devL1);
		cudaFree(devL2);
		cudaFree(devResult);
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

	void cuda_copyResult(){
		int *end;
		cudaMemcpyFromSymbol(&end,_result,sizeof(int *),0,D_T_H );
		Lresult = (int)(end - devResult );
		cudaMemcpy( resultList,devResult, sizeof(int)*Lresult,D_T_H );
	}



