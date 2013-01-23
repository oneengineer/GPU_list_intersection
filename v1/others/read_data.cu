#include "read_data.cuh"
#include "../common_defines.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

//using namespace std;

const char *dir = "/home/sudakentemp/data1.dat";
const int MAX_BLOCK_SIZE = 1024 * 3072;

extern struct list_info *cpuData;
/*
 * Store in int 4 byte
 * Every list start from a len, the id,id,id...
 *
 * */

	struct list_info read_one_list(int * datapool,int file_offset,std::ifstream &reader){
		struct list_info result;
		reader.seekg(file_offset,reader.beg);
		int list_len = -1;
		reader.read((char *)&list_len,4);
		reader.read((char *)datapool,4*list_len);
		result.start_addr = datapool;
		result.len = list_len;
		return result;
	}

	bool filter(int len){
		if (len < 1000*100)
			return false;
		return true;
	}

	void readStructure(std::ifstream &reader,int *lens,int &availiable_list_num,int &data_len_sum){
		int fileLen;
		reader.seekg(0,reader.end);
		fileLen = reader.tellg();
		reader.seekg(0,reader.beg);
		std::cout<<"file length = "<<fileLen<<std::endl;

		int current_offset = 0;
		int list_num = 0;
		availiable_list_num = 0;
		data_len_sum = 0;
		while (current_offset < fileLen){
			reader.seekg(current_offset,reader.beg);
			int list_len = -1;
			reader.read((char *)&list_len,4);
			if (filter(list_len)){
				lens[availiable_list_num ++] = current_offset;
				data_len_sum += list_len;
			}
			current_offset += list_len*4 +4;
			list_num ++;
		}
	}

	__global__ void show_data(struct list_info *list_info){
		printf("len is %d\n",list_info->len);
		FOR_I(0,10){
			printf("%d\t",list_info->start_addr[i]);
		}
	}

	__global__ void show_raw_data(int *addr){
		FOR_I(0,10){
			printf("%d\t",addr[i]);
		}
	}

	__global__ void correct_address(struct list_info *list_infos,int size,int *cpu_addr0,int *gpu_addr0){
		int id = threadIdx.x;
		while (id < size){
			list_infos[id].start_addr = list_infos[id].start_addr - cpu_addr0 + gpu_addr0;
			//printf("<%d> %lld\n",id,list_infos[id].start_addr);
			id += blockDim.x;
		}

	}

void read_gov2(){

	std::ifstream reader;
	reader.open(dir,std::fstream::binary | std::fstream::in);
	int list_num;
	int array[10000];
	int sum;
	readStructure(reader,array,list_num,sum);
	printf("SUM = %d, list_num %d\n",sum,list_num);//debug
	data_pool = new int[sum];
	int* current_pool_pos = data_pool;
	cpuData = new struct list_info[sum];
	FOR_I(0,list_num){
		struct list_info temp = read_one_list(current_pool_pos,array[i],reader);
		current_pool_pos += temp.len;
		cpuData[i] = temp;
	}

	cudaMalloc(&gpu_data_pool,4*sum);
	cudaMemcpy(gpu_data_pool,data_pool,4*sum,H_T_D);
	cudaMalloc(&gpuData,sizeof(struct list_info) * list_num);
	cudaMemcpy(gpuData,cpuData,sizeof(struct list_info) * list_num,H_T_D);

	correct_address<<<1,400>>>(gpuData,list_num,data_pool,gpu_data_pool);
	cudaDeviceSynchronize();

}



