/*
 * memory_data.cu
 *
 *  Created on: Nov 27, 2012
 *      Author: xchen
 */
#include "../common_defines.h"

#ifndef MEMORY_DATA_CUH_
#define MEMORY_DATA_CUH_

extern int *host_lists[2];

extern int *devL1,*devL2;
extern bool *v;
extern int *devV[3],*devResult, *devMark;
extern int * devIndices;
extern int n,m;

extern int Lresult;
extern int *resultList;
extern int *cpuResult;
extern int *cpu;




void init_data(int block_size);
void init_device_variables();
void free_device_memory();
void back_next_relative_len(int & len1,int & len2,int indices_now);
void cuda_copyResult();



#endif /* MEMORY_DATA_CUH_ */
