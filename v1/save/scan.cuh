/*
 * scan.cuh
 *
 *  Created on: Nov 27, 2012
 *      Author: xchen
 */

#include "../common_defines.h"
#include <cstdio>

#ifndef SCAN_CUH_
#define SCAN_CUH_


__global__ void scan_large(int *array,int *buffer,int blocks,int block_n);

#endif /* SCAN_CUH_ */
