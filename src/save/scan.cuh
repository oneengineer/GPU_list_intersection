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



void scan(int *,int );
void scan2(int *,int );
void init_scan(cudaStream_t * ,int );

__global__ void scan_x(int *array,int small_size);

extern void cu_host_print(int *list,int size);

struct ScanSequence{

	int steps;
	int* devL;
	int n;
	int save_indices_now;

	ScanSequence(int steps,int *devL,int n):steps(steps),devL(devL),n(n){
		//! small_size AT LEAST 64
		small_size = 1024;
	}

	void run_scan(int step);
	void run_large();
	void run_save(int step);
	void run_saveLarge();

	void init(){
		D2 = small_size / 2;
		D1 = n / (small_size * steps);
		step_size = n / steps;
	}

	void set(int *devL,int save_indices_now){
		this->devL = devL;this->save_indices_now=save_indices_now;
	}



	void run_all(){
		FOR_I(0,steps) run_scan(i);
		run_large();
		FOR_I(0,steps) run_save(i);
		run_saveLarge();
	}

	int D1,D2;
	int small_size ;

	int step_size;

};

#endif /* SCAN_CUH_ */
