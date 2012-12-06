/*
 * save_result.h
 *
 *  Created on: Nov 27, 2012
 *      Author: xchen
 */

#ifndef SAVE_RESULT_H_
#define SAVE_RESULT_H_

__global__ void save_result(int *V,int indices_now);
__global__ void move_result_addr(int *V,int indices_now);

__global__ void update_and_save(int *V,int size,int indices_now);


#endif /* SAVE_RESULT_H_ */
