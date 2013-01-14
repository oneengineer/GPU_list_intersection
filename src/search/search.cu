#include "../common_defines.h"
#include "search.cuh"
#include <cstdio>


#if DEPLETED

	//binary search lower bound in list2 for each element in list1
	//algo2 should not care about swap or not


	__device__ void search_1(int * V,int search_now,int offset){
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

	//Too many command
	__global__ void gallop_search_stage2(int *devL1,int *devL,int n){
		int id = threadIdx.x;
		int value = devL1[id];
		int pos = id;
		int left = 0,right = n;
		int bound;
		int result = 0;
		int len;
		//printf("id:%d get value:%d\n",id,value);
		int value2;

		int debugid = 20;
		int add_to_len;

		bool direction;
		//decide direction
		if ( devL[pos] <= value ) // right -->> direction
			direction = true,bound = right,len=1,add_to_len=0;
		else direction = false,bound = left,len=-1,add_to_len=1;
		printf(" (%d) value: %d  direction:%d  len:%d\n",id,value,direction,len);

		//len increase only one direction
		int pos2;
		while ( true ){
			value2 = devL[pos];
			if (id == debugid) printf(" compared with [%d]:%d,it move to %d for next time %d\n",pos,value2,pos,len);
			if ( value2 == value ) break;
			else if ( direction == ( value2 < value ) ){
				pos2 = pos + len;
			}
			else {
				len = (len+add_to_len)>>1;
				break;}
			if (!( left <= pos2 && pos2 <=right )) break;
			pos = pos2;
			len <<= 1;
		}

		if (!id) printf(" stage 1 over\n");
		printf(" (%d) value: %d  pos:%d  len:%d\n",id,value,pos,len);

		//len = 1;
		//len decrease, possible two directions
		while ( len!=0 ){
			len = (len+add_to_len)>>1;
			if (id == debugid) printf(" compared with %d,it move to %d for next time %d\n",value2,pos,len);
			if ( value2 == value ){
				result = 1;
				//add a break?
				break;
			}
			else if ( direction == (value2 < value )){
				pos += len;
			}
			else pos -= len;

			value2 = devL[pos];
		}

		if ( result )
			printf("(%d)  found %d at [%d]\n",id,value,pos);
		syncthreads();
		devL1[id] = result;
	}

	void test_gallop_search(){
		n = 1001;
		generate_random(1.0,1,1);
		//FOR_I(4,n)
//		host_lists[0][i] *=10;
		//host_lists[0][1] = 33;
		//host_lists[0][0] = 17;
		//generate_same(2);
		//generate_case3();
		debug_a(host_lists[0],n);outline;debug_a(host_lists[1],m);//debug
		init_data(128);
		init_device_variables();
		cudaDeviceSynchronize();
		HERE;
		//n = 32;

		int * cpuresult = new int[n];
		FOR_I(0,n){
			cpuresult[i] = 0;
			FOR_J(0,n)
			if ( host_lists[0][i] == host_lists[1][j] ){
				cpuresult[i] = 1;
				break;
			}
		}
		gallop_search<<<1,n>>>(devL1,devL2,n-1);
		cudaMemcpy(resultList,devL1,sizeof(int)*n,D_T_H);

		cudaDeviceSynchronize();

		debug_a(resultList,n);
		debug_a(cpuresult,n);

		cu_checkError();

		FOR_I(0,n)
			if ( resultList[i] != cpuresult[i] ){
				printf("WRONG AT [%d]:%d\n",i,host_lists[0][i]);
				exit(0);
			}
	}

#endif



	//extern __device__ void scan_a_block_neat(int * src_data,volatile int shared[][WARP_SIZE*2]);

	template<int loops,bool exclusive>
	inline __device__ void scan_warp_neet(volatile int * shared,int value,int id){
		shared[id] = 0;
		id += WARP_SIZE;
		shared[id] = value;

		if ( 1 <= loops ) shared[id] += shared[id - 1];
		if ( 2 <= loops ) shared[id] += shared[id - 2];
		if ( 3 <= loops ) shared[id] += shared[id - 4];
		if ( 4 <= loops ) shared[id] += shared[id - 8];
		if ( 5 <= loops ) shared[id] += shared[id - 16];

		if ( exclusive )
			shared[id] -= value;
	}

	__device__ void scan_a_block_neat(volatile int * src_data,volatile int shared[][WARP_SIZE*2]){
		int id = threadIdx.x;
		int local_id = id&(WARP_SIZE-1);
		int warp_num = id >> LOG_WARP_SIZE ;
		volatile __shared__ int shared_level2[WARP_SIZE*2];

		//printf("id:%d src: %llx  shared:%llx\n",id,src_data+id,shared[warp_num]+id);//debug
		scan_warp_neet< LOG_WARP_SIZE,false>(shared[warp_num],src_data[id],local_id );//basic level
		syncthreads();
		if ( id < WARP_SIZE){
			int myvalue = shared[id][WARP_SIZE+WARP_SIZE-1];
			scan_warp_neet<LOG_SCAN_BLOCK_SIZE - LOG_WARP_SIZE ,true>(shared_level2,myvalue,id);//second level, use calculated 32,in one block
		}
		syncthreads();
		//uniform update
		int a = shared_level2[ warp_num+WARP_SIZE ];
		src_data[id] = shared[warp_num][ local_id + WARP_SIZE ] + a;
	}

	/**
	 *
	 * To write the program use minimum command
	 * Three things decide the next position:
	 * 1. position out of boundary?
	 * 2. get value, direction? change or not?
	 * 3. length? how l
	 * L: position = what
	 *
	 *
	 * value: the value to search
	 * pos2: starting place, extrapolated position, it need to be return!
	 * return:result 0/1
	 *
	 */

	inline __device__ int gallop_search (int value,int & pos2,volatile int *list,int left,int right){

		int pos = pos2;
		int result = 0;
		int len;
		int value2;

		char last_direction = 3;
		char direction = 0;
		char start_decrease = 0;

		len = 1;
		while ( len > 0){

			if (!( left <= pos2 && pos2 <= right )){
				start_decrease = 1;
			}
			else{
				value2 = list[pos2] - value;
				if ( 0 == value2 ){
					result = 1;break;
				}
				else direction = value2 < 0;
				start_decrease |= (direction + last_direction) ==1;// 0 + 1 or 1 + 0
				pos = pos2;
				last_direction = direction;
			}
			if ( start_decrease )
				len >>=1;
			else len <<=1;

			if ( direction )
				pos2 = pos + len;
			else pos2 = pos - len;
		}
		return result;
	}

	inline __device__ int binary_search(int value,int & pos,volatile int *list,int left,int right){
		int result = 0;
		while ( left <= right ){
			if ( value == list[ pos ] ){
				//printf("block:%d thread: %d INTERSECTION %d\n",blockIdx.x,threadIdx.x,value);
				result = 1;
				break;
			}
			else if ( list[pos] < value )
				left = pos + 1;
			else right = pos - 1;
			pos = (left + right)/2;
		}
		return result;
	}

	/*
	 * because the search_2 is done in one SM in GPU, no divergence occur
	 * begin and end are assigned from blockIdx, one block only have on value => no divergence
	 *
	 */
	inline __device__ int search_one_value(int value,int & pos,volatile int *opposite_addr,const int &left,const int &right){

		int result = binary_search(value,pos,opposite_addr,left,right);
		//printf("<%d,%d>search %d in (%d %d) to %d\n",blockIdx.x,threadIdx.x,value,left,right,pos);//debug
//
		if (result){
			int write_pos = atomicAdd(&gpu_result_size,1);
			_result[write_pos] = value;
		}

//		if (result)
//		printf("B<%d> %d was found at [%d]\n",blockIdx.x,value,pos);

		return result;
	}

	/*
	 * search 4 integer stored in uint4, a,b,c,d
	 * use divide-and-conquer way a,c,b,d order search
	 *
	 */
	inline __device__ void search_uint4(const int & id,const uint4 &myvalue,volatile int *opposite_list,
			volatile int *shared_range,const struct partition_info &info,
			int &mask,int &uint4_sum){
		int pos_A,pos_X,pos_C,pos_D;
		//pos_A = (info.left + info.right)/2;
		pos_A = id;
		mask = 0,uint4_sum=0;
		if (search_one_value(myvalue.x,pos_A,opposite_list,info.left,info.right)){
			mask |=1,uint4_sum ++;
		}
		shared_range[id] = pos_A;
		if ( pos_A <0 || pos_A > info.right ) printf("!!!");
		syncthreads();
		int id_addone = id + 1;
		if ( id_addone != info.len )
			pos_D = shared_range[id_addone];
		else pos_D = info.right;

		pos_C = (pos_A + pos_D) /2 ;
		if (search_one_value(myvalue.z,pos_C,opposite_list,pos_A,pos_D)){
			mask |=4,uint4_sum ++;
		}
		if ( pos_C <0 || pos_C > info.right ) printf("!!!");
		pos_X = (pos_A + pos_C) /2 ;
		if (search_one_value(myvalue.y,pos_X,opposite_list,pos_A,pos_C)){
			mask |=2,uint4_sum ++;
		}
		if ( pos_X <0 || pos_X > info.right ) printf("!!!");
		pos_X = (pos_C + pos_D) /2 ;
		if (search_one_value(myvalue.w,pos_X,opposite_list,pos_C,pos_D)){
			mask |=8,uint4_sum ++;
		}
		if ( pos_X <0 || pos_X > info.right ) printf("!!!");
	}

	/*
	 * Simple method, not use shared
	 * */
//	inline __device__ void search_uint4_2(const int & id,const uint4 &myvalue,volatile int *opposite_list,
//			volatile int *shared_range,const struct partition_info &info,
//			int &mask,int &uint4_sum){
//		int pos_A,pos_X,pos_D;
//		pos_A = id;
//		mask = 0,uint4_sum=0;
//		if (search_one_value(myvalue.x,pos_A,opposite_list,info.left,info.right)){
//			mask |=1,uint4_sum ++;
//		}
//		pos_D =  ( pos_A+ info.right)/2;
//		if (search_one_value(myvalue.z,pos_D,opposite_list,pos_A,info.right)){
//			mask |=8,uint4_sum ++;
//		}
//		pos_X = (pos_A + pos_D) /2 ;
//		if (search_one_value(myvalue.y,pos_X,opposite_list,pos_A,pos_D)){
//			mask |=2,uint4_sum ++;
//		}
//		if (search_one_value(myvalue.w,pos_X,opposite_list,pos_X,pos_D)){
//			mask |=4,uint4_sum ++;
//		}
//	}

	inline __device__ void scan_and_save_buffer(volatile int *scan_array,volatile int *scan_shared,int * save_buffer,
			int id,uint4 &myvalue,int &mask,int &uint4_sum,const struct partition_info & info){
		scan_array[id] = uint4_sum;
		//printf("<%d %d> position:%d %d\n",blockIdx.x,threadIdx.x,scan_array[id],info.len);
		syncthreads();

//		if (!id){
//			FOR_I(0,blockDim.x)
//				printf("[%llx] %d\t",scan_array+i,scan_array[i]);
//			printf("\n");
//			FOR_I(0,blockDim.x)
//				printf("[%llx] %d\t",scan_shared+i,scan_shared[i]);
//			printf("\n");
//		}
		scan_a_block_neat((int *)scan_array,(int (*)[WARP_SIZE*2])scan_shared);

//		if (mask & 1)
//				printf("B<%d> %d was found at\n",blockIdx.x,myvalue.x);
//		if (mask & 2)
//				printf("B<%d> %d was found at\n",blockIdx.x,myvalue.y);
//		if (mask & 4)
//				printf("B<%d> %d was found at\n",blockIdx.x,myvalue.z);
//		if (mask & 8)
//				printf("B<%d> %d was found at\n",blockIdx.x,myvalue.w);


		int position = scan_array[id] - uint4_sum; // inclusive scan
		//printf("<%d %d> position:%d\n",blockIdx.x,threadIdx.x,position);

//		if ( mask & 1 )
//			save_buffer[position] = myvalue.x;
//		if ( mask & 2 )
//			save_buffer[position+1] = myvalue.y;
//		if ( mask & 4 )
//			save_buffer[position+2] = myvalue.z;
//		if ( mask & 8 )
//			save_buffer[position+3] = myvalue.w;

	}

	inline __device__ void check_address(const struct partition_info &info,int * shared,uint4 myvalue){
		int *abs_add1,*abs_opposite;
		bool flag1 = false;
		if ( info.addr - list_p0[0] <0 && info.opposite_addr - list_p0[1] <0 ){
			printf("ONLY APPEAR ONCE\n");
		}

		if ( info.addr - list_p0[0] >=0 && info.opposite_addr - list_p0[1] >=0 ){
			abs_add1 = list_p0[0];
			abs_opposite = list_p0[1];
		}
		else {
			abs_add1 = list_p0[1];
			abs_opposite = list_p0[0];
			flag1 = true;
		}
		//FOR_I(0,4*info.len){
			LL offset = info.addr + (LL)(4*threadIdx.x) - abs_add1;
/*			if ( offset <0 || offset >= _nm[0] + 128 *  8*32*4 ){
				atomicAdd(&debug1.wrong_2,1);
				return;
			}*/
			atomicAdd(&debug1.wrong_2,myvalue.x);
			return;
			if ( myvalue.y != abs_add1[offset+1] ){
				atomicAdd(&debug1.wrong_2,1);
			}
		//}

//		FOR_I(0,4*info.len_opposite){
//			LL offset = info.opposite_addr + (LL)i - abs_opposite;
//
//			if ( abs_opposite[offset] != shared[i] ){
//				atomicAdd(&debug1.wrong_1,1);
//				//printf("wrong copy %d %d\n",info.opposite_addr[offset],shared[i]);
//			}
//			if (335544314 == abs_opposite[offset] || 335544319 == abs_opposite[offset]){
//				printf("END : %d \n",offset);
//			}
//
//			if ( offset <0 || offset>= _nm[0] + 128 *  8*32*4 )
//				printf("wrong %d offset",offset-_nm[0]);
//		}

	}

	inline __device__ void brute_force(uint4 & myvalue,volatile int * list,const struct partition_info &info){
		FOR_I(info.left,info.right+1){
			if ( myvalue.x == list[i] ){
				int write_pos = atomicAdd(&gpu_result_size,1);
				_result[write_pos] = myvalue.x;
			}
			if ( myvalue.y == list[i] ){
				int write_pos = atomicAdd(&gpu_result_size,1);
				_result[write_pos] = myvalue.y;
			}
			if ( myvalue.z == list[i] ){
				int write_pos = atomicAdd(&gpu_result_size,1);
				_result[write_pos] = myvalue.z;
			}
			if ( myvalue.w == list[i] ){
				int write_pos = atomicAdd(&gpu_result_size,1);
				_result[write_pos] = myvalue.w;
			}
		}
	}

	inline __device__ void search_in_block(int * V,int search_now,const struct partition_info &info){
		int id = threadIdx.x;
		// begin and end is closed interval [begin , end]

		volatile __shared__ int opposite_list[10*32*4];
		//volatile __shared__ int shared_range[1024];// only use once, the right most thread does not use it
		int mask,uint4_sum;
		uint4 myvalue ;
		bool flag1 = false;
		/*
		 * shared_range __shared__ use as src_data for scan
		 * opposite_list __shared__ use as
		 *  */

		// the thread which should copy opposited list elements

		if (id < info.len_opposite){

			myvalue = ((uint4 *)info.opposite_addr)[id];
			//atomicAdd(&debug1.wrong_1,myvalue.w);
			((uint4 *)opposite_list)[id] = myvalue;
			if ( id < info.len ){
				myvalue = ((uint4 *)info.addr)[id];
				//atomicAdd(&debug1.wrong_2,myvalue.w);
				flag1 = true;
				//search_uint4(id,myvalue,opposite_list,shared_range,info,mask,uint4_sum);
			}
		}
		syncthreads();

//		if ( !threadIdx.x ){
//			FOR_I(0,info.len_opposite)
//				printf("%d\t",info.opposite_addr[i]);
//			FOR_I(0,info.len_opposite)
//				printf("%d\t",info.opposite_addr[i]);
//		}

		if (flag1){
			//check_address(info,(int *)opposite_list,myvalue);
			brute_force(myvalue, info.opposite_addr ,info);
			//search_uint4(id,myvalue,opposite_list,shared_range,info,mask,uint4_sum);
		}
		//

		//if ( id < info.warp_len )
		//scan_and_save_buffer(shared_range,opposite_list,V,id,myvalue,mask,uint4_sum,info);
	}

	__global__ void algo2_search(int * V,int search_now,int offset){
		if (partitions_info[search_now][blockIdx.x].len>0){
			search_in_block(V,search_now,partitions_info[search_now][ blockIdx.x ]);
		}
	}

#if DEPLETED

	template<bool use1>
	__device__ void search_2(int * V,int search_now,int offset,int begin1,int end1,int begin2,int end2){
		int id = threadIdx.x;
		// begin and end is closed interval [begin , end]

		int *list1 ;
		int *list2 ;
		int value;

		volatile __shared__ int opposite_list[2048];

		int left(0),right;
		if ( use1 ){
			list2 = list_p[ search_now ][ 1 ];
			opposite_list[ id ] = list2[ begin2 + id ];
			if ( begin1 + id > end1 ) return;
			list1 = list_p[ search_now ][ 0 ];
			//left = begin2;
			//right = end2;
			right = end2 - begin2;
			value = list1[ begin1 + id ];
		}
		if ( !use1 ){
			list2 = list_p[ search_now ][ 0 ];
			if ( begin1 + id <= end1 ){
				opposite_list[ id ] = list2[begin1 + id];
				V[ begin1 + id ] = 0;
			}
			if ( begin2 + id > end2 ) return;
			list1 = list_p[ search_now ][ 1 ];
			//left = begin1;
			//right = end1;
			right = end1 - begin1;
			value = list1[ begin2 + id ];
		}
		syncthreads();

//		if ( !id )printf("block:%d flag:%d { %d  -> %d } [%d] -- [%d]\n",
//					blockIdx.x,use1,end1-begin1,end2 - begin2,list_p[ search_now ][ 0 ][end1],list_p[ search_now ][ 1 ][end2]);//debug


		int result = 0;
		int pos = id;
		//result = binary_search(value,pos,opposite_list,left,right);
		result = gallop_search(value,pos,opposite_list,left,right);


		if (use1){
			//printf("(%d,%d) set V[%d] %d to zero \n",blockIdx.x,threadIdx.x,begin1 + id,V[ begin1 + id ]);
			V[ begin1 + id ] = result;
			if (result){
				//printf("(%d,%d) set V[%d] %d to 1\n",blockIdx.x,threadIdx.x,begin1 + id,value);
			}
		}
		if (!use1){
			//printf("(%d,%d) set V[%d] %d to real_zero \n",blockIdx.x,threadIdx.x,begin1 + id,V[ begin1 + id ]);
			//V[ begin1 + id ] = 0;
			if (result){
				V[ begin1 + pos ] = 1;
				//printf("(%d,%d) set oV[%d] %d to 1\n",blockIdx.x,threadIdx.x,begin1 + mid,value);
			}
		}

	}

	__global__ void algo2_search(int * V,int search_now,int offset){
		//search_1(V,search_now,offset);
		int begin1 = partitions[search_now][blockIdx.x][0]+1;
		int end1 = partitions[search_now][blockIdx.x+1][0];
		int begin2 = partitions[search_now][blockIdx.x][1]+1;
		int end2 = partitions[search_now][blockIdx.x+1][1];
		if (end1 - begin1 <= end2 - begin2){
			search_2<true>(V,search_now,offset,begin1,end1,begin2,end2);
		}
		else {
			search_2<false>(V,search_now,offset,begin1,end1,begin2,end2);
		}
	}
#endif




