

#include "cal_indices.cuh"
#include "../common_defines.h"
#include <cstdio>

	__device__ int next_queue_pos(int value){
			return (value+1) % QUEUE_SIZE;
	}

	// config should be <1,4>
	__device__ void cal_indx_1(int block_size,int block_2_size,int indices_now){
		int id = threadIdx.x;
		int group_id = id >> 1;
		int sub_id = id & 1;
		int opposite_sub_id = sub_id ^ 1; //!sub_id

		int indices_next = (indices_now + 1) % QUEUE_SIZE;
		int temp_blocks[] = {block_size, block_2_size};

		int *myList = list_p[indices_now][ group_id ^ sub_id ];
		int *oppositeList = list_p[indices_now][ group_id ^ opposite_sub_id ];
		int myLen = temp_blocks[sub_id];
		int oppositeLen = temp_blocks[ opposite_sub_id ];
		int value = myList[ myLen - 1 ];
		int left = 0, right = oppositeLen -1;

		while ( left < right ){
			int mid = (left + right + 1)/2;
			if ( value < oppositeList[mid] )
				right = mid - 1;
			else left = mid;
		}

		//printf("id:%d value %d get to %d\n",id,value,oppositeList[left]);//debug

		int next_opposite_offset = left + ( oppositeList[left] <= value );

		int *save_indices0 = calculated_indices_len[indices_now];
		int *save_indices = save_indices0 + (group_id << 1);
		int *opposite_indices = save_indices0 + ( (group_id^1) << 1);
		save_indices[sub_id] = next_opposite_offset ;

		int mysum = save_indices[ sub_id ] + save_indices[ opposite_sub_id ];
		int oppositeSum = opposite_indices[ sub_id ] + opposite_indices[ opposite_sub_id ];

		//decide opposite list offset and length
		if ( mysum + group_id > oppositeSum){

			//printf("cal TURN: id %d len: %d  \n",id,next_opposite_offset);//debug
			save_indices0[ opposite_sub_id ] = next_opposite_offset - 1;
			list_p[indices_now][sub_id] = myList;

			list_p[indices_next][ opposite_sub_id ] = oppositeList + next_opposite_offset;//set opposite pointer
			swapped[ indices_next ] = swapped[ indices_now ] ^ group_id;
		}
	}

	__device__ void bitonic_merge(bool up, volatile int * data,int id, int n){
		if ( n <1 ) return ;
		int n2 = n >> 1;

		syncthreads();
		if  ((data[id] > data[id + n]) ^ up ){
			int t = data[id];
			data[id] = data[id+n];
			data[id+n] = t;
		}
		// change part of the thread
		if ( id >= n2 )
			id -= n2,data += n;
		bitonic_merge(up, data,id, n2);
	}

	//config should be dim3 ths(16,2)

#define ALIGN_MUL 4 // 4 int
#define ALIGN_ADDR_MUL (4*ALIGN_MUL-1) // int = 4 byte
#define ALIGN_MOD (ALIGN_MUL-1)


	inline __device__ int complement(int value,int mod){
		return (mod - ( value &mod) )& mod;
	}

	/*
	 * begin ,end : the data range. may larger than the calculation needs
	 * left, right: the search range. or the real data range
	 */
	inline __device__ void make_memory_alignment(int *addr1,int *addr2
									,int &begin1, int &begin2
									,int &end1, int &end2
									,int &left1, int &left2
									,int &right1, int &right2
									,int &len1, int &len2){
//		int offset1 = ((long long)(addr1) & ALIGN_ADDR_MUL) / 4;
//		int offset2 = ((long long)(addr2) & ALIGN_ADDR_MUL) / 4;
//
//		printf("vof1: %d vof2: %d\n",*addr1,*addr2);
//		printf("[ %llx  %llx ] of1: %d of2: %d\n",addr1,addr2,offset1,offset2);



		left1 = begin1 & ALIGN_MOD;
		left2 = begin2 & ALIGN_MOD; // === %4 for the memory alignment
		right1 = complement(end1,ALIGN_MOD);
		right2 = complement(end2,ALIGN_MOD);
		begin1 -= left1,begin2 -=left2;
		end1 += right1,end2 += right2; // === %4 for the memory alignment

		len1 = end1 - begin1,len2 = end2 - begin2;
		right1 = len1 - right1,right2 = len2 - right2;

//		if ( (len1+1) % 4!=0 || (len2+1) %4 !=0 ) printf("!!wrong\n");
//		if ( begin1 < 0 || begin2 < 0) printf("!!very wrong\n");

	}

	__device__ bool monotone_check(int *list,int n){
		FOR_I(1,n)
				if ( !(list[i-1] <= list[i]) )
					return false;
		return true;
	}

	__device__ void cal_indx_2(int parts,int part_size,int block_size,int indices_now){
		int id = threadIdx.x;
		int myside = threadIdx.y;
		int opposite_side = !myside;
		int idx = (id+1)*part_size-1;
		int opposite_idx;
		int *myList = list_p[indices_now][ myside ];
		int *oppositeList = list_p[indices_now][ opposite_side ];

		int myValue = myList[ idx ];

		//------ bsearch upper bound part----------
		int left = 0,right = block_size;
		//TODO the logical might be wrong
		while ( left < right ){
			int mid = (left + right + 1)/2;
			if ( myValue < oppositeList[mid] )
				right = mid - 1;
			else left = mid;
		}


		//------ END bsearch upper bound part----------
		opposite_idx = left + ( oppositeList[left] <= myValue ) - 1;

		__shared__ volatile int shared[2][2*DEF_D1+1];
		shared[myside][ parts - id - 1 ] = idx; //reverse save
		shared[opposite_side][ parts + id ] = opposite_idx;

		syncthreads();
		bitonic_merge(false,shared[myside],id,parts);
		syncthreads();

//		if (!id && !monotone_check( (int *)(shared[myside]),parts*2))
//			printf("WRONG!!");

		//debug print out
//		if ( 12 == debug1.num_loop )
//		if ( !myside && !id ){
//			printf("[%d]:%d ==== [%d]:%d\n",0,myList[0],0,oppositeList[0]);
//			FOR_I(0,parts*2){
//				int a = shared[0][ i ];
//				int b = shared[1][ i ];
//				printf("[%d]:%d --- [%d]:%d\n",a,myList[a],b,oppositeList[b]);
//			}
//		}

		syncthreads();
		int whole_id = blockDim.x * myside + id;
		__shared__ volatile int shared_decide_next_addr[2*DEF_D1+1];//default may be not 0!!
		shared_decide_next_addr[whole_id] = max( shared[0][whole_id], shared[1][whole_id] ) < block_size;
		syncthreads();
		int indices_next = (indices_now + 1) % QUEUE_SIZE;

		//decide partitions address, which is list 1 and which is list 2
		int begin1,end1,begin2,end2,len1,len2;
		if ( whole_id == 0 ){
			begin1 = calculated_indices_len[indices_now][0];
			begin2 = calculated_indices_len[indices_now][1];
			//begin1 = 0;
			//begin2 = 0;
		}
		else {
			begin1 = shared[0][whole_id-1]+1;
			begin2 = shared[1][whole_id-1]+1;
		}
		int left1,left2,right1,right2;
		end1 = shared[0][whole_id], end2 = shared[1][whole_id];

		make_memory_alignment(list_p[indices_now][ 0 ],list_p[ indices_now ][ 1 ]
							,begin1, begin2,end1 ,end2, left1, left2,right1, right2,len1, len2);


		if ( 0 == shared_decide_next_addr[whole_id] ){
			len1 = len2 = -999;//do not do calculation
		}
		struct partition_info *info = & partitions_info[indices_now][whole_id];
		if ( len1 <= len2 ){
			// A ---> B
			info->B2A = false;
			info->addr = list_p[indices_now][0]+begin1;
			info->opposite_addr = list_p[indices_now][1]+begin2;
			info->left = left2;
			info->right = right2;
			info->len = (len1+1)>>2;
			info->len_opposite = (len2+1)>>2;
		}
		else{
			// B ---> A
			info->B2A = true;
			info->addr = list_p[indices_now][1]+begin2;
			info->opposite_addr = list_p[indices_now][0]+begin1;
			info->left = left1;
			info->right = right1;
			info->len = (len2+1)>>2;
			info->len_opposite = (len1+1)>>2;
		}

		info->warp_len = info->len - info->len % WARP_SIZE;
		_scan_buffers[indices_now][whole_id] = 0;

		if ( info->len % WARP_SIZE ){
			info->warp_len += WARP_SIZE;
		}

		syncthreads();
		if (( whole_id == 2*parts - 1 && shared_decide_next_addr[ whole_id ] == 1 ) ||
			( shared_decide_next_addr[ whole_id ] && !shared_decide_next_addr[whole_id+1] ) ){
			int begin_new1 = (shared[0][whole_id]+1 )& ALIGN_MOD;
			int begin_new2 = (shared[1][whole_id]+1 )& ALIGN_MOD;
			swapped[ indices_next ] = 0;
			swapped[ indices_now ] = 0;

			list_p[indices_next][ 0 ] = list_p[indices_now][ 0 ] + shared[0][whole_id]+1 - begin_new1;
			list_p[indices_next][ 1 ] = list_p[indices_now][ 1 ] + shared[1][whole_id]+1 - begin_new2;
			calculated_indices_len[indices_next][0] = begin_new1; //changed meaning
			calculated_indices_len[indices_next][1] = begin_new2; //changed meaning

			//printf("Next real start: [%d]:%d [%d]:%d\n",begin_new1,list_p[indices_next][0][begin_new1],begin_new2,list_p[indices_next][1][begin_new2]);
		}

	}

	__device__ void cal_indx_3(int block_size,int indices_now){
		int id = threadIdx.x;
		int side = threadIdx.y;
		int n = blockDim.x;
		int part_size = block_size / n;
		int x,y;
		int left = 0,right;
		int *myList = list_p[indices_now][ 0];
		int *oppositeList = list_p[indices_now][ 1];

		_scan_buffers[indices_now][threadIdx.x+threadIdx.y*blockDim.x] = 0;

		if ( 0 == id && 1 == side ) return;

		if ( 1 == side){
			x = block_size-1;
			y = id * part_size ;
			right = (n-id)*part_size;
		}
		else {
			x = (id+1) * part_size-1;
			y = 0;
			right = (id+1)*part_size;
		}

		//printf("<%d %d> x:%d y:%d  value1:%d,%d\n",side,id,x,y,myList[x],oppositeList[y]);

		int x2,y2;
		while (left<right){
			int mid = (left + right + 1)/2;
			x2 =  x - mid;//hash back
			y2 =  y + mid;
			if (  myList[x2] < oppositeList[y2] )
				right = mid - 1;
			else left = mid;
		}

		x2 =  x - left;//hash back
		y2 =  y + left;

		//printf("<%d %d> found %d <%d,%d> (%d,%d)\n",side,id,left,x2,y2,myList[x2] ,oppositeList[y2] );

		if ( x2 >=0  ){
			if (myList[x2] > oppositeList[y2] ){
				x2 --;
			}
//			else if ((myList[x2] < oppositeList[y2] ))
//				y2 --;
		}

		//printf("<%d %d> found %d <%d,%d> (%d,%d)\n",side,id,left,x2,y2,myList[x2] ,oppositeList[y2] );
		__shared__ volatile int shared[2][2*DEF_D1+1];
		int whole_id = id;
		if ( side == 1 )
			whole_id = id-1 + blockDim.x;
		shared[0][whole_id+1] = x2;
		shared[1][whole_id+1] = y2;
		if ( 0 == whole_id &&  0 == blockIdx.x)
			shared[0][0] = -1,shared[1][0] = -1;
		syncthreads();

//		if ( IS_FIRST){
//			printf("[%d]:%d ==== [%d]:%d\n",0,myList[0],0,oppositeList[0]);
//			FOR_I(1,part_size){
//				int a = shared[0][ i ];
//				int b = shared[1][ i ];
//				printf("[%d]:%d --- [%d]:%d\n",a,myList[a],b,oppositeList[b]);
//			}
//		}

		__shared__ volatile int shared_decide_next_addr[2*DEF_D1+1];//default may be not 0!!

		if ( shared[0][whole_id+1] == shared[0][whole_id] ){
			shared_decide_next_addr[whole_id] = -1;//vertical line
		}
		else if ( shared[1][whole_id+1] == shared[1][whole_id] ){
			shared_decide_next_addr[whole_id] = 1; //line
		}
		else shared_decide_next_addr[whole_id] = 0; //OK
		int indices_next = (indices_now + 1) % QUEUE_SIZE;


		int begin1,end1,begin2,end2,len1,len2;
		if ( whole_id == 0 ){

			begin1 = calculated_indices_len[indices_now][0];
			begin2 = calculated_indices_len[indices_now][1];
			//begin1 = 0;
			//begin2 = 0;
		}
		else {
			begin1 = shared[0][whole_id]+1;
			begin2 = shared[1][whole_id]+1;
		}
		int left1,left2,right1,right2;
		end1 = shared[0][whole_id+1], end2 = shared[1][whole_id+1];

//		printf("wid: wholeID %d <%d %d> begin [%d]%d <---> [%d]%d\n",whole_id,threadIdx.y,threadIdx.x,begin1,myList[begin1],begin2,oppositeList[begin2]);
//		printf("wid: wholeID %d <%d %d> end [%d]%d <---> [%d]%d\n",whole_id,threadIdx.y,threadIdx.x,end1,myList[end1],end2,oppositeList[end2]);

		len1 = end1 - begin1+1;
		len2 = end2 - begin2+1;

		make_memory_alignment(list_p[indices_now][ 0 ],list_p[ indices_now ][ 1 ],begin1, begin2,end1 ,end2, left1, left2,right1, right2,len1, len2);

		if ( 0 != shared_decide_next_addr[whole_id] ){
			len1 = len2 = -999;//do not do calculation
		}
		struct partition_info *info = & partitions_info[indices_now][whole_id];
		if ( len1 <= len2 ){
			// A ---> B
			info->B2A = false;
			info->addr = list_p[indices_now][0]+begin1;
			info->opposite_addr = list_p[indices_now][1]+begin2;
			info->left = left2;
			info->right = right2;
			info->len = (len1+1)>>2;
			info->len_opposite = (len2+1)>>2;

			//info->len = len1;

		}
		else{
			// B ---> A
			info->B2A = true;
			info->addr = list_p[indices_now][1]+begin2;
			info->opposite_addr = list_p[indices_now][0]+begin1;
			info->left = left1;
			info->right = right1;
			info->len = (len2+1)>>2;
			info->len_opposite = (len1+1)>>2;


			//info->len = len2;

		}

		info->warp_len = info->len - info->len % WARP_SIZE;


		if ( info->len % WARP_SIZE ){
			info->warp_len += WARP_SIZE;
		}


		__shared__ int ending;
		if ( whole_id == 0 ){
			int start = shared_decide_next_addr[blockDim.x*2-2];
			if ( 0 == start ){
				ending = blockDim.x*2-3;
			}
			else {
				for (int i = blockDim.x*2-3;i>=0;i--)
					if ( start != shared_decide_next_addr[i] ){
						ending = i;
						break;
					}
			}
		}

		syncthreads();
		if ( ending+1 == whole_id ){

			int begin_new1 = (shared[0][whole_id]+1 )& ALIGN_MOD;
			int begin_new2 = (shared[1][whole_id]+1 )& ALIGN_MOD;
			swapped[ indices_next ] = 0;
			swapped[ indices_now ] = 0;

			list_p[indices_next][ 0 ] = list_p[indices_now][ 0 ] + shared[0][whole_id]+1 - begin_new1;
			list_p[indices_next][ 1 ] = list_p[indices_now][ 1 ] + shared[1][whole_id]+1 - begin_new2;
			calculated_indices_len[indices_next][0] = begin_new1; //changed meaning
			calculated_indices_len[indices_next][1] = begin_new2; //changed meaning
			//printf("movement: %d,%d\n",shared[0][whole_id]+1,shared[1][whole_id]+1);

			//printf("Next real start: [%d]:%d [%d]:%d\n",begin_new1,list_p[indices_next][0][begin_new1],begin_new2,list_p[indices_next][1][begin_new2]);
		}
//		if ( whole_id > ending +1 )
//			info->len = -999;


	}

	__global__ void cal_indx (int block_size,int block_2_size,int indices_now){
		//cal_indx_1(block_size,block_2_size,indices_now);
		int n = blockDim.x;
		cal_indx_2(n,block_size/n,block_size,indices_now);
		//cal_indx_3(block_size,indices_now);
	}



	//special case for cal_indices
	__global__ void move_indices(int len1,int len2,int block_size,int indices_now){
		int indices_next = (indices_now + 1) % QUEUE_SIZE;
		int swapflag = 0;
		//make sure that len1 is always the shorter one
		if ( len1 > len2 ){
			int *temp = list_p[indices_now][0];
			list_p[indices_now][0] = list_p[indices_now][1];
			list_p[indices_now][1] = temp;
			int temp2 = len1;
			len1 = len2;
			len2 = temp2;
			swapflag = 1;
		}
		int b1 = min(len1,block_size);
		list_p[ indices_next ][ 0 ] = list_p[ indices_now ][ 0 ] + b1;
		list_p[ indices_next ][ 1 ] = list_p[ indices_now ][ 1 ] ;//keep the same
		swapped[ indices_next ] = swapped[ indices_now ] ^ swapflag;
		calculated_indices_len[ indices_now ][ 0 ] = b1-1;// or b1
		calculated_indices_len[ indices_now ][ 1 ] = len2-1;// or b1
	}

