#ifndef H_T_D
#define H_T_D cudaMemcpyHostToDevice
#endif

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5
#define LOG_HALFWARP_SIZE 4
#define SCAN_BLOCK_SIZE 1024
#define LOG_SCAN_BLOCK_SIZE 10

void scan(int *,int );
void scan2(int *,int );
void init_scan(cudaStream_t * ,int );
