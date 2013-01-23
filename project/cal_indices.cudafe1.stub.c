#define __NV_MODULE_ID _19_cal_indices_cpp1_ii_2e93a852
#define __NV_CUBIN_HANDLE_STORAGE__ extern
#include "crt/host_runtime.h"
#include "cal_indices.fatbin.c"
extern void __device_stub__Z8cal_indxiii(int, int, int);
extern void __device_stub__Z12move_indicesiiii(int, int, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll_19_cal_indices_cpp1_ii_2e93a852(void) __attribute__((__constructor__));
void __device_stub__Z8cal_indxiii(int __par0, int __par1, int __par2){__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 4UL);__cudaSetupArgSimple(__par2, 8UL);__cudaLaunch(((char *)((void ( *)(int, int, int))cal_indx)));}
# 411 "../src/indices/cal_indices.cu"
void cal_indx( int __cuda_0,int __cuda_1,int __cuda_2)
# 411 "../src/indices/cal_indices.cu"
{__device_stub__Z8cal_indxiii( __cuda_0,__cuda_1,__cuda_2);




}
# 1 "cal_indices.cudafe1.stub.c"
void __device_stub__Z12move_indicesiiii( int __par0,  int __par1,  int __par2,  int __par3) {  __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 4UL); __cudaSetupArgSimple(__par2, 8UL); __cudaSetupArgSimple(__par3, 12UL); __cudaLaunch(((char *)((void ( *)(int, int, int, int))move_indices))); }
# 421 "../src/indices/cal_indices.cu"
void move_indices( int __cuda_0,int __cuda_1,int __cuda_2,int __cuda_3)
# 421 "../src/indices/cal_indices.cu"
{__device_stub__Z12move_indicesiiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 440 "../src/indices/cal_indices.cu"
}
# 1 "cal_indices.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T21) {  __nv_dummy_param_ref(__T21); __cudaRegisterEntry(__T21, ((void ( *)(int, int, int, int))move_indices), _Z12move_indicesiiii, (-1)); __cudaRegisterEntry(__T21, ((void ( *)(int, int, int))cal_indx), _Z8cal_indxiii, (-1)); }
static void __sti____cudaRegisterAll_19_cal_indices_cpp1_ii_2e93a852(void) {  ____cudaRegisterLinkedBinary(__nv_cudaEntityRegisterCallback);  }
