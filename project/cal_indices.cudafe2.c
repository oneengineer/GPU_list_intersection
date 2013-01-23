# 1 "cal_indices.cudafe1.gpu"
# 56 "../src/indices/../common_defines.h"
struct partition_info;
# 77 "../src/indices/../common_defines.h"
struct debug_structure;
# 1 "cal_indices.cudafe1.gpu"
typedef char __nv_bool;
# 211 "/usr/lib/gcc/x86_64-linux-gnu/4.4.3/include/stddef.h" 3
typedef unsigned long size_t;
#include "crt/host_runtime.h"
# 56 "../src/indices/../common_defines.h"
struct partition_info {
# 57 "../src/indices/../common_defines.h"
int *addr;
# 57 "../src/indices/../common_defines.h"
int *opposite_addr;
# 58 "../src/indices/../common_defines.h"
int len;
# 58 "../src/indices/../common_defines.h"
int len_opposite;
# 58 "../src/indices/../common_defines.h"
int warp_len;
# 59 "../src/indices/../common_defines.h"
int left;
# 59 "../src/indices/../common_defines.h"
int right;
# 60 "../src/indices/../common_defines.h"
__nv_bool B2A; char __nv_no_debug_dummy_end_padding_0[3];};
# 77 "../src/indices/../common_defines.h"
struct debug_structure {
# 78 "../src/indices/../common_defines.h"
int num_loop;
# 79 "../src/indices/../common_defines.h"
int wrong_1;
# 80 "../src/indices/../common_defines.h"
int wrong_2;};
void *memcpy(void*, const void*, size_t); void *memset(void*, int, size_t);

#include "cal_indices.cudafe2.stub.c"
