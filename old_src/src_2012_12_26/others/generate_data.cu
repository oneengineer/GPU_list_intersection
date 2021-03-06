#include "../common_defines.h"
#include <algorithm>
#include "./memory_data.cuh"
#include "generate_data.h"
#include <cstdio>

using namespace std;

	void generate_different(int * array,int n,int mod){
		FOR_I(0,mod) v[i] = false;
		for (int i=0;i<n;i++){
			int j=-1;
			do
			{
				j = rand() % mod;
			}while ( v[j] );
			v[j] = true;
			array[i] = j;
		}
	}

	void generate_random(double alpha,double scala1,double scala2){
		printf("alpha:%lf  scala1:%lf  scala2:%lf\n",alpha,scala1,scala2);//debug
		int mod = n*4;
		m = (int)((double)n*alpha + eps);
		int mod1 = (int)((double)mod*scala1 + eps);
		int mod2 = (int)((double)mod*scala2 + eps);
		generate_different(host_lists[0],n,mod1);
		sort(host_lists[0],host_lists[0]+n);
		generate_different(host_lists[1],m,mod2);
		sort(host_lists[1],host_lists[1]+m);
	}

	void generate_case2(){
		printf("a general test. mainly for the search bound\nPlease use D1 = D2 = 2,block2size = 5\n\n");
		int temp1[] = {1,2,3,4,           20,            29, 34,36,37,38 ,39,30,66
					,100,110,120,130,140,200,211,230,231,540 };
		int temp2[] = {1,2,3,4,6,7,10,12, 20,25,26,27,28 ,29                   ,60
				,100,110,120,130,140,200,210,220,230,540};

		n = sizeof(temp1)/sizeof(int);
		m = sizeof(temp2)/sizeof(int);
		FOR_I(0,n) host_lists[0][i] = temp1[i];
		FOR_I(0,m) host_lists[1][i] = temp2[i];
	}

	void generate_case3(){
		printf("a generate test. mainly for the search swap\nPlease use D1 = 1; D2 = 3,block2size = 5\n\n");
		int temp1[] = {1,  2,  9,   11,15,16,17,26,27
					,100,110,120,130,140,200,211,230,231,540 };
		int temp2[] = {1,2,5,6,9,          21,25,  27
					,100,110,120,130,140,200,210,220,230,540 };

		n = sizeof(temp1)/sizeof(int);
		m = sizeof(temp2)/sizeof(int);
		FOR_I(0,n) host_lists[0][i] = temp1[i];
		FOR_I(0,m) host_lists[1][i] = temp2[i];
	}

	void generate_case4(){
		printf("a end point test. Please use D1 = 2; D2 = 2,block2size = 6\n\n");
		int temp1[] = {196};
		int temp2[] = {195,196,200};

		n = sizeof(temp1)/sizeof(int);
		m = sizeof(temp2)/sizeof(int);
		FOR_I(0,n) host_lists[0][i] = temp1[i];
		FOR_I(0,m) host_lists[1][i] = temp2[i];
	}

	void generate_case5(){
		printf(" a total equal test \n\n");
		n = 64;
		m = 64;
		FOR_I(0,n) host_lists[0][i] = host_lists[1][i] = i*10 + rand() % 5;
	}
