#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <fstream>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

using namespace std;
#define MAX_NUM_LISTS 256
#define BlockSize 1024 
#define MAX_ELEMENTS_PER_BLOCK 1024
#define Way_N 4
cudaEvent_t start, stop;
float *Data,*dev_srcData,*dev_dstData;
unsigned int* dev_BlockSum,* dev_BlockSum_Prefix,*dev_BlockSum_Status,*local_prefix0,*local_prefix1,*local_prefix2,*local_prefix3;

// radix sort v1 : intra block radix sort
__device__ unsigned int BLOCK_PREFIX_COUNTER,BLOCK_PREFIX_COMPLETE;

__global__ void Init_Flag(){
    if(threadIdx.x==0){
        BLOCK_PREFIX_COUNTER=0;
        BLOCK_PREFIX_COMPLETE=0;
    }
}
__global__ void Data_Preprocess(float* src_data,int num_data);
__global__ void Data_Postprocess(float* src_data,int num_data);

// radix sort v3 : block radix sort and block prefix sum
__device__ void ScanWarp(unsigned int* sData,unsigned int lane){
    // Kogge-Stone
    if(lane>=1)
        sData[lane]+=sData[lane-1];
    __syncwarp();
    if(lane>=2)
        sData[lane]+=sData[lane-2];
     __syncwarp();
    if(lane>=4)
        sData[lane]+=sData[lane-4];
    __syncwarp();
    if(lane>=8)
        sData[lane]+=sData[lane-8];
    __syncwarp();
    if(lane>=16)
        sData[lane]+=sData[lane-16];
    __syncwarp();
}

__device__ void ScanBlock(unsigned int* sData){
    unsigned int warp_id=threadIdx.x>>5; // each warp has 32 thread
    unsigned int lane = threadIdx.x & 31;  // 31 = 00011111 (i.e. mod 31)
    __shared__  unsigned int warp_sum[32];  // block size 1024 / warp size 32 = 32 
    ScanWarp(sData+(warp_id<<5),lane);
    __syncthreads();
    if(lane==31){
        warp_sum[warp_id]=sData[threadIdx.x];
    }
    __syncthreads();
    // Reduce-then-scan
    if(warp_id==0){
        ScanWarp(warp_sum,lane);
    }
    __syncthreads();
    if(warp_id>0){
        *(sData+threadIdx.x)+=warp_sum[warp_id-1];
    }
    __syncthreads();
}

__global__ void Radix_GSum_LPrefix(unsigned int* src_data,unsigned int* local_prefix0,unsigned int* local_prefix1,unsigned int* local_prefix2,unsigned int* local_prefix3,unsigned int* dev_BlockSum,int num_data,int PASS){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    if(tid>= num_data) return ;

    unsigned int bit_mask_0=(0 << (PASS*2));
    unsigned int bit_mask_1=(1 << (PASS*2));
    unsigned int bit_mask_2=(2 << (PASS*2));    
    unsigned int bit_mask_3=(3 << (PASS*2));
    unsigned int bit_value;
    int BlockNum=(num_data+BlockSize-1)/BlockSize;

    unsigned int block_element_num=(blockIdx.x+1)*BlockSize<=num_data?BlockSize:num_data%BlockSize;
    __shared__ unsigned int sData[BlockSize];
    __shared__ unsigned int FalseBuffer_0[BlockSize+1]; // e0 buffer
    __shared__ unsigned int FalseBuffer_1[BlockSize+1]; // e1 buffer
    __shared__ unsigned int FalseBuffer_2[BlockSize+1]; // e2 buffer
    __shared__ unsigned int FalseBuffer_3[BlockSize+1]; // e3 buffer
    
    __shared__ unsigned int PrefixFalseBuffer_0[BlockSize+1]; // f0 buffer
    __shared__ unsigned int PrefixFalseBuffer_1[BlockSize+1]; // f1 buffer
    __shared__ unsigned int PrefixFalseBuffer_2[BlockSize+1]; // f2 buffer
    __shared__ unsigned int PrefixFalseBuffer_3[BlockSize+1]; // f3 buffer

    sData[threadIdx.x]=((unsigned int*)src_data)[tid]; 
    __syncthreads();
    unsigned int temp_bit_value;
    temp_bit_value=sData[threadIdx.x]&bit_mask_3;
    if((temp_bit_value)==bit_mask_0){
        bit_value=0;
    }else if((temp_bit_value)==bit_mask_1){
        bit_value=1;
    }else if((temp_bit_value)==bit_mask_2){
        bit_value=2;
    }else if((temp_bit_value)==bit_mask_3){
        bit_value=3;
    }
    // compute False potision
    FalseBuffer_0[threadIdx.x]=(bit_value == 0)?1:0;
    FalseBuffer_1[threadIdx.x]=(bit_value == 1)?1:0;
    FalseBuffer_2[threadIdx.x]=(bit_value == 2)?1:0;
    FalseBuffer_3[threadIdx.x]=(bit_value == 3)?1:0;
    __syncthreads();
    
    PrefixFalseBuffer_0[threadIdx.x+1]=FalseBuffer_0[threadIdx.x];
    PrefixFalseBuffer_1[threadIdx.x+1]=FalseBuffer_1[threadIdx.x];
    PrefixFalseBuffer_2[threadIdx.x+1]=FalseBuffer_2[threadIdx.x];
    PrefixFalseBuffer_3[threadIdx.x+1]=FalseBuffer_3[threadIdx.x];
    __syncthreads();
    ScanBlock(PrefixFalseBuffer_0+1);// The last one is total false
    ScanBlock(PrefixFalseBuffer_1+1);// The last one is total false
    ScanBlock(PrefixFalseBuffer_2+1);// The last one is total false
    ScanBlock(PrefixFalseBuffer_3+1);// The last one is total false
    __syncthreads();

    // save to global prefixFalseBuffer
    local_prefix0[tid]=PrefixFalseBuffer_0[threadIdx.x];
    local_prefix1[tid]=PrefixFalseBuffer_1[threadIdx.x];
    local_prefix2[tid]=PrefixFalseBuffer_2[threadIdx.x];
    local_prefix3[tid]=PrefixFalseBuffer_3[threadIdx.x];

    if(threadIdx.x==0){
        dev_BlockSum[blockIdx.x+0*BlockNum]=PrefixFalseBuffer_0[block_element_num];
        dev_BlockSum[blockIdx.x+1*BlockNum]=PrefixFalseBuffer_1[block_element_num];
        dev_BlockSum[blockIdx.x+2*BlockNum]=PrefixFalseBuffer_2[block_element_num];
        dev_BlockSum[blockIdx.x+3*BlockNum]=PrefixFalseBuffer_3[block_element_num];
    }
    __threadfence();
    __syncthreads();
}

__global__ void BLOCK_PREFIX(unsigned int* dev_BlockSum,unsigned int* dev_BlockSum_Prefix , int num_data){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    if(tid>= num_data) return ;
    for(int i=1;i<num_data;i++){
        dev_BlockSum_Prefix[i]=dev_BlockSum_Prefix[i-1]+dev_BlockSum[i-1];
        __threadfence();
    }
}

__global__ void Reorder(unsigned int* src_data,unsigned int* BlockSum_Prefix,unsigned int* local_prefix0,unsigned int* local_prefix1,unsigned int* local_prefix2,unsigned int* local_prefix3,unsigned int* dev_BlockSum,int num_data,int PASS){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    if(tid>= num_data) return;
    
    unsigned int bit_mask_0=(0 << (PASS*2));
    unsigned int bit_mask_1=(1 << (PASS*2));
    unsigned int bit_mask_2=(2 << (PASS*2));    
    unsigned int bit_mask_3=(3 << (PASS*2));
    
    int BlockNum=(num_data+BlockSize-1)/BlockSize;
    
    __shared__ unsigned int sData[BlockSize];
    __shared__ unsigned int GlobalPosition[BlockSize];
    sData[threadIdx.x]=((unsigned int*)src_data)[tid]; 
    unsigned int temp_bit_value;
    temp_bit_value=sData[threadIdx.x]&bit_mask_3;
    // // compute the global position
    if(temp_bit_value==bit_mask_0){
        GlobalPosition[threadIdx.x]=BlockSum_Prefix[blockIdx.x+0*BlockNum]+local_prefix0[(blockIdx.x*BlockSize)+threadIdx.x];
    }else if(temp_bit_value==bit_mask_1){
        GlobalPosition[threadIdx.x]=BlockSum_Prefix[blockIdx.x+1*BlockNum]+local_prefix1[(blockIdx.x*BlockSize)+threadIdx.x];
    }else if(temp_bit_value==bit_mask_2){
        GlobalPosition[threadIdx.x]=BlockSum_Prefix[blockIdx.x+2*BlockNum]+local_prefix2[(blockIdx.x*BlockSize)+threadIdx.x];
    }else if(temp_bit_value==bit_mask_3){
        GlobalPosition[threadIdx.x]=BlockSum_Prefix[blockIdx.x+3*BlockNum]+local_prefix3[(blockIdx.x*BlockSize)+threadIdx.x];
    }
    __threadfence();
    __syncthreads();
    // //re order the element 
    src_data[GlobalPosition[threadIdx.x]]=(sData)[threadIdx.x];
}

// Auxiliariy function
void input(char* FIN,int N);
void output(char* FOUT,int N);
void SHOW(int N);
void SHOW_BIN(int N);
bool EVALUATE(int N);

void SHOW_DEV_BUFFER(void* A,int N){
    int* BUF=new int[N];
    cudaMemcpy(BUF, A, sizeof(float)*N, cudaMemcpyDeviceToHost);
    for(int i=0;i<N;i++){
        cout << BUF[i]<<" ";
    }
    cout <<endl;
    delete BUF;
};

void SHOW_DEV_BUFFER_BIN(void* A,int N){
    cout << "======== Current Data BINARY =======\n";
    int* BUF=new int[N];
    cudaMemcpy(BUF, A, sizeof(float)*N, cudaMemcpyDeviceToHost);
    for(int i=0;i<N;i++){
        for(int k=0;k<32;k++){
            cout << ((((unsigned int *)BUF)[i]&(0x80000000>>k))?"1":"0");
        }
        cout << endl;
    }
    cout <<endl;
    delete BUF;
    cout << "\n==================================\n";
};



int main(int argc, char **argv){
    if(argc!=4){
        cout << "The input format must be ./program N input_data output_path" <<endl;
    }
    
    int N=atoi(argv[1]);
    char* FIN=argv[2];
    char* FOUT=argv[3];
    float   elapsedTime;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    // input 
    Data=new float[N];
    input(FIN,N);
    // FILE* file = fopen(FIN, "rb");
    // fread(Data,sizeof(float)*N,N,file);
    // fclose(file);

    // show the init data
    printf("The INITIAL DATA\n");
    SHOW(N);
    SHOW_BIN(N);

    // cuda preprocess
    int BlockNum=(N+BlockSize-1)/BlockSize;
    cudaMalloc((void**)&dev_srcData,sizeof(float)*N);
    cudaMalloc((void**)&dev_dstData,sizeof(float)*N);
    cudaMalloc((void**)&dev_BlockSum,sizeof(unsigned int)*Way_N*BlockNum);
    cudaMalloc((void**)&dev_BlockSum_Status,sizeof(unsigned int)*(Way_N*BlockNum+1));
    cudaMalloc((void**)&dev_BlockSum_Prefix,sizeof(unsigned int)*(Way_N*BlockNum+1));
    
    cudaMalloc((void**)&local_prefix0,sizeof(unsigned int)*(N+1));
    cudaMalloc((void**)&local_prefix1,sizeof(unsigned int)*(N+1));
    cudaMalloc((void**)&local_prefix2,sizeof(unsigned int)*(N+1));
    cudaMalloc((void**)&local_prefix3,sizeof(unsigned int)*(N+1));
    cudaMemcpy(dev_srcData, Data, sizeof(float)*N, cudaMemcpyHostToDevice);    
    
    // radix sort
    cudaEventRecord( start, 0 );
    Data_Preprocess<<<BlockNum,BlockSize>>>(dev_srcData,N);    
    cudaMemcpy(Data, dev_srcData, sizeof(float)*N, cudaMemcpyDeviceToHost);
    
    for(int i=0;i<16;i++){

        // compute global sum and the local_prefix
        Radix_GSum_LPrefix<<<BlockNum,BlockSize>>>((unsigned int*)dev_srcData,local_prefix0,local_prefix1,local_prefix2,local_prefix3,dev_BlockSum,N,i);
        cout << "local_prefix0\n";
        SHOW_DEV_BUFFER(local_prefix0,N);
        cout << "local_prefix1\n";
        SHOW_DEV_BUFFER(local_prefix1,N);
        cout << "local_prefix2\n";
        SHOW_DEV_BUFFER(local_prefix2,N);
        cout << "local_prefix3\n";
        SHOW_DEV_BUFFER(local_prefix3,N);
        cout << "BLOCK SUM\n";
        SHOW_DEV_BUFFER(dev_BlockSum,Way_N*BlockNum);
        
        // compute global prefix sum
        BLOCK_PREFIX<<<1,1>>>(dev_BlockSum,dev_BlockSum_Prefix ,Way_N*BlockNum);
        cout << "BLOCK PREFIX SUM\n";
        SHOW_DEV_BUFFER(dev_BlockSum_Prefix,Way_N*BlockNum);

        // reorder
        Reorder<<<BlockNum,BlockSize>>>((unsigned int*)dev_srcData, dev_BlockSum_Prefix, local_prefix0,local_prefix1,local_prefix2,local_prefix3,dev_BlockSum,N,i);
        
        cout << "ReOrder Data\n";
        SHOW_DEV_BUFFER_BIN(dev_srcData,N);

        printf("The PASS %d RESULT\n",i);
        SHOW_BIN(N);
        
        cudaThreadSynchronize();
    }
    Data_Postprocess<<< BlockNum,BlockSize>>>(dev_srcData,N);
    cudaEventRecord( stop, 0 ) ;
    cudaEventSynchronize( stop );

    // cuda postprocess
    cudaMemcpy(Data, dev_srcData, sizeof(float)*N, cudaMemcpyDeviceToHost);
    // output
    cudaFree(dev_srcData);
    cudaFree(dev_dstData);
    cudaFree(dev_BlockSum);
    cudaFree(dev_BlockSum_Status);
    cudaFree(dev_BlockSum_Prefix);
    cudaFree(local_prefix0);
    cudaFree(local_prefix1);
    cudaFree(local_prefix2);
    cudaFree(local_prefix3);
    // SHOW(N);
    output(FOUT,N);

    // Test the correstness and output the running time
    cudaEventElapsedTime( &elapsedTime,start, stop );
    cout << "The Evaluation STATUS \n";
    cout << "==================================\n";
    cout << "CORRECTNESS : "<< (EVALUATE(N)==true?"PASS":"FAIL")<<"\nCOST TIME : "<< elapsedTime <<endl;
    cout << "==================================\n";
    return 0;
}

void SHOW(int N){
    cout << "========== Current Data ==========\n";
    for(int i=0;i<N;i++){
        cout << setw(8) <<Data[i]<<" ";
    }
    cout << "\n==================================\n";
}

void SHOW_BIN(int N){
    
    cout << "======== Current Data BINARY =======\n";
    for(int i=0;i<N;i++){
        for(int k=0;k<32;k++){
            cout << ((((unsigned int *)Data)[i]&(0x80000000>>k))?"1":"0");
        }
        cout << endl;
    }
    cout << "\n==================================\n";
}

void input(char* FIN,int N){
    FILE* file = fopen(FIN, "rb");
    fread(Data,sizeof(float)*N,N,file);
    fclose(file);
}

void output(char* FOUT,int N){
    FILE* file = fopen(FOUT, "w");
    fwrite(Data,sizeof(float)*N,N,file);
    fclose(file);
}

bool EVALUATE(int N){
    bool PASS=true;
    float prev=Data[0];
    for(int i =0;i<N;i++)
    {
        if(prev>Data[i]){
            PASS=false;
        }
        prev=Data[i];
    }
    return PASS;
}


__global__ void Data_Preprocess(float* src_data,int num_data){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    if(tid>=num_data) return ;
    unsigned int data_temp=((unsigned int*)src_data)[tid];
    ((unsigned int*)src_data)[tid]=(( (data_temp>>31) & 0x1) ? ~(data_temp) : ((data_temp)|0x80000000));
}
__global__ void Data_Postprocess(float* src_data,int num_data){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    if(tid>=num_data) return;
    unsigned int data_temp=((unsigned int*)src_data)[tid];
    ((unsigned int*)src_data)[tid]=(((data_temp)>>31 & 0x1) ? ((data_temp) & 0x7fffffff) : ~(data_temp));
}

__device__ void SHOW_BUFFER(void* Buffer,int size){
    if(threadIdx.x==0){
        for(int i=0;i<size;i++){
            printf("%d ",((unsigned int*)Buffer)[i]);
        }
        printf("\n");
    }
}
__device__ void SHOW_BUFFER_BIN(void* Buffer,int size){
    printf("======== Current Data BINARY =======\n");
    for(int i=0;i<size;i++){
        for(int k=0;k<32;k++){
            printf("%c", ((((unsigned int *)Buffer)[i]&(0x80000000>>k))?'1':'0'));
        }
        printf("\n");
    }
    printf("\n==================================\n");
}