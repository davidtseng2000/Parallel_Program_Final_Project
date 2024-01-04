#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <fstream>
// #include <cub/cub.cuh>

using namespace std;
#define MAX_NUM_LISTS 256
#define BlockSize 1024
cudaEvent_t start, stop;
float *Data,*dev_srcData,*dev_dstData;


// radix sort
__global__ void GPU_radix_sort(float* src_data, float*  dest_data, \
    int num_lists, int num_data);
__device__ void radix_sort(float*  data_0, float*  data_1, \
    int num_lists, int num_data, int tid); 
__device__ void merge_list( float* src_data, float*  dest_list, \
    int num_lists, int num_data, int tid); 
__device__ void preprocess_float(float*  data, int num_lists, int num_data, int tid);
__device__ void Aeprocess_float(float*  data, int num_lists, int num_data, int tid);

// radix sort v1 : intra block radix sort
__global__ void Intra_Block_radix_sort(float* src_data,float* dest_data,int num_data);
__global__ void Data_Preprocess(float* src_data,int num_data);
__global__ void Data_Postprocess(float* src_data,int num_data);

// Auxiliariy function
void INPUT(char* FIN,int N);
void OUTPUT(char* FOUT,int N);
void SHOW(int N);
void SHOW_BIN(int N);
bool EVALUATE(int N);

int main(int argc, char **argv){
    if(argc!=4){
        cout << "The input format must be ./program N input_data output_path" <<endl;
    }
    
    int N=atoi(argv[1]);
    char* FIN=argv[2];
    char* FOUT=argv[3];
    float elapsedTime;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    // input 
    Data=new float[N];
    INPUT(FIN,N);
    // SHOW(N);
    // SHOW_BIN(N);
    // cuda preprocess
    cudaMalloc((void**)&dev_srcData,sizeof(float)*N);
    cudaMalloc((void**)&dev_dstData,sizeof(float)*N);
    cudaMemcpy(dev_srcData, Data, sizeof(float)*N, cudaMemcpyHostToDevice);

    // int num_lists = 128; // the number of parallel threads
    // radix sort
    cudaEventRecord( start, 0 ) ;
    // GPU_radix_sort<<<1,num_lists>>>(dev_srcData, dev_dstData, num_lists, N);
    Data_Preprocess<<< (N+BlockSize-1)/BlockSize,BlockSize>>>(dev_srcData,N);
    Intra_Block_radix_sort<<< (N+BlockSize-1)/BlockSize,BlockSize>>>(dev_srcData,dev_dstData,N);
    Data_Postprocess<<< (N+BlockSize-1)/BlockSize,BlockSize>>>(dev_dstData,N);
    cudaEventRecord( stop, 0 ) ;
    cudaEventSynchronize( stop );

    // cuda postprocess
    cudaMemcpy(Data, dev_dstData, sizeof(float)*N, cudaMemcpyDeviceToHost);
    // output
    cudaFree(dev_srcData);
    cudaFree(dev_dstData);
    SHOW(N);
    OUTPUT(FOUT,N);

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


void INPUT(char* FIN,int N){
    FILE* file = fopen(FIN, "rb");
    fread(Data,sizeof(float)*N,N,file);
    fclose(file);
}

void OUTPUT(char* FOUT,int N){
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

__global__ void GPU_radix_sort(float*  src_data, float*  dest_data, \
    int num_lists, int num_data) {
    // temp_data:temporarily store the data
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    // special preprocessing of IEEE floating-point numbers before applying radix sort
    preprocess_float(src_data, num_lists, num_data, tid); 
    __syncthreads();
   // no shared memory
    radix_sort(src_data, dest_data, num_lists, num_data, tid);
    __syncthreads();
    merge_list(src_data, dest_data, num_lists, num_data, tid);    
    __syncthreads();
    Aeprocess_float(dest_data, num_lists, num_data, tid);
    __syncthreads();
}

__device__ void preprocess_float(float*  src_data, int num_lists, int num_data, int tid){
    for(int i = tid;i<num_data;i+=num_lists)
    {
        unsigned int *data_temp = (unsigned int *)(&src_data[i]);    
        *data_temp = (*data_temp >> 31 & 0x1)? ~(*data_temp): ((*data_temp) | 0x80000000); 
    }
}

__device__ void Aeprocess_float(float*  data, int num_lists, int num_data, int tid){
    for(int i = tid;i<num_data;i+=num_lists)
    {
        unsigned int* data_temp = (unsigned int *)(&data[i]);
        *data_temp = (*data_temp >> 31 & 0x1)? (*data_temp) & 0x7fffffff: ~(*data_temp);
    }
}


__device__ void radix_sort(float*  data_0, float*  data_1, \
    int num_lists, int num_data, int tid) {
    for(int bit=0;bit<32;bit++)
    {
        int bit_mask = (1 << bit);
        int count_0 = 0;
        int count_1 = 0;
        for(int i=tid; i<num_data;i+=num_lists)
        {
            unsigned int *temp =(unsigned int *) &data_0[i];
            if(*temp & bit_mask)
            {
                data_1[tid+count_1*num_lists] = data_0[i];
                count_1 += 1;
            }
            else{
                data_0[tid+count_0*num_lists] = data_0[i];
                count_0 += 1;
            }
        }
        for(int j=0;j<count_1;j++)
        {
            data_0[tid + count_0*num_lists + j*num_lists] = data_1[tid + j*num_lists]; 
        }
    }
}

__device__ void merge_list( float* src_data, float*  dest_list, \
    int num_lists, int num_data, int tid) {
    int num_per_list = ceil((float)num_data/num_lists);
    __shared__ int list_index[MAX_NUM_LISTS];
    __shared__ float record_val[MAX_NUM_LISTS];
    __shared__ int record_tid[MAX_NUM_LISTS];
    list_index[tid] = 0;
    record_val[tid] = 0;
    record_tid[tid] = tid;
    __syncthreads();
    for(int i=0;i<num_data;i++)
    {
        record_val[tid] = 0;
        record_tid[tid] = tid;
        if(list_index[tid] < num_per_list)
        {
            int src_index = tid + list_index[tid]*num_lists;
            if(src_index < num_data)
            {
                record_val[tid] = src_data[src_index];
            }else{
                unsigned int *temp = (unsigned int *)&record_val[tid];
                *temp = 0xffffffff;
            }
        }else{
                unsigned int *temp = (unsigned int *)&record_val[tid];
                *temp = 0xffffffff;
        }
        __syncthreads();
        int tid_max = num_lists >> 1;
        while(tid_max != 0 )
        {
            if(tid < tid_max)
            {
                unsigned int* temp1 = (unsigned int*)&record_val[tid];
                unsigned int *temp2 = (unsigned int*)&record_val[tid + tid_max];
                if(*temp2 < *temp1)
                {
                    record_val[tid] = record_val[tid + tid_max];
                    record_tid[tid] = record_tid[tid + tid_max];
                }
            }
            tid_max = tid_max >> 1;
            __syncthreads();
        }
        if(tid == 0)
        {
            list_index[record_tid[0]]++;
            dest_list[i] = record_val[0];
        }
        __syncthreads();
    }
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


__device__ void SHOW_BUFFER(unsigned int* sData , int num_data){
    for(int i=0;i<num_data;i++){
        printf("%d ",sData[i]);
    }
    printf("\n");
}


__device__ void ScanWarp(unsigned int* sData,unsigned int lane){
    // if(lane==0){
    //     for(int i=1;i<32;i++){
    //         sData[i]+=sData[i-1];
    //     }
    // }

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

    // if(lane&0x00000001)
    //     sData[lane]+=sData[lane-1];
    // __syncwarp();
    // if(lane&0x00000011)
    //     sData[lane]+=sData[lane-2];
    // __syncwarp();
    // if(lane&0x00000111)
    //     sData[lane]+=sData[lane-1];
    // __syncwarp();
    
}
__device__ void ScanBlock(unsigned int* sData){
    unsigned int warp_id=threadIdx.x>>5; // each warp has 32 thread
    unsigned int lane = threadIdx.x & 31;  // 31 = 00011111 (i.e. mod 31)
    __shared__  unsigned int warp_sum[32];  // block size 1024 / warp size 32 = 32 
    ScanWarp(sData+(warp_id<<5),lane);
    __syncthreads();

    // Kogge-Stone
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

__global__ void Intra_Block_radix_sort(float* src_data,float* dest_data,int num_data){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    if(tid>= num_data) return ;
    // // load block data to share memory
    __shared__ unsigned int sData[BlockSize];
    __shared__ unsigned int tempData[BlockSize];

    __shared__ unsigned int FalseBuffer_0[BlockSize+1]; // e0 buffer
    __shared__ unsigned int FalseBuffer_1[BlockSize+1]; // e1 buffer
    __shared__ unsigned int FalseBuffer_2[BlockSize+1]; // e2 buffer
    __shared__ unsigned int FalseBuffer_3[BlockSize+1]; // e3 buffer

    __shared__ unsigned int PrefixFalseBuffer_0[BlockSize+1]; // f0 buffer
    __shared__ unsigned int PrefixFalseBuffer_1[BlockSize+1]; // f1 buffer
    __shared__ unsigned int PrefixFalseBuffer_2[BlockSize+1]; // f2 buffer
    __shared__ unsigned int PrefixFalseBuffer_3[BlockSize+1]; // f3 buffer

    __shared__ unsigned int Position[BlockSize];

    // unsigned int bit_mask_0=0;
    unsigned int bit_mask_1= 1;
    unsigned int bit_mask_2= 2;
    unsigned int bit_mask_3= 3;     

    sData[threadIdx.x]=((unsigned int*)src_data)[tid]; 

    __syncthreads();
    // 16 pass radix sort
    
    for(int i=0;i<16;i++){        
        // compte False potision  
        // bit_mask_3 必須先 check，否則 "3(11)" 可能被算到 "1(01)" 或 "2(10)"
        int num;
        if((sData[threadIdx.x]&bit_mask_3) == 0)
            num = 0;
        else if ((sData[threadIdx.x]&bit_mask_3) == bit_mask_3)
            num = 3;
        else if ((sData[threadIdx.x]&bit_mask_1) == bit_mask_1)
            num = 1;
        else if ((sData[threadIdx.x]&bit_mask_2) == bit_mask_2)
            num = 2;

        FalseBuffer_0[threadIdx.x]=(num == 0)?1:0;
        __syncthreads();
        FalseBuffer_1[threadIdx.x]=(num == 1)?1:0;
        __syncthreads();
        FalseBuffer_2[threadIdx.x]=(num == 2)?1:0;
        __syncthreads();
        FalseBuffer_3[threadIdx.x]=(num == 3)?1:0;
    
        
        __syncthreads();
        //prefix sum
        // if(tid==0){
        //     PrefixFalseBuffer[0]=0;
        //     for(int k=1;k<num_data;k++){
        //         PrefixFalseBuffer[k]=PrefixFalseBuffer[k-1]+FalseBuffer[k-1];
        //     }
        //     total_False=PrefixFalseBuffer[num_data-1]+FalseBuffer[num_data-1];
        // }
        // __syncthreads();
        PrefixFalseBuffer_0[tid+1]=FalseBuffer_0[tid];
        PrefixFalseBuffer_1[tid+1]=FalseBuffer_1[tid];
        PrefixFalseBuffer_2[tid+1]=FalseBuffer_2[tid];
        PrefixFalseBuffer_3[tid+1]=FalseBuffer_3[tid];
        __syncthreads();

        ScanBlock(PrefixFalseBuffer_0+1);// The last one is total false
        ScanBlock(PrefixFalseBuffer_1+1);// The last one is total false
        ScanBlock(PrefixFalseBuffer_2+1);// The last one is total false
        ScanBlock(PrefixFalseBuffer_3+1);// The last one is total false
        __syncthreads();

        // // compute position
        int num_0 = PrefixFalseBuffer_0[num_data];
        int num_01 = num_0 + PrefixFalseBuffer_1[num_data];
        int num_012 = num_01 + PrefixFalseBuffer_2[num_data];

        if(FalseBuffer_0[tid]){
            Position[tid]=PrefixFalseBuffer_0[tid];            
        }
        else if(FalseBuffer_1[tid]){
            Position[tid]= num_0 + PrefixFalseBuffer_1[tid]; // (0 有幾個) + (這是第幾個 1)
        }
        else if(FalseBuffer_2[tid]){
            Position[tid]= num_01 + PrefixFalseBuffer_2[tid]; // (0 有幾個) + (1 有幾個) + (這是第幾個 2)
        }
        else if(FalseBuffer_3[tid]){
            Position[tid]= num_012 + PrefixFalseBuffer_3[tid]; // (0 有幾個) + (1 有幾個) + (2 有幾個) + (這是第幾個 3)
        }
        __syncthreads();

        // scatter
        tempData[Position[tid]]=sData[tid];

        bit_mask_1<<=2;
        bit_mask_2<<=2;
        bit_mask_3<<=2;

        __syncthreads();
        // save data
        sData[threadIdx.x]=tempData[tid];
        __syncthreads();
    }
    
    dest_data[tid]=((float*)sData)[threadIdx.x];
    __syncthreads();
}