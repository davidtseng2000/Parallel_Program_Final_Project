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
// 第一種寫法: 全部 global data 分四塊 Block_sum (0~3)，每個長度是 num_block            
// unsigned int *block_sum_0, *block_sum_1, *block_sum_2, *block_sum_3;
// 第二種寫法: 全部 global data 只有一塊 Block_sum，長度是 num_block * 4，此寫法方便後面算 prefix_block_sum
unsigned int *block_sum, *prefix_block_sum;



// radix sort v1 : intra block radix sort
// __global__ void Intra_Block_radix_sort(float* src_data,float* dest_data,int num_data, \
//                                         unsigned int* block_sum_0, unsigned int* block_sum_1, unsigned int* block_sum_2, unsigned int* block_sum_3);
__global__ void Intra_Block_radix_sort(float* src_data,float* dest_data,int num_data, unsigned int* block_sum, unsigned int* prefix_block_sum, int num_block);
__global__ void Data_Preprocess(float* src_data, int num_data);
__global__ void Data_Postprocess(float* src_data, int num_data);

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

    int num_block = (N+BlockSize-1)/BlockSize;

    // input 
    Data=new float[N];
    INPUT(FIN,N);
    // SHOW(N);
    // SHOW_BIN(N);
    // cuda preprocess
    cudaMalloc((void**)&dev_srcData,sizeof(float)*N);
    cudaMalloc((void**)&dev_dstData,sizeof(float)*N);    
    cudaMemcpy(dev_srcData, Data, sizeof(float)*N, cudaMemcpyHostToDevice);

    // cudaMalloc((void**)&block_sum_0,sizeof(unsigned int)*num_block);
    // cudaMalloc((void**)&block_sum_1,sizeof(unsigned int)*num_block);
    // cudaMalloc((void**)&block_sum_2,sizeof(unsigned int)*num_block);
    // cudaMalloc((void**)&block_sum_3,sizeof(unsigned int)*num_block);
    cudaMalloc((void**)&block_sum,sizeof(unsigned int)*num_block*4);
    cudaMalloc((void**)&prefix_block_sum,sizeof(unsigned int)*num_block*4);


    // int num_lists = 128; // the number of parallel threads
    // radix sort
    cudaEventRecord( start, 0 ) ;
    // GPU_radix_sort<<<1,num_lists>>>(dev_srcData, dev_dstData, num_lists, N);
    Data_Preprocess<<< num_block , BlockSize>>>(dev_srcData,N);
    // Intra_Block_radix_sort<<< num_block, BlockSize>>>(dev_srcData,dev_dstData,N, block_sum_0, block_sum_1, block_sum_2, block_sum_3);
    Intra_Block_radix_sort<<< num_block, BlockSize>>>(dev_srcData,dev_dstData,N, block_sum, prefix_block_sum, num_block);
    Data_Postprocess<<< num_block, BlockSize>>>(dev_dstData,N);
    cudaEventRecord( stop, 0 ) ;
    cudaEventSynchronize( stop );

    // cuda postprocess
    cudaMemcpy(Data, dev_dstData, sizeof(float)*N, cudaMemcpyDeviceToHost);
    // output
    cudaFree(dev_srcData);
    cudaFree(dev_dstData);
    cudaFree(block_sum);
    cudaFree(prefix_block_sum);
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

// __global__ void Intra_Block_radix_sort(float* src_data,float* dest_data,int num_data,\
//                                         unsigned int* block_sum_0, unsigned int* block_sum_1, unsigned int* block_sum_2, unsigned int* block_sum_3)
__global__ void Intra_Block_radix_sort(float* src_data,float* dest_data,int num_data, unsigned int* block_sum, unsigned int* prefix_block_sum, int num_block)
{
    int tid=blockDim.x*blockIdx.x+threadIdx.x;

    if(tid>= num_data) return;
    // // load block data to share memory
    __shared__ unsigned int sData[BlockSize];
    // __shared__ unsigned int tempData[BlockSize];

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
        // [步驟 i, ii, iii] 在 block 內計算 False potision (local prefix)
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

        // [步驟 b] block 內做 local prefix sum
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

        // // compute position
        // int num_0 = PrefixFalseBuffer_0[num_data];
        // int num_01 = num_0 + PrefixFalseBuffer_1[num_data];
        // int num_012 = num_01 + PrefixFalseBuffer_2[num_data];

        // if(FalseBuffer_0[threadIdx.x]){
        //     // Position[tid]=PrefixFalseBuffer_0[tid];
        //     Position[threadIdx.x]=PrefixFalseBuffer_0[threadIdx.x];          
        // }
        // else if(FalseBuffer_1[threadIdx.x]){
        //     // Position[tid]= num_0 + PrefixFalseBuffer_1[tid]; // (0 有幾個) + (這是第幾個 1)
        //     Position[threadIdx.x]= num_0 + PrefixFalseBuffer_1[threadIdx.x]; // (0 有幾個) + (這是第幾個 1)
        // }
        // else if(FalseBuffer_2[threadIdx.x]){
        //     // Position[tid]= num_01 + PrefixFalseBuffer_2[tid]; // (0 有幾個) + (1 有幾個) + (這是第幾個 2)
        //     Position[threadIdx.x]= num_01 + PrefixFalseBuffer_2[threadIdx.x]; // (0 有幾個) + (1 有幾個) + (這是第幾個 2)
        // }
        // else if(FalseBuffer_3[threadIdx.x]){
        //     // Position[tid]= num_012 + PrefixFalseBuffer_3[tid]; // (0 有幾個) + (1 有幾個) + (2 有幾個) + (這是第幾個 3)
        //     Position[threadIdx.x]= num_012 + PrefixFalseBuffer_3[threadIdx.x]; // (0 有幾個) + (1 有幾個) + (2 有幾個) + (這是第幾個 3)
        // }
        // __syncthreads();

        // scatter
        // tempData[Position[tid]]=sData[tid];
        // tempData[Position[threadIdx.x]]=sData[threadIdx.x];

        bit_mask_1<<=2;
        bit_mask_2<<=2;
        bit_mask_3<<=2;

        __syncthreads();
        // save data

        // sData[threadIdx.x]=tempData[tid];
        // sData[threadIdx.x]=tempData[threadIdx.x];


        if(threadIdx.x == 0){

            ///////////////////////////////////// 測試區塊 START /////////////////////////////////////
            // if(blockIdx.x == 0)
            //     printf("\033[1;34mRound %d\033[0m\n", i);
            // // 印出該 pass 跑完後，編號為 blockIdx.x 的 block 做完的 Local shuffle 長怎樣
            // printf("Local shuffle for block %d:\n", blockIdx.x);
            // for(int j = 0; j < blockDim.x; j++){
            //     if(sData[j] == 0) break;
            //     for(int k=0;k<32;k++){
            //         // 變色看得比較清楚這輪是在處理哪兩個位數
            //         if(k <= 31-2*i && k >= 30-2*i)
            //             printf("\033[1;31m%s\033[0m", ((((unsigned int *)sData)[j]&(0x80000000>>k))?"1":"0"));
            //         else
            //             printf("%s", ((((unsigned int *)sData)[j]&(0x80000000>>k))?"1":"0"));
            //     }
            //     printf("\n");
            // }
            ///////////////////////////////////// 測試區塊 END /////////////////////////////////////

          
            
            /*************************** [步驟 d] 計算 block_sum (START) ***************************/
            // example: 第 blockIdx.x 個 block 的 0 總共有 PrefixFalseBuffer_0[num_data] 個
            // 第一種寫法: 全部 global data 分四塊 Block_sum (0~3)，每個長度是 num_block
            // block_sum_0[blockIdx.x] = PrefixFalseBuffer_0[num_data];
            // block_sum_1[blockIdx.x] = PrefixFalseBuffer_1[num_data];
            // block_sum_2[blockIdx.x] = PrefixFalseBuffer_2[num_data];
            // block_sum_3[blockIdx.x] = PrefixFalseBuffer_3[num_data];

            // 第二種寫法: 全部 global data 只有一塊 Block_sum，長度是 num_block * 4，此寫法方便後面算 prefix_block_sum
            block_sum[num_block*0 + blockIdx.x] = PrefixFalseBuffer_0[num_data];
            block_sum[num_block*1 + blockIdx.x] = PrefixFalseBuffer_1[num_data];
            block_sum[num_block*2 + blockIdx.x] = PrefixFalseBuffer_2[num_data];
            block_sum[num_block*3 + blockIdx.x] = PrefixFalseBuffer_3[num_data];
            /*************************** [步驟 d] 計算 block_sum (END) ***************************/
        

            ///////////////////////////////////// 測試區塊 START /////////////////////////////////////
            //// 第一種寫法
            // printf("===============================================\n");
            // printf("This is input block %d\n", blockIdx.x);
            // printf("block_sum_0 position %d:\n", blockIdx.x);
            // printf("%d\n", block_sum_0[blockIdx.x]);
            // printf("block_sum_1 position %d:\n", blockIdx.x);
            // printf("%d\n", block_sum_1[blockIdx.x]);
            // printf("block_sum_2 position %d:\n", blockIdx.x);
            // printf("%d\n", block_sum_2[blockIdx.x]);
            // printf("block_sum_3 position %d:\n", blockIdx.x);
            // printf("%d\n", block_sum_3[blockIdx.x]);
            // printf("===============================================\n");

            //// 第二種寫法
            // printf("===============================================\n");
            // printf("This is input block %d\n", blockIdx.x);
            // printf("block_sum_0 position %d:\n", blockIdx.x);
            // printf("%d\n", block_sum[num_block*0 + blockIdx.x]);
            // printf("block_sum_1 position %d:\n", blockIdx.x);
            // printf("%d\n", block_sum[num_block*1 + blockIdx.x]);
            // printf("block_sum_2 position %d:\n", blockIdx.x);
            // printf("%d\n", block_sum[num_block*2 + blockIdx.x]);
            // printf("block_sum_3 position %d:\n", blockIdx.x);
            // printf("%d\n", block_sum[num_block*3 + blockIdx.x]);
            // printf("===============================================\n");
            ///////////////////////////////////// 測試區塊 END /////////////////////////////////////
            
        } 
        __syncthreads();

        prefix_block_sum[blockIdx.x * num_block + threadIdx.x + 1] = block_sum[blockIdx.x * num_block + threadIdx.x];
        __syncthreads();

        // prefix_block_sum 是全局共享的陣列而非單個 block 內的，所以把下面原本寫的註解掉了
        // ScanBlock(prefix_block_sum+1+blockDim.x*blockIdx.x); // The last one is total false
        // __syncthreads();

        // [步驟 e] 全局的 prefix_block_sum 交給第一個人做，因為想來想去沒想到比較好的方法，而且 prefix_block_sum 不會太長才對
        if(tid == 0){
            prefix_block_sum[0] = 0;
            for(int j = 1; j < num_block*4; j++){
                prefix_block_sum[j] = block_sum[j-1];
            }
            for(int j = 1; j < num_block*4; j++){
                prefix_block_sum[j] += prefix_block_sum[j-1];
            }
        }
        __syncthreads();        
        

        // 印出 prefix_block_sum (有時候加這個會有 bug)
        // if(threadIdx.x == 0 && blockIdx.x == 0){
        //     printf("%d\n", num_block);
        //     for(int j = 0; j < num_data; j++)
        //         printf("%d ", prefix_block_sum[j]);
        //     printf("\n");
        // }

        // [步驟 f] 將步驟b(local prefix sum)和步驟e(prefix block sum)結果合起來，就是步驟f(global position)
        if(num == 0)
            Position[threadIdx.x] =  prefix_block_sum[num_block*0 + blockIdx.x] + PrefixFalseBuffer_0[threadIdx.x];
        else if(num == 1)
            Position[threadIdx.x] =  prefix_block_sum[num_block*1 + blockIdx.x] + PrefixFalseBuffer_1[threadIdx.x];
        else if(num == 2)
            Position[threadIdx.x] =  prefix_block_sum[num_block*2 + blockIdx.x] + PrefixFalseBuffer_2[threadIdx.x];
        else if(num == 3)
            Position[threadIdx.x] =  prefix_block_sum[num_block*3 + blockIdx.x] + PrefixFalseBuffer_3[threadIdx.x];
        __syncthreads();

        // [步驟 g] 這輪 pass 跑完重新分配位置
        dest_data[Position[threadIdx.x]]=sData[threadIdx.x];
        __syncthreads();

        sData[threadIdx.x]=dest_data[tid];
        __syncthreads();

    }
    
    dest_data[tid]=((float*)sData)[threadIdx.x];
    __syncthreads();
}