#include <stdio.h>
#include <cuda_runtime.h>
#include "helper_functions.h"
#include "helper_cuda.h" 
#define Iter 10 //Number of iterations

#define N 3 //Size of the mask


#define tile_size 16

#define block_size (tile_size + N -1)

//Filters
float identity[9] = {0,0,0,0,1,0,0,0,0};
float edge[9] = {1,0,-1,0,0,0,-1,0,1};
float sharp[9] = {0,-1,0,-1,5,-1,0,-1,0};

float blur9[9] = {(float) 1/9,(float) 1/9,(float) 1/9,(float) 1/9,(float) 1/9,(float) 1/9,(float) 1/9,(float) 1/9,(float) 1/9};
float blur25[25] = {(float) 1/25,(float) 1/25,(float) 1/25,(float) 1/25,(float) 1/25,(float) 
                    1/25,(float) 1/25,(float) 1/25,(float) 1/25,(float) 1/25,(float) 1/25,(float) 1/25,(float) 
                    1/25,(float) 1/25,(float) 1/25,(float) 1/25,(float) 1/25,(float) 1/25,(float) 1/25,(float) 
                    1/25,(float) 1/25,(float) 1/25,(float) 1/25,(float) 1/25,(float) 1/25};


float* filter = blur9;
const char* filter_s = "Blur 3x3";

//Constant Memory Allocation
__constant__ float deviceFilter[N * N];

// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;

void printArray(float* arr, int width,int height)
{
    for (int i=0;i<width*height;i++)
    {
        int row = i/height;
        int col = i % width;
        if (col == 0)
        {
            printf("\n");
        }
    printf("%f ",arr[row*height + col]);

    }
}

void checkCUDAError(const char *msg) 
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}



//function for padding the input image with zeros for the convolution/////
 void padding(float* src, int width,int height,int pad_size,float* out,int padded_w,int padded_h){
            for (int i = 0; i< padded_h;++i){
                for(int j = 0;j<padded_w;++j){
                  if(i < pad_size || j < pad_size || i >= (height+pad_size) || j >= (width+pad_size)){
                    out[i*padded_w + j]=0.0;
                }
                else{
                    out[i*padded_w + j] = src[(i-pad_size)*width + (j-pad_size)];

                }
            }
        }
}

//Serial Convolution
void Seq_convolution(float* src,float* Filter,int filter_dim, int width, int height, float* output,
                     int padder)
{
    float Sum;
    float val,fval;
    for (int i = 0; i< height-padder;i++)
    {
        for (int j = 0; j<width-padder;j++)
        {
            Sum = 0.0;
            for (int k = (-1*filter_dim/2); k<=filter_dim/2;k++)
            {
                for (int l = (-1*filter_dim/2); l<=filter_dim/2;l++)
                {
                    val = src[(j-l) + (i-k)*width];
                    fval = Filter[(l+filter_dim/2) + (k+filter_dim/2)*filter_dim];
                    Sum += val*fval;
                }
            }
         output[j + (i*width)] = Sum;
        }
    }
}
// Convolution Global
__global__ void naive_convolution(float* src, float* out, float* Filter, int width, int height,
                                  int padder, int filter_dim)
{
    unsigned int col = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int row = threadIdx.y + blockIdx.y*blockDim.y;
 
    int row_start = row - padder ;
    int col_start = col - padder;
 
    float Sum = 0;
 
    for (int i = 0; i < filter_dim;i++)
    {
        for (int j = 0; j < filter_dim;j++)
        {
            if((row_start + i) >= 0 && (row_start + i) < width)
            {
                if ((col_start + j) >=0 && (col_start + j) < height)
                {
                    Sum += src[(row_start + i)*width + (col_start + j)]*Filter[i*filter_dim + j];
                }
            }
        }
    }
    out[row*width + col] = Sum;
 
 }


// Convolution Constant Global
__global__ void gConst_convolution(float* src, float* out, const float *__restrict__ kernel,
                                   int width, int height,int padder, int filter_dim)
{
    unsigned int col = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int row = threadIdx.y + blockIdx.y*blockDim.y;
 
    int row_start = row - padder ;
    int col_start = col - padder;
 
    float Sum = 0;
 
    for (int i = 0; i < filter_dim;i++)
    {
        for (int j = 0; j < filter_dim;j++)
        {
            if((row_start + i) >= 0 && (row_start + i) < height)
            {
                if ((col_start + j) >=0 && (col_start + j) < width)
                {
                    Sum += src[(row_start + i)*width + (col_start + j)]*deviceFilter[i*filter_dim + j];
                }
            }
        }
    }
    out[row*width + col] = Sum;
 
 }

__global__ void share_convolution(float* src, float* out, float* Filter,
                                   int width, int height,int padder, int filter_dim)
{
    __shared__ float s_array[block_size][block_size];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    unsigned int col = tx + blockIdx.x*blockDim.x;
    unsigned int row = ty + blockIdx.y*blockDim.y;
    
    const int t_loc = tx + blockDim.x*blockIdx.x + threadIdx.y*width + blockDim.y*blockIdx.y*width;

    int row_start = row - padder ;
    int col_start = col - padder;

    if(row_start < 0 || col_start < 0){
        s_array[ty][tx] = 0.0;
    }else{
        s_array[ty][tx] = src[t_loc - padder - width*padder];
    }

    row_start = row - padder ;
    col_start = col + padder;
    if(row_start < 0 || col_start > width - 1){
         s_array[ty][tx + 2*padder] = 0.0;
    }else{
        s_array[ty][tx + 2*padder] = src[t_loc + padder - (width)*padder];
    }

    row_start = row + padder ;
    col_start = col - padder;
    if(row_start > height - 1 || col_start < 0){
        s_array[ty+ 2*padder][tx] =  0.0;
    }else{
        s_array[ty+ 2*padder][tx] = src[t_loc - padder + (width)*padder];
    }

    row_start = row + padder;
    col_start = col + padder;
    if(row_start > height - 1 || col_start > width - 1){
        s_array[ty+ 2*padder][tx+ 2*padder] = 0.0;
    }else{
        s_array[ty+ 2*padder][tx+ 2*padder] = src[t_loc + padder + (width)*padder];
    }
 
    __syncthreads();
 
    row_start = ty + padder;
    col_start = tx + padder;
 
    float Sum = 0.0;
    for (int y = -1*filter_dim/2; y <= filter_dim/2; y++)
    {
        for (int x = -1*filter_dim/2; x <= filter_dim/2;x++)
        {
            Sum += Filter[(y+padder)*filter_dim + (x + padder)]*s_array[y+row_start][x+col_start];
        }
    }
    out[t_loc] = Sum;
    
}




__global__ void sConst_convolution(float* src, float* out, float *__restrict__ kernel,int width, int height,int padder, int filter_dim)
{


__shared__ float s_array[block_size][block_size];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    unsigned int col = tx + blockIdx.x*blockDim.x;
    unsigned int row = ty + blockIdx.y*blockDim.y;
    
    const int t_loc = tx + blockDim.x*blockIdx.x + threadIdx.y*width + blockDim.y*blockIdx.y*width;

    int row_start = row - padder ;
    int col_start = col - padder;

    if(row_start < 0 || col_start < 0){
        s_array[ty][tx] = 0.0;
    }else{
        s_array[ty][tx] = src[t_loc - padder - width*padder];
    }

    row_start = row - padder ;
    col_start = col + padder;
    if(row_start < 0 || col_start > width - 1){
         s_array[ty][tx + 2*padder] = 0.0;
    }else{
        s_array[ty][tx + 2*padder] = src[t_loc + padder - (width)*padder];
    }

    row_start = row + padder ;
    col_start = col - padder;
    if(row_start > height - 1 || col_start < 0){
        s_array[ty+ 2*padder][tx] =  0.0;
    }else{
        s_array[ty+ 2*padder][tx] = src[t_loc - padder + (width)*padder];
    }

    row_start = row + padder;
    col_start = col + padder;
    if(row_start > height - 1 || col_start > width - 1){
        s_array[ty+ 2*padder][tx+ 2*padder] = 0.0;
    }else{
        s_array[ty+ 2*padder][tx+ 2*padder] = src[t_loc + padder + (width)*padder];
    }
 
    __syncthreads();
 
    row_start = ty + padder;
    col_start = tx + padder;
 
    float Sum = 0.0;
    for (int y = -1*filter_dim/2; y <= filter_dim/2; y++)
    {
        for (int x = -1*filter_dim/2; x <= filter_dim/2;x++)
        {
            Sum += deviceFilter[(y+padder)*filter_dim + (x + padder)]*s_array[y+row_start][x+col_start];
        }
    }
    out[t_loc] = Sum;
}

// Texture Memory Convolution
__global__ void tex_convolution( float* out, float* Filter, int width, int height,int filter_dim)
{
    unsigned int col = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int row = threadIdx.y + blockIdx.y*blockDim.y;
 
    float Sum = 0;
 
    for (int i = -1*filter_dim/2; i < (filter_dim/2) + 1 ;i++)
    {
        for (int j = -1*filter_dim/2; j < (filter_dim/2) + 1;j++)
        {
            Sum += Filter[(j + filter_dim/2)*(filter_dim/2) + (i + filter_dim/2)]*tex2D(tex, col + i, row + j );
        }
    }
    out[row*width + col] = Sum;
 
 }

int main()
{
    float* image = NULL;
	unsigned int width, height;
    float* Filter = (float *) malloc(N*N*sizeof(float));
    
    Filter = blur9;

	char *imageFilename = sdkFindFilePath("lena_bw.pgm", 0);

    if (imageFilename == NULL)
    {
        printf("Unable to source image file: %s\n", "lena_bw.pgm");
        exit(EXIT_FAILURE);
    }
    sdkLoadPGM(imageFilename, &image, &width, &height);
	

    //Padding variables
    unsigned int padder = N/2;
    unsigned int padded_w = width+2*(padder);
    unsigned int padded_h = height+2*(padder);
    unsigned int pad_size = padded_h * padded_w * sizeof(float);
    
    //Host arrays
    float* padded_image = (float *) malloc(pad_size);
    float* output = (float *) malloc(pad_size);
    
    
    //Device arrays
    float* d_image;
    float* d_filter;
    float* d_output;
	

    //Times
    float time = 0;
    float seq_time = 0;
    float naive_time = 0;
    float gConst_time = 0;
    float share_time = 0;
    float sConst_time = 0;
    float tex_time = 0;

    
	printf("============ %s %dx%d=============\n\n", imageFilename, padded_w, padded_h);
    printf("============ %s =======================\n\n", filter_s);
    printf("          Number of threads = %d \n\n", tile_size);
	
    padding(image,width,height,padder,padded_image,padded_w,padded_h);
    
    //Initialise Timing(Sequential)
    cudaEvent_t launch_begin, launch_end;
    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);

    for (int i = 0; i < Iter; i++)
    {
        memset(output,0,pad_size);
        
        cudaEventRecord(launch_begin,0);
    
        Seq_convolution(padded_image, filter,N,padded_w,padded_h,output,padder);
    
        cudaEventRecord(launch_end,0);
        cudaEventSynchronize(launch_end);

        cudaEventElapsedTime(&time, launch_begin, launch_end);

        seq_time += time;

    }
	char outputFilename[1024];
    strcpy(outputFilename, imageFilename);
    strcpy(outputFilename + strlen(imageFilename) - 4, "_seq.pgm");
    sdkSavePGM(outputFilename, output, width, height);
 
 
    printf("CPU:Run Time:%f ms\n", seq_time/Iter);
///////////////////////////////////////CUDA////////////////////////////////////////////////////////////


///////////////////////////////////////////Naive///////////////////////////////////////////////////////////////////
    memset(output,0,pad_size);

    //sdkLoadPGM("/content/drive/My Drive/Colab Notebooks/HPC-Assignment-3/data/lena_bw.pgm",&image, &width, &height);


    //Allocate Device Memory
    cudaMalloc((void**)&d_image,pad_size);
    cudaMalloc((void**)&d_output,pad_size);
    cudaMalloc((void**)&d_filter,N*N*sizeof(float));

    cudaMemcpy(d_image,padded_image, pad_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output,output, pad_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter,filter, N*N*sizeof(float), cudaMemcpyHostToDevice);
    

    int threads = tile_size;
    dim3 dimBlock(threads, threads);
    dim3 dimGrid(ceil((float)padded_w/threads), ceil((float)padded_h/threads));
    
    for (int i = 0; i < Iter; i++)
    {

        cudaEventRecord(launch_begin,0);

        naive_convolution<<<dimGrid,dimBlock>>>(d_image,d_output,d_filter,padded_w,padded_h,padder,N);
        
        cudaEventRecord(launch_end,0);
        cudaEventSynchronize(launch_end);

        cudaEventElapsedTime(&time, launch_begin, launch_end);

        naive_time += time;
    }
    cudaMemcpy(output, d_output, pad_size, cudaMemcpyDeviceToHost);

    checkCUDAError("Naive");

    strcpy(outputFilename, imageFilename);
    strcpy(outputFilename + strlen(imageFilename) - 4, "_naive.pgm");
    sdkSavePGM(outputFilename, output, width, height);
    
    printf("GPU(Global):Run Time:%f ms\n", naive_time/Iter);

///////////////////////////////////////////Global Constant///////////////////////////////////////////////////////////////////
    memset(output,0,pad_size);
    cudaMemset(d_output, 0, pad_size);


    cudaMemcpy(d_image,padded_image, pad_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output,output, pad_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceFilter, Filter, N*N*sizeof(float));
    
        for (int i = 0; i < Iter; i++)
    {

		cudaEventRecord(launch_begin,0);

		gConst_convolution<<<dimGrid,dimBlock>>>(d_image,d_output,d_filter,padded_w,padded_h,padder,N);
		
		cudaEventRecord(launch_end,0);
		cudaEventSynchronize(launch_end);

		cudaEventElapsedTime(&time, launch_begin, launch_end);

		gConst_time += time;
    }
    cudaMemcpy(output, d_output, pad_size, cudaMemcpyDeviceToHost);
    
    checkCUDAError("gConst");

    strcpy(outputFilename, imageFilename);
    strcpy(outputFilename + strlen(imageFilename) - 4, "_gConst.pgm");
    sdkSavePGM(outputFilename, output, width, height);
	
    printf("GPU(Global Constant):Run Time:%f ms\n", gConst_time/Iter);

///////////////////////////////Shared/////////////////////////////////////////////////////
    memset(output,0,pad_size);
    cudaMemset(d_output, 0, pad_size);


    cudaMemcpy(d_image,padded_image, pad_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output,output, pad_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter,Filter, N*N*sizeof(float), cudaMemcpyHostToDevice);


    dim3 dimBlock1(tile_size, tile_size);
    dim3 dimGrid1(ceil((float)width/tile_size), ceil((float)height/tile_size));

    for (int i = 0; i < Iter; i++)
    {
        cudaEventRecord(launch_begin,0);

        share_convolution<<<dimGrid,dimBlock>>>(d_image,d_output,d_filter,padded_w,padded_h,padder,N);
        
        cudaEventRecord(launch_end,0);
        cudaEventSynchronize(launch_end);

        cudaEventElapsedTime(&time, launch_begin, launch_end);
        share_time += time;
    }
	
    cudaMemcpy(output, d_output, pad_size, cudaMemcpyDeviceToHost);
    checkCUDAError("share");

    strcpy(outputFilename, imageFilename);
    strcpy(outputFilename + strlen(imageFilename) - 4, "_share.pgm");
    sdkSavePGM(outputFilename, output, width, height);

    printf("GPU(Share) Tile Size = %d:Run Time:%f ms\n",block_size, share_time/Iter);



///////////////////////////////Shared Constant/////////////////////////////////////////////////////
    memset(output,0,pad_size);
    cudaMemset(d_output, 0, pad_size);


    cudaMemcpy(d_image,padded_image, pad_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output,output, pad_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceFilter, Filter, N*N*sizeof(float));

    for (int i = 0; i < Iter; i++)
    {
        cudaEventRecord(launch_begin,0);

        sConst_convolution<<<dimGrid,dimBlock>>>(d_image,d_output,d_filter,padded_w,padded_h,padder,N);
        
        cudaEventRecord(launch_end,0);
        cudaEventSynchronize(launch_end);

        cudaEventElapsedTime(&time, launch_begin, launch_end);
        sConst_time += time;
    }
	
    cudaMemcpy(output, d_output, pad_size, cudaMemcpyDeviceToHost);
    checkCUDAError("sConst");

    strcpy(outputFilename, imageFilename);
    strcpy(outputFilename + strlen(imageFilename) - 4, "_sConst.pgm");
    sdkSavePGM(outputFilename, output, width, height);


    printf("GPU(Shared Constant) Tile Size = %d:Run Time:%f ms\n", block_size, sConst_time/Iter);


///////////////////////////////Texture Constant/////////////////////////////////////////////////////
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0, cudaChannelFormatKindFloat);
    cudaArray* cuda_image;
    checkCudaErrors(cudaMallocArray(&cuda_image, &channelDesc,padded_w,padded_h));
    checkCudaErrors(cudaMemcpyToArray(cuda_image,0,0,padded_image, pad_size,cudaMemcpyHostToDevice));
 

    tex.normalized = false;
    tex.addressMode[0] = cudaAddressModeBorder;
    tex.addressMode[1] = cudaAddressModeBorder;
    tex.filterMode = cudaFilterModePoint;
 
    checkCudaErrors(cudaBindTextureToArray(tex,cuda_image, channelDesc));
    
    
    memset(output,0,pad_size);
    cudaMemset(d_output, 0, pad_size);


    cudaMemcpy(d_image,padded_image, pad_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output,output, pad_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceFilter, Filter, N*N*sizeof(float));

    for (int i = 0; i < Iter; i++)
    {
        cudaEventRecord(launch_begin,0);

        tex_convolution<<<dimGrid,dimBlock>>>(d_output,d_filter,padded_w,padded_h,N);
    
        cudaEventRecord(launch_end,0);
        cudaEventSynchronize(launch_end);

        cudaEventElapsedTime(&time, launch_begin, launch_end);
        tex_time += time;
    }

    cudaMemcpy(output, d_output, pad_size, cudaMemcpyDeviceToHost);
    checkCUDAError("tex");

    strcpy(outputFilename, imageFilename);
    strcpy(outputFilename + strlen(imageFilename) - 4, "_tex.pgm");
    sdkSavePGM(outputFilename, output, width, height);
	

    printf("GPU(Texture):Run Time:%f ms\n", tex_time/Iter);



    free(output);
    free(padded_image);
    cudaFree(d_image);
    cudaFree(d_filter);
    cudaFree(d_output);
    cudaFreeArray(cuda_image);
    cudaEventDestroy(launch_begin);
    cudaEventDestroy(launch_end);
    
    return 0;
}