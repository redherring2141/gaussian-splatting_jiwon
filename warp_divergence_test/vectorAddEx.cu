#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n, double max_limit, int n_iter)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    for (int iter = 0; iter < n_iter; iter++)
    {
        if (id < n)
        {
            if (c[id] < max_limit)
            {
                c[id] = a[id] + b[id];
            }
        }
    }
}
 
int main( int argc, char* argv[] )
{
    // Size of vectors
    int n_x = 4;
    int n_y = 4;
    int n = n_x*n_y;
    double portion = 0.5;
    double max_limit = 100;
    int outer_iter = int(max_limit);
    int inner_iter = int(max_limit);
 
    // Host input vectors
    double *h_a;
    double *h_b;
    //Host output vector
    double *h_c;
 
    // Device input vectors
    double *d_a;
    double *d_b;
    //Device output vector
    double *d_c;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
 
    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);
 
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
 
    for (int iter=0; iter<int(max_limit); iter++)
    {
        // Initialize vectors on host
        for(int idx = 0; idx < n; idx++ )
        {
            //h_a[i] = sin(i)*sin(i);
            //h_b[i] = cos(i)*cos(i);
            h_a[idx] = (idx%2)==0 ? 0 : (max_limit*portion);
            h_b[idx] = 1;
        }

        // Copy host vectors to device
        cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
        int blockSize, gridSize;
    
        // Number of threads in each thread block
        blockSize = 1024;
    
        // Number of thread blocks in grid
        gridSize = (int)ceil((float)n/blockSize);
    
        // Execute the kernel
        vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n, max_limit, n_iter);
    
        // Copy array back to host
        cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
    }
 
    // Sum up vector c and print result divided by n, this should equal 1 within error

    printf("<Final result>\n");
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            printf("%d ");
        }
        printf("\n");
    }
 
    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
 
    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);
 
    return 0;
}