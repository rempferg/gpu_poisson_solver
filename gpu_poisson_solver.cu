#include <stdio.h>
#include <cufft.h>

#define OUTPUT
//#define OUTPUT_GF
//#define OUTPUT_CHARGE
//#define OUTPUT_CHARGE_FFT
//#define OUTPUT_CHARGE_FFT_GF
//#define OUTPUT_POTENTIAL

void displayDeviceProperties(cudaDeviceProp* pDeviceProp);

__global__ void multiplyGreensFunc(cufftComplex* data, cufftReal* greensfunc, int n) {
    for(int i = blockDim.x*blockIdx.x+threadIdx.x; i < n*(n/2+1); i += gridDim.x*blockDim.x) {
        data[i].x *= greensfunc[i];
        data[i].y *= greensfunc[i];
    }
}

int main(int argc, char** argv) {
    /* usage message */
    if(argc != 2) {
        fprintf(stderr, "USAGE: %s gridsize\n       %s info\n\nCalculates the electrostatic potential of a hardcoded charge distribution on a 2D grid of size gridsize x gridsize. The grid spacing is fixed as 1.\n", argv[0], argv[0]);
        return 1;
    }
    
    /* cuda info */
    if(strcmp(argv[1], "info") == 0) {
        cudaDeviceProp deviceProp;
        int devCount = 0;

        cudaGetDeviceCount(&devCount);
        printf("Number of devices: %d\n", devCount);
        
        for (int i = 0; i < devCount; ++i) {
            memset(&deviceProp, 0, sizeof(deviceProp));
            
            if(cudaGetDeviceProperties(&deviceProp, i) == cudaSuccess)
                displayDeviceProperties(&deviceProp);
            else
                printf("\n%s", cudaGetErrorString(cudaGetLastError()));
        }
        
        return 0;
    }
    
    int n = atoi(argv[1]);
    
    fprintf(stderr, "Calculating electrostatic potential on a %dx%d grid\n", n, n);
    
    /* timing */
    float time = 0.0, time_tmp;
    cudaEvent_t start, stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    /* allocations */
#if defined(OUTPUT) || defined(OUTPUT_GF) || defined(OUTPUT_CHARGE) || defined(OUTPUT_CHARGE_FFT) || defined(OUTPUT_CHARGE_FFT_GF) || defined(OUTPUT_POTENTIAL)
    FILE* fp;
#endif

    cufftHandle plan_fft;
    cufftHandle plan_ifft;
    cufftComplex* data_dev;
    cufftReal* greensfunc_dev;
    cufftComplex* data_host;
    cufftReal* data_real_host;
    cufftReal* greensfunc_host;
    
    cudaMalloc((void**) &data_dev, sizeof(cufftComplex)*n*(n/2+1));
    
    if(cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to allocate\n");
        return 1;
    }
    
    cudaMalloc((void**) &greensfunc_dev, sizeof(cufftReal)*n*(n/2+1));
    
    if(cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to allocate\n");
        return 1;
    }
    
    cudaMallocHost((void**) &data_host, sizeof(cufftComplex)*n*(n/2+1));
    
    if(cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to allocate\n");
        return 1;
    }
    
    data_real_host = (cufftReal*) data_host;
    
    cudaMallocHost((void**) &greensfunc_host, sizeof(cufftReal)*n*(n/2+1));
    
    if(cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to allocate\n");
        return 1;
    }
    
    /* greens function */
    fprintf(stderr, "Creating k-space greens function in host memory\n");
    
    for(int y = 0; y < n; y++)
        for(int x = 0; x < n/2+1; x++)
            if(x == 0 && y == 0)
                greensfunc_host[(n/2+1)*y+x] = 0.0; //setting 0th fourier mode to 0 enforces charge neutrality
            else
                greensfunc_host[(n/2+1)*y+x] = -0.5 / (cos(2.0*M_PI*x/(cufftReal)n) + cos(2.0*M_PI*y/(cufftReal)n) - 2.0);
    
#if defined(OUTPUT) || defined(OUTPUT_GF)
    fprintf(stderr, "Output of greens function: gf.dat\n");
    
    if((fp = fopen("gf.dat", "w")) == NULL) {
        fprintf(stderr, "ERROR: Could not open file\n");
        return 1;
    }

    for(int x = 0; x < n; x++) {            
        for(int y = 0; y < n; y++)
            if(x >= n/2+1)
                fprintf(fp, "%d %d %f\n", x, y, greensfunc_host[(n/2+1)*(n-y)+(n-x)]);
            else
                fprintf(fp, "%d %d %f\n", x, y, greensfunc_host[(n/2+1)*y+x]);
        
        fprintf(fp, "\n");
    }

    fclose(fp);
#endif
                
    fprintf(stderr, "Copying greens function to device\n");
    
    cudaMemcpy(greensfunc_dev, greensfunc_host, sizeof(cufftReal)*n*(n/2+1), cudaMemcpyHostToDevice);
    
    if(cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to copy\n");
        return 1;
    }
    
    /* charge density */
    fprintf(stderr, "Writing charge density in host memory\n");
    
    for(int i = 0; i < n*(n/2+1); i++) {
        data_host[i].x = 0.0;
        data_host[i].y = 0.0;
    }
    
    /*
    for(int x = 0; x < n; x++)
        for(int y = 0; y < n; y++)
            data_real_host[n*y+x] = 1.0;
    
    for(int x = 0; x < n; x++)
        for(int y = 0; y < n; y++)
            if(x==n/3 && y==n/4)
                data_real_host[n*y+x] = 1.0;
            else
                data_real_host[n*y+x] = 0.0;
    
    for(int x = 0; x < n; x++)
        for(int y = 0; y < n; y++) {
            data_real_host[n*x+y] = sin(2.0*PI/n*(10*x+10*y));
        }
            
    for(int x = 0; x < n; x++)
        for(int y = 0; y < n; y++)
            if((x-n/2)*(x-n/2)/10.0 + (y-n/2)*(y-n/2) <= n*n/128.0)
                data_real_host[n*y+x] = 1.0;
            else
                data_real_host[n*y+x] = 0.0;
    
    for(int x = 0; x < n; x++)
        for(int y = 0; y < n; y++)
            if(x==n/2 && y==n/2)
                data_real_host[n*y+x] = 1.0;
            else
                data_real_host[n*y+x] = 0.0;
    
    double d; //those tilted plates work well for a power of two grid
    double l = n * sqrt(5.0)/2.0;
    
    for(int x = 0; x < n; x++)
        for(int y = 0; y < n; y++) {
            d = ((double)x + (double)y/2.0) * 2.0/sqrt(5.0);
            
            if( (d >= 2*l/5.0 && d < 2*l/5.0+0.5) || (d >= 4*l/5.0 && d < 4*l/5.0+0.5) )
                data_real_host[n*y+x] = 1.0;
            else if( (d >= 1*l/5.0 && d < 1*l/5.0+0.5) || (d >= 3*l/5.0 && d < 3*l/5.0+0.3) || (d >= 5*l/5.0 && d < 5*l/5.0+0.3) )
                data_real_host[n*y+x] = -1.0;
            else
                data_real_host[n*y+x] = 0.0;
        }
    */
    
    for(int x = 0; x < n; x++)
        for(int y = 0; y < n; y++)
            if(y == n/4)
                data_real_host[n*y+x] = 1.0;
            else if(y == 3*n/4)            
                data_real_host[n*y+x] = -1.0;
            else
                data_real_host[n*y+x] = 0.0;
                
    fprintf(stderr, "Copying charge density to device\n");
    
    cudaMemcpy(data_dev, data_host, sizeof(cufftComplex)*n*(n/2+1), cudaMemcpyHostToDevice);
    
    if(cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to copy\n");
        return 1;
    }
    
#if defined(OUTPUT) || defined(OUTPUT_CHARGE)
    fprintf(stderr, "Output of charge density: charge.dat\n");
    
    if((fp = fopen("charge.dat", "w")) == NULL) {
        fprintf(stderr, "ERROR: Could not open file\n");
        return 1;
    }

    for(int x = 0; x < n; x++) {            
        for(int y = 0; y < n; y++)
            fprintf(fp, "%d %d %f\n", x, y, data_real_host[y*n+x]);
        
        fprintf(fp, "\n");
    }

    fclose(fp);
#endif

    /* create 2D FFT plans */
    fprintf(stderr, "Setting up FFT and iFFT plans\n");
        
    if(cufftPlan2d(&plan_fft, n, n, CUFFT_R2C) != CUFFT_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to create fft plan\n");
        return 1;
    }
    
    if(cufftSetCompatibilityMode(plan_fft, CUFFT_COMPATIBILITY_NATIVE) != CUFFT_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to set fft compatibility mode to native\n");
        return 1;
    }
        
    if(cufftPlan2d(&plan_ifft, n, n, CUFFT_C2R) != CUFFT_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to create ifft plan\n");
        return 1;
    }
    
    if(cufftSetCompatibilityMode(plan_ifft, CUFFT_COMPATIBILITY_NATIVE) != CUFFT_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to set ifft compatibility mode to native\n");
        return 1;
    }
    
    /* FFT in place */
    fprintf(stderr, "executing FFT in place\n");
    
    cudaEventRecord(start, 0);
    
    if(cufftExecR2C(plan_fft, (cufftReal*) data_dev, data_dev) != CUFFT_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to execute FFT plan\n");
        return 1;
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_tmp, start, stop);
    printf("execution time: %f ms\n", time_tmp);
    time += time_tmp;
    
    if(cudaThreadSynchronize() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to synchronize\n");
        return 1;
    }
    
#if defined(OUTPUT) || defined(OUTPUT_CHARGE_FFT)
    /* retrieving result from device */
    fprintf(stderr, "Retrieving result from device\n");
    
    cudaMemcpy(data_host, data_dev, sizeof(cufftComplex)*n*(n/2+1), cudaMemcpyDeviceToHost);
    
    /* output result */
    fprintf(stderr, "Output of FFT(charge_density): charge_fft.dat\n");
    
    if((fp = fopen("charge_fft.dat", "w")) == NULL) {
        fprintf(stderr, "ERROR: Could not open file\n");
        return 1;
    }
    
    for(int x = 0; x < n; x++) {
        for(int y = 0; y < n; y++)
            if(x >= n/2+1)
                fprintf(fp, "%d %d %f %f\n", x, y, data_host[(n/2+1)*(n-y)+(n-x)].x/n, -data_host[(n/2+1)*(n-y)+(n-x)].y/n);
            else
                fprintf(fp, "%d %d %f %f\n", x, y, data_host[(n/2+1)*y+x].x/n, data_host[(n/2+1)*y+x].y/n);
        
        fprintf(fp, "\n");
    }
    
    fclose(fp);
#endif
    
    /* multiplying with greens function */
    fprintf(stderr, "Executing multiplication with greens function in place\n");
    
    cudaEventRecord(start, 0);
    
    multiplyGreensFunc<<<14,32*32>>>(data_dev, greensfunc_dev, n); //18-fold occupation seems to be optimal for the GT520 and 32-fold for the C2050
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_tmp, start, stop);
    printf("execution time: %f ms\n", time_tmp);
    time += time_tmp;
    
    if(cudaThreadSynchronize() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to synchronize\n");
        return 1;
    }
    
#if defined(OUTPUT) || defined(OUTPUT_CHARGE_FFT_GF)
    /* retrieving result from device */
    fprintf(stderr, "Retrieving result from device\n");
    
    cudaMemcpy(data_host, data_dev, sizeof(cufftComplex)*n*(n/2+1), cudaMemcpyDeviceToHost);
    
    /* output result */
    fprintf(stderr, "Output of FFT(charge_density)*greensfkt: charge_fft_gf.dat\n");
    
    if((fp = fopen("charge_fft_gf.dat", "w")) == NULL) {
        fprintf(stderr, "ERROR: Could not open output file\n");
        return 1;
    }
    
    for(int x = 0; x < n; x++) {
        for(int y = 0; y < n; y++)
            if(x >= n/2+1)
                fprintf(fp, "%d %d %f %f\n", x, y, data_host[(n/2+1)*(n-y)+(n-x)].x/n, -data_host[(n/2+1)*(n-y)+(n-x)].y/n);
            else
                fprintf(fp, "%d %d %f %f\n", x, y, data_host[(n/2+1)*y+x].x/n, data_host[(n/2+1)*y+x].y/n);
        
        fprintf(fp, "\n");
    }
    
    fclose(fp);
#endif
    
    /* inverse FFT in place */
    fprintf(stderr, "executing iFFT in place\n");
    
    cudaEventRecord(start, 0);
    
    if(cufftExecC2R(plan_ifft, data_dev, (cufftReal*) data_dev) != CUFFT_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to execute iFFT plan\n");
        return 1;
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_tmp, start, stop);
    printf("execution time: %f ms\n", time_tmp);
    time += time_tmp;
    
    if(cudaThreadSynchronize() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to synchronize\n");
        return 1;
    }
    
#if defined(OUTPUT) || defined(OUTPUT_POTENTIAL)
    /* retrieving result from device */
    fprintf(stderr, "Retrieving result from device\n");
    cudaMemcpy(data_host, data_dev, sizeof(cufftComplex)*n*(n/2+1), cudaMemcpyDeviceToHost);
    
    /* output result */
    fprintf(stderr, "Output of iFFT(FFT(charge_density)*greensftk): charge_fft_gf_ifft.dat\n");
    
    if((fp = fopen("charge_fft_gf_ifft.dat", "w")) == NULL) {
        fprintf(stderr, "ERROR: Could not open output file\n");
        return 1;
    }

    for(int x = 0; x < n; x++) {
        for(int y = 0; y < n; y++)
            fprintf(fp, "%d %d %f\n", x, y, data_real_host[n*y+x]/(n*n));
        
        fprintf(fp, "\n");
    }

    fclose(fp);
#endif

    /* cleanup */
    fprintf(stderr, "Cleanup\n");
    
    cufftDestroy(plan_fft);
    cufftDestroy(plan_ifft);
    
    cudaFree(data_dev);
    cudaFree(greensfunc_dev);
    cudaFree(data_host);
    cudaFree(greensfunc_host);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("net device execution time: %f ms\n", time);
    
    return 0;
}

void displayDeviceProperties(cudaDeviceProp* pDeviceProp)
{
    if(!pDeviceProp)
        return;

    printf("\nDevice Name \t – %s ", pDeviceProp->name);
    printf("\n**************************************");
    printf("\nTotal Global Memory\t\t -%d KB", (int) pDeviceProp->totalGlobalMem/1024);
    printf("\nShared memory available per block \t – %d KB", (int) pDeviceProp->sharedMemPerBlock/1024);
    printf("\nNumber of registers per thread block \t – %d", pDeviceProp->regsPerBlock);
    printf("\nWarp size in threads \t – %d", pDeviceProp->warpSize);
    printf("\nMemory Pitch \t – %d bytes", (int) pDeviceProp->memPitch);
    printf("\nMaximum threads per block \t – %d", pDeviceProp->maxThreadsPerBlock);
    printf("\nMaximum Thread Dimension (block) \t – %d %d %d", pDeviceProp->maxThreadsDim[0], pDeviceProp->maxThreadsDim[1], pDeviceProp->maxThreadsDim[2]);
    printf("\nMaximum Thread Dimension (grid) \t – %d %d %d", pDeviceProp->maxGridSize[0], pDeviceProp->maxGridSize[1], pDeviceProp->maxGridSize[2]);
    printf("\nTotal constant memory \t – %d bytes", (int) pDeviceProp->totalConstMem);
    printf("\nCUDA ver \t – %d.%d", pDeviceProp->major, pDeviceProp->minor);
    printf("\nClock rate \t – %d KHz", pDeviceProp->clockRate);
    printf("\nTexture Alignment \t – %d bytes", (int) pDeviceProp->textureAlignment);
    printf("\nDevice Overlap \t – %s", pDeviceProp-> deviceOverlap?"Allowed":"Not Allowed");
    printf("\nNumber of Multi processors \t – %d\n", pDeviceProp->multiProcessorCount);
}
