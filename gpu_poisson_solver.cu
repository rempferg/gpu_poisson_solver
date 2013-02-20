#include <stdio.h>
#include <math.h>
#include <cufft.h>

#define PI_FLOAT 3.14159265358979323846264338327f

#define OUTPUT
//#define OUTPUT_GF
//#define OUTPUT_CHARGE
//#define OUTPUT_CHARGE_FFT
//#define OUTPUT_CHARGE_FFT_GF
//#define OUTPUT_POTENTIAL

void displayDeviceProperties(cudaDeviceProp* pDeviceProp);

__global__ void createGreensFunc(cufftReal* greensfunc, unsigned int Nx, unsigned int Ny, unsigned int Nz, float h) {
    unsigned int tmp;
    unsigned int coord[3];
    
    for(int i = blockDim.x*blockIdx.x+threadIdx.x; i < Nz * Ny * (Nx/2+1); i += gridDim.x*blockDim.x) {
        coord[0] = i % (Nx/2+1);
        tmp = i / (Nx/2+1);
        coord[1] = tmp % Ny;
        coord[2] = tmp / Ny;
        
        /* Setting 0th fourier mode to 0.0 enforces charge neutrality (effectively
           adds homogeneous counter charge). This is necessary, since the equation
           otherwise has no solution in periodic boundaries (an infinite amount of
           charge would create an infinite potential). */
        if(i == 0)
            greensfunc[i] = 0.0f;
        else
            greensfunc[i] = -0.5f * h * h / (cos(2.0f*PI_FLOAT*coord[0]/(cufftReal)Nx) + cos(2.0f*PI_FLOAT*coord[1]/(cufftReal)Ny) + cos(2.0f*PI_FLOAT*coord[2]/(cufftReal)Nz) - 3.0f);
    }
}

__global__ void multiplyGreensFunc(cufftComplex* data, cufftReal* greensfunc, unsigned int N) {
    for(int i = blockDim.x*blockIdx.x+threadIdx.x; i < N; i += gridDim.x*blockDim.x) {
        data[i].x *= greensfunc[i];
        data[i].y *= greensfunc[i];
    }
}

int main(int argc, char** argv) {
    /* usage message */
    if(!(argc == 2 && strcmp(argv[1], "info") == 0) && argc != 5) {
        printf("USAGE: %s Nx Ny Nz h\n       %s info\n\nCalculates the electrostatic potential of a hardcoded charge distribution on a 3D grid of size Nx*Ny*Nz with grid spacing h.\n", argv[0], argv[0]);
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
    
    unsigned int Nx = atoi(argv[1]);
    unsigned int Ny = atoi(argv[2]);
    unsigned int Nz = atoi(argv[3]);
    float h = atof(argv[4]);
    
    printf("Calculating electrostatic potential on a %d*%d*%d grid with spacing %f\n", Nx, Ny, Nz, h);
    
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
    cufftComplex* data_host;
    cufftReal* data_real_host;
    cufftReal* greensfunc_dev;
    cufftReal* greensfunc_host;
    
    cudaMalloc((void**) &data_dev, sizeof(cufftComplex)*Nz*Ny*(Nx/2+1));
    
    if(cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to allocate\n");
        return 1;
    }
    
    cudaMalloc((void**) &greensfunc_dev, sizeof(cufftReal)*Nz*Ny*(Nx/2+1));
    
    if(cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to allocate\n");
        return 1;
    }
    
    cudaMallocHost((void**) &data_host, sizeof(cufftComplex)*Nz*Ny*(Nx/2+1));
    
    if(cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to allocate\n");
        return 1;
    }
    
    data_real_host = (cufftReal*) data_host;
    
    cudaMallocHost((void**) &greensfunc_host, sizeof(cufftReal)*Nz*Ny*(Nx/2+1));
    
    if(cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to allocate\n");
        return 1;
    }
    
    /* greens function */
    printf("Creating greens function in device memory\n");
    
    createGreensFunc<<<14,32*32>>>(greensfunc_dev, Nx, Ny, Nz, h);
    
#if defined(OUTPUT) || defined(OUTPUT_GF)
    printf("Output of greens function: gf.vtk\n");
    
    cudaMemcpy(greensfunc_host, greensfunc_dev, sizeof(cufftReal)*Nz*Ny*(Nx/2+1), cudaMemcpyDeviceToHost);
    
    if((fp = fopen("gf.vtk", "w")) == NULL) {
        fprintf(stderr, "ERROR: Could not open file\n");
        return 1;
    }
    
    fprintf(fp, "# vtk DataFile Version 2.0\ngreens_function\nASCII\n\nDATASET STRUCTURED_POINTS\nDIMENSIONS %u %u %u\nORIGIN 0 0 0\nSPACING %f %f %f\n\nPOINT_DATA %u\nSCALARS greens_function float 1\nLOOKUP_TABLE default\n", Nx, Ny, Nz, h, h, h, Nx*Ny*Nz);

    for(int z = 0; z < Nz; z++) {
        for(int y = 0; y < Ny; y++)
            for(int x = 0; x < Nx; x++)
                if(x >= Nx/2+1)
                    fprintf(fp, " %f", greensfunc_host[Ny*(Nx/2+1)*(Nz-z-1)+(Nx/2+1)*(Ny-y-1)+(Nx-x-1)]);
                else
                    fprintf(fp, " %f", greensfunc_host[Ny*(Nx/2+1)*z+(Nx/2+1)*y+x]);
            
        fprintf(fp, "\n");
    }

    fclose(fp);
#endif
    
    /* charge density */
    printf("Writing charge density in host memory\n");
    
    for(int z = 0; z < Nz; z++)
        for(int y = 0; y < Ny; y++)
            for(int x = 0; x < Nx; x++)
                if((x-Nx/2)*(x-Nx/2) + (y-Ny/2)*(y-Ny/2) + (z-Nz/2)*(z-Nz/2) <= 5*5/(h*h)) //homogeneously chargeed sphere of radius 5
                    data_real_host[Ny*Nx*z+Nx*y+x] = h*h*h;
                else
                    data_real_host[Ny*Nx*z+Nx*y+x] = 0.0;
                
    printf("Copying charge density to device\n");
    
    cudaMemcpy(data_dev, data_host, sizeof(cufftComplex)*Nz*Ny*(Nx/2+1), cudaMemcpyHostToDevice);
    
    if(cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to copy\n");
        return 1;
    }
    
#if defined(OUTPUT) || defined(OUTPUT_CHARGE)
    printf("Output of charge density: charge.vtk\n");
    
    if((fp = fopen("charge.vtk", "w")) == NULL) {
        fprintf(stderr, "ERROR: Could not open file\n");
        return 1;
    }
    
    fprintf(fp, "# vtk DataFile Version 2.0\ncharge_density\nASCII\n\nDATASET STRUCTURED_POINTS\nDIMENSIONS %u %u %u\nORIGIN 0 0 0\nSPACING %f %f %f\n\nPOINT_DATA %u\nSCALARS charge_density float 1\nLOOKUP_TABLE default\n", Nx, Ny, Nz, h, h, h, Nx*Ny*Nz);

    for(int z = 0; z < Nz; z++) {
        for(int y = 0; y < Ny; y++)
            for(int x = 0; x < Nx; x++)
                fprintf(fp, " %f", data_real_host[Ny*Nx*z+Nx*y+x]);
        
        fprintf(fp, "\n");
    }

    fclose(fp);
#endif

    /* create 3D FFT plans */
    printf("Setting up FFT and iFFT plans\n");
    
    /* Notice how the directions x and z are exchanged. This is because for R2C
       transforms, cuda only stores half the results in the 3rd direction. At
       the same time cuda expects the fastest running index to be the one with
       only half the values stored, which effectively forces one to make the 3rd
       index (usually z) the fastest running one. I find this rather uncommon
       and want x to be the festest running index and z the slowest running, so
       I chose to exchange the two in the fourier transforms. */
    if(cufftPlan3d(&plan_fft, Nz, Ny, Nx, CUFFT_R2C) != CUFFT_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to create fft plan\n");
        return 1;
    }
    
    if(cufftSetCompatibilityMode(plan_fft, CUFFT_COMPATIBILITY_NATIVE) != CUFFT_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to set fft compatibility mode to native\n");
        return 1;
    }
        
    if(cufftPlan3d(&plan_ifft, Nz, Ny, Nx, CUFFT_C2R) != CUFFT_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to create ifft plan\n");
        return 1;
    }
    
    if(cufftSetCompatibilityMode(plan_ifft, CUFFT_COMPATIBILITY_NATIVE) != CUFFT_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to set ifft compatibility mode to native\n");
        return 1;
    }
    
    /* FFT in place */
    printf("Executing FFT in place\n");
    
    cudaEventRecord(start, 0);
    
    if(cufftExecR2C(plan_fft, (cufftReal*) data_dev, data_dev) != CUFFT_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to execute FFT plan\n");
        return 1;
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_tmp, start, stop);
    printf("Execution time: %f ms\n", time_tmp);
    time += time_tmp;
    
    if(cudaThreadSynchronize() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to synchronize\n");
        return 1;
    }
    
#if defined(OUTPUT) || defined(OUTPUT_CHARGE_FFT)
    /* retrieving result from device */
    printf("Retrieving result from device\n");
    
    cudaMemcpy(data_host, data_dev, sizeof(cufftComplex)*Nz*Ny*(Nx/2+1), cudaMemcpyDeviceToHost);
    
    /* output result */
    printf("Output of FFT(charge_density): charge_fft.vtk\n");
    
    if((fp = fopen("charge_fft.vtk", "w")) == NULL) {
        fprintf(stderr, "ERROR: Could not open file\n");
        return 1;
    }
    
    fprintf(fp, "# vtk DataFile Version 2.0\ncharge_fft\nASCII\n\nDATASET STRUCTURED_POINTS\nDIMENSIONS %u %u %u\nORIGIN 0 0 0\nSPACING %f %f %f\n\nPOINT_DATA %u\nSCALARS charge_fft float 2\nLOOKUP_TABLE default\n", Nx, Ny, Nz, h, h, h, Nx*Ny*Nz);
    
    for(int z = 0; z < Nz; z++) {
        for(int y = 0; y < Ny; y++)
            for(int x = 0; x < Nx; x++)
                if(x >= Nx/2+1)
                    fprintf(fp, " %f %f", data_host[Ny*(Nx/2+1)*(Nz-z-1)+(Nx/2+1)*(Ny-y-1)+(Nx-x-1)].x/sqrt(Nx*Ny*Nz), -data_host[Ny*(Nx/2+1)*(Nz-z-1)+(Nx/2+1)*(Ny-y-1)+(Nx-x-1)].y/sqrt(Nx*Ny*Nz));
                else
                    fprintf(fp, " %f %f", data_host[Ny*(Nx/2+1)*z+(Nx/2+1)*y+x].x/sqrt(Nx*Ny*Nz), data_host[Ny*(Nx/2+1)*z+(Nx/2+1)*y+x].y/sqrt(Nx*Ny*Nz));
        
        fprintf(fp, "\n");
    }
    
    fclose(fp);
#endif
    
    /* multiplying with greens function */
    printf("Executing multiplication with greens function in place\n");
    
    cudaEventRecord(start, 0);
    
    multiplyGreensFunc<<<14,32*32>>>(data_dev, greensfunc_dev, Nz*Ny*(Nx/2+1)); //18-fold occupation seems to be optimal for the GT520 and 32-fold for the C2050
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_tmp, start, stop);
    printf("Execution time: %f ms\n", time_tmp);
    time += time_tmp;
    
    if(cudaThreadSynchronize() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to synchronize\n");
        return 1;
    }
    
#if defined(OUTPUT) || defined(OUTPUT_CHARGE_FFT_GF)
    /* retrieving result from device */
    printf("Retrieving result from device\n");
    
    cudaMemcpy(data_host, data_dev, sizeof(cufftComplex)*Nz*Ny*(Nx/2+1), cudaMemcpyDeviceToHost);
    
    /* output result */
    printf("Output of FFT(charge_density)*greensfunc: charge_fft_gf.vtk\n");
    
    if((fp = fopen("charge_fft_gf.vtk", "w")) == NULL) {
        fprintf(stderr, "ERROR: Could not open output file\n");
        return 1;
    }
    
    fprintf(fp, "# vtk DataFile Version 2.0\ncharge_fft_gf\nASCII\n\nDATASET STRUCTURED_POINTS\nDIMENSIONS %u %u %u\nORIGIN 0 0 0\nSPACING %f %f %f\n\nPOINT_DATA %u\nSCALARS charge_fft_gf float 2\nLOOKUP_TABLE default\n", Nx, Ny, Nz, h, h, h, Nx*Ny*Nz);
    
    for(int z = 0; z < Nz; z++) {
        for(int y = 0; y < Ny; y++)
            for(int x = 0; x < Nx; x++)
                if(x >= Nx/2+1)
                    fprintf(fp, " %f %f", data_host[Ny*(Nx/2+1)*(Nz-z-1)+(Nx/2+1)*(Ny-y-1)+(Nx-x-1)].x/sqrt(Nx*Ny*Nz), -data_host[Ny*(Nx/2+1)*(Nz-z-1)+(Nx/2+1)*(Ny-y-1)+(Nx-x-1)].y/sqrt(Nx*Ny*Nz));
                else
                    fprintf(fp, " %f %f", data_host[Ny*(Nx/2+1)*z+(Nx/2+1)*y+x].x/sqrt(Nx*Ny*Nz), data_host[Ny*(Nx/2+1)*z+(Nx/2+1)*y+x].y/sqrt(Nx*Ny*Nz));
        
        fprintf(fp, "\n");
    }
    
    fclose(fp);
#endif
    
    /* inverse FFT in place */
    printf("Executing iFFT in place\n");
    
    cudaEventRecord(start, 0);
    
    if(cufftExecC2R(plan_ifft, data_dev, (cufftReal*) data_dev) != CUFFT_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to execute iFFT plan\n");
        return 1;
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_tmp, start, stop);
    printf("Execution time: %f ms\n", time_tmp);
    time += time_tmp;
    
    if(cudaThreadSynchronize() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to synchronize\n");
        return 1;
    }
    
#if defined(OUTPUT) || defined(OUTPUT_POTENTIAL)
    /* retrieving result from device */
    printf("Retrieving result from device\n");
    cudaMemcpy(data_host, data_dev, sizeof(cufftComplex)*Nz*Ny*(Nx/2+1), cudaMemcpyDeviceToHost);
    
    /* output result */
    printf("Output of iFFT(FFT(charge_density)*greensfunc): charge_fft_gf_ifft.vtk\n");
    
    if((fp = fopen("charge_fft_gf_ifft.vtk", "w")) == NULL) {
        fprintf(stderr, "ERROR: Could not open output file\n");
        return 1;
    }
    
    fprintf(fp, "# vtk DataFile Version 2.0\npotential\nASCII\n\nDATASET STRUCTURED_POINTS\nDIMENSIONS %u %u %u\nORIGIN 0 0 0\nSPACING %f %f %f\n\nPOINT_DATA %u\nSCALARS potential float 1\nLOOKUP_TABLE default\n", Nx, Ny, Nz, h, h, h, Nx*Ny*Nz);

    for(int z = 0; z < Nz; z++) {
        for(int y = 0; y < Ny; y++)
            for(int x = 0; x < Nx; x++)
                fprintf(fp, " %f", data_real_host[Ny*Nx*z+Nx*y+x]/(Nx*Ny*Nz));
        
        fprintf(fp, "\n");
    }

    fclose(fp);
#endif

    /* cleanup */
    printf("Cleanup\n");
    
    cufftDestroy(plan_fft);
    cufftDestroy(plan_ifft);
    
    cudaFree(data_dev);
    cudaFree(greensfunc_dev);
    cudaFree(data_host);
    cudaFree(greensfunc_host);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("Net device execution time: %f ms\n", time);
    
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
