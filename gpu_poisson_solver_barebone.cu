#include <cufft.h>

#define PI_FLOAT 3.14159265358979323846264338327f

__global__ void createGreensFunc(cufftReal* greensfunc, unsigned int Nx, unsigned int Ny, unsigned int Nz, float h) {
    unsigned int tmp, coord[3];
    
    for(int i = blockDim.x*blockIdx.x+threadIdx.x; i < Nz * Ny * (Nx/2+1); i += gridDim.x*blockDim.x) {
        coord[0] = i % (Nx/2+1);
        tmp = i / (Nx/2+1);
        coord[1] = tmp % Ny;
        coord[2] = tmp / Ny;
        
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
    unsigned int Nx = atoi(argv[1]);
    unsigned int Ny = atoi(argv[2]);
    unsigned int Nz = atoi(argv[3]);
    float h = atof(argv[4]);

    cufftHandle plan_fft;
    cufftHandle plan_ifft;
    cufftComplex* data_dev;
    cufftComplex* data_host;
    cufftReal* data_real_host;
    cufftReal* greensfunc_dev;
    
    cudaMalloc((void**) &data_dev, sizeof(cufftComplex)*Nz*Ny*(Nx/2+1));
    cudaMallocHost((void**) &data_host, sizeof(cufftComplex)*Nz*Ny*(Nx/2+1));
    cudaMalloc((void**) &greensfunc_dev, sizeof(cufftReal)*Nz*Ny*(Nx/2+1));
    data_real_host = (cufftReal*) data_host;
    
    createGreensFunc<<<14,32*32>>>(greensfunc_dev, Nx, Ny, Nz, h);
    
    /* charge density */
    for(int z = 0; z < Nz; z++)
        for(int y = 0; y < Ny; y++)
            for(int x = 0; x < Nx; x++)
                if((x-Nx/2)*(x-Nx/2) + (y-Ny/2)*(y-Ny/2) + (z-Nz/2)*(z-Nz/2) <= 5*5/(h*h))
                    data_real_host[Ny*Nx*z+Nx*y+x] = 1.0f;
                else
                    data_real_host[Ny*Nx*z+Nx*y+x] = 0.0f;
    
    cudaMemcpy(data_dev, data_host, sizeof(cufftComplex)*Nz*Ny*(Nx/2+1), cudaMemcpyHostToDevice);

    /* create 3D FFT plans */
    cufftPlan3d(&plan_fft, Nz, Ny, Nx, CUFFT_R2C);
    cufftSetCompatibilityMode(plan_fft, CUFFT_COMPATIBILITY_NATIVE);
    cufftPlan3d(&plan_ifft, Nz, Ny, Nx, CUFFT_C2R);
    cufftSetCompatibilityMode(plan_ifft, CUFFT_COMPATIBILITY_NATIVE);
    
    /* FFT in place */
    cufftExecR2C(plan_fft, (cufftReal*) data_dev, data_dev);
    
    /* multiplying with greens function */
    multiplyGreensFunc<<<14,32*32>>>(data_dev, greensfunc_dev, Nz*Ny*(Nx/2+1)); //18-fold occupation seems to be optimal for the GT520 and 32-fold for the C2050
    
    /* inverse FFT in place */
    cufftExecC2R(plan_ifft, data_dev, (cufftReal*) data_dev);
    
    /* retrieving result from device */
    cudaMemcpy(data_host, data_dev, sizeof(cufftComplex)*Nz*Ny*(Nx/2+1), cudaMemcpyDeviceToHost);
    
    /* output result */
    FILE* fp = fopen("potential.vtk", "w");
    
    fprintf(fp, "# vtk DataFile Version 2.0\npotential\nASCII\n\nDATASET STRUCTURED_POINTS\nDIMENSIONS %u %u %u\nORIGIN 0 0 0\nSPACING %f %f %f\n\nPOINT_DATA %u\nSCALARS potential float 1\nLOOKUP_TABLE default\n", Nx, Ny, Nz, h, h, h, Nx*Ny*Nz);

    for(int z = 0; z < Nz; z++) {
        for(int y = 0; y < Ny; y++)
            for(int x = 0; x < Nx; x++)
                fprintf(fp, " %f", data_real_host[Ny*Nx*z+Nx*y+x]/(Nx*Ny*Nz));
        
        fprintf(fp, "\n");
    }

    fclose(fp);

    /* cleanup */
    cufftDestroy(plan_fft);
    cufftDestroy(plan_ifft);
    cudaFree(data_dev);
    cudaFree(greensfunc_dev);
    cudaFree(data_host);
    
    return 0;
}
