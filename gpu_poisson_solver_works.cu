#include <stdio.h>
#include <math.h>
#include <cufft.h>

__global__ void multiplyGreensFunc(cufftComplex* data, cufftReal* greensfunc, int N) {
    for(int i = blockDim.x*blockIdx.x+threadIdx.x; i < N; i += gridDim.x*blockDim.x) {
        data[i].x *= greensfunc[i];
        data[i].y *= greensfunc[i];
    }
}

int main(int argc, char** argv) {    
    int Nx = atoi(argv[1]);
    int Ny = atoi(argv[2]);
    int Nz = atoi(argv[3]);
    float h = atof(argv[4]);

    cufftHandle plan_fft;
    cufftHandle plan_ifft;
    cufftComplex* data_dev;
    cufftReal* greensfunc_dev;
    cufftComplex* data_host;
    cufftReal* data_real_host;
    cufftReal* greensfunc_host;
    
printf("Nx/2+1=%f\nNy/2+1=%f\nNz/2+1=%f\n", (float) (Nx/2+1), (float) (Ny/2+1), (float) (Nz/2+1));
    
    cudaMalloc((void**) &data_dev, sizeof(cufftComplex)*Nx*Ny*(Nz/2+1));
    cudaMalloc((void**) &greensfunc_dev, sizeof(cufftReal)*Nx*Ny*(Nz/2+1));
    cudaMallocHost((void**) &data_host, sizeof(cufftComplex)*Nx*Ny*(Nz/2+1));
    cudaMallocHost((void**) &greensfunc_host, sizeof(cufftReal)*Nx*Ny*(Nz/2+1));
    data_real_host = (cufftReal*) data_host;
    
    /* greens function */
    for(int x = 0; x < Nx; x++)
        for(int y = 0; y < Ny; y++)
            for(int z = 0; z < Nz/2+1; z++)
                if(x == 0 && y == 0 && z == 0)
                    greensfunc_host[Ny*(Nz/2+1)*x+(Nz/2+1)*y+z] = 0.0; //setting 0th fourier mode to 0 enforces charge neutrality
                else
                    greensfunc_host[Ny*(Nz/2+1)*x+(Nz/2+1)*y+z] = -0.5 * h * h / (cos(2.0*M_PI*x/(cufftReal)Nx) + cos(2.0*M_PI*y/(cufftReal)Ny) + cos(2.0*M_PI*z/(cufftReal)Nz) - 3.0);
    
    cudaMemcpy(greensfunc_dev, greensfunc_host, sizeof(cufftReal)*Nx*Ny*(Nz/2+1), cudaMemcpyHostToDevice);
    
printf("Nx=%d\nNy=%d\nNz=%d\n", Nx, Ny, Nz);
    
    /* charge density */
    for(int z = 0; z < Nz; z++)
        for(int y = 0; y < Ny; y++)
            for(int x = 0; x < Nx; x++)
                if(y == 0)
                    data_real_host[Ny*Nz*x+Nz*y+z] = 1.0;
                else
                    data_real_host[Ny*Nz*x+Nz*y+z] = 0.0;
    
    cudaMemcpy(data_dev, data_host, sizeof(cufftComplex)*Nx*Ny*(Nz/2+1), cudaMemcpyHostToDevice);

    /* create 3D FFT plans */
    cufftPlan3d(&plan_fft, Nx, Ny, Nz, CUFFT_R2C);
    cufftSetCompatibilityMode(plan_fft, CUFFT_COMPATIBILITY_NATIVE);
    cufftPlan3d(&plan_ifft, Nx, Ny, Nz, CUFFT_C2R);
    cufftSetCompatibilityMode(plan_ifft, CUFFT_COMPATIBILITY_NATIVE);
    
    /* FFT in place */
    cufftExecR2C(plan_fft, (cufftReal*) data_dev, data_dev);
    
    
    
    cudaMemcpy(data_host, data_dev, sizeof(cufftComplex)*Nx*Ny*(Nz/2+1), cudaMemcpyDeviceToHost);
    
    FILE* fp = fopen("fft.vtk", "w");
    
    fprintf(fp, "# vtk DataFile Version 2.0\ncharge_fft\nASCII\n\nDATASET STRUCTURED_POINTS\nDIMENSIONS %u %u %u\nORIGIN 0 0 0\nSPACING %f %f %f\n\nPOINT_DATA %u\nSCALARS charge_fft float 2\nLOOKUP_TABLE default\n", Nx, Ny, Nz, h, h, h, Nx*Ny*Nz);
    
    for(int z = 0; z < Nz; z++) {
        for(int y = 0; y < Ny; y++)
            for(int x = 0; x < Nx; x++)
                if(z >= Nz/2+1)
                    fprintf(fp, " %f %f", data_host[Ny*(Nz/2+1)*(Nx-x-1)+(Nz/2+1)*(Ny-y-1)+(Nz-z-1)].x/sqrt(Nx*Ny*Nz), -data_host[Ny*(Nx/2+1)*(Nz-z-1)+(Nx/2+1)*(Ny-y-1)+(Nx-x-1)].y/sqrt(Nx*Ny*Nz));
                else
                    fprintf(fp, " %f %f", data_host[Ny*(Nz/2+1)*x+(Nz/2+1)*y+z].x/sqrt(Nx*Ny*Nz), data_host[Ny*(Nx/2+1)*z+(Nx/2+1)*y+x].y/sqrt(Nx*Ny*Nz));
        
        fprintf(fp, "\n");
    }
    
    fclose(fp);
    
    
    
    /* multiplying with greens function */
    multiplyGreensFunc<<<14,32*32>>>(data_dev, greensfunc_dev, Nx*Ny*(Nz/2+1)); //18-fold occupation seems to be optimal for the GT520 and 32-fold for the C2050
    
    /* inverse FFT in place */
    cufftExecC2R(plan_ifft, data_dev, (cufftReal*) data_dev);
    
    cudaMemcpy(data_host, data_dev, sizeof(cufftComplex)*Nx*Ny*(Nz/2+1), cudaMemcpyDeviceToHost);
    
    fp = fopen("potential.vtk", "w");
    
    fprintf(fp, "# vtk DataFile Version 2.0\npotential\nASCII\n\nDATASET STRUCTURED_POINTS\nDIMENSIONS %u %u %u\nORIGIN 0 0 0\nSPACING %f %f %f\n\nPOINT_DATA %u\nSCALARS potential float 1\nLOOKUP_TABLE default\n", Nx, Ny, Nz, h, h, h, Nx*Ny*Nz);

    for(int z = 0; z < Nz; z++) {
        for(int y = 0; y < Ny; y++)
            for(int x = 0; x < Nx; x++)
                fprintf(fp, " %f", data_real_host[Ny*Nz*x+Nz*y+z]/(Nx*Ny*Nz));
        
        fprintf(fp, "\n");
    }

    fclose(fp);

    /* cleanup */
    cufftDestroy(plan_fft);
    cufftDestroy(plan_ifft);
    
    cudaFree(data_dev);
    cudaFree(greensfunc_dev);
    cudaFree(data_host);
    cudaFree(greensfunc_host);
    
    return 0;
}
