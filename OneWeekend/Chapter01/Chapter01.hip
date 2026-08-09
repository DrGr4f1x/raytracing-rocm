
#include <hip/hip_runtime.h>

#include <fstream>
#include <iostream>
#include <time.h>
#include <vector>

using namespace std;

// Image parameters
constexpr int IMAGE_WIDTH = 1280;
constexpr int IMAGE_HEIGHT = 720;

constexpr int error_exit_code = -1;
#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        const hipError_t error = condition;                                                 \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " \
                      << __FILE__ << ':' << __LINE__ << endl;                          \
            exit(error_exit_code);                                                     \
        }                                                                                   \
    }

__global__ void Render(float* fb, int maxX, int maxY)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i >= maxX || j >= maxY)
    {
        return;
    }

    int pixelIndex = j * maxX * 3 + i * 3;

    fb[pixelIndex + 0] = (float)i / maxX;
    fb[pixelIndex + 1] = (float)j / maxY;
    fb[pixelIndex + 2] = 0.2;
}

void SaveImage(const char* filename, const int width, const int height, const float* const fb)
{
    ofstream outfile;
    outfile.open(filename, ios::out | ios::trunc);

    outfile << "P3\n" << width << " " << height << "\n255\n";

    for (int j = height - 1; j >= 0; --j)
    {
        for (int i = 0; i < width; ++i)
        {
            const size_t pixelIndex = j * 3 * IMAGE_WIDTH + i * 3;
            const float r = fb[pixelIndex + 0];
            const float g = fb[pixelIndex + 1];
            const float b = fb[pixelIndex + 2];
            const int ir = (int)(255.99 * r);
            const int ig = (int)(255.99 * g);
            const int ib = (int)(255.99 * b);
            outfile << ir << " " << ig << " " << ib << "\n";
        }
    }

    outfile.close();
}

int main()
{
    // Allocate framebuffer
    const int numPixels = IMAGE_WIDTH * IMAGE_HEIGHT;
    const size_t fbSize = 3 * numPixels * sizeof(float);
    
    float* fb_device = nullptr;
    HIP_CHECK(hipMalloc((void**)&fb_device, fbSize));

    // Block size
    const int tx = 8;
    const int ty = 8;

    float      kernel_ms{};
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Record the start event.
    HIP_CHECK(hipEventRecord(start, hipStreamDefault));

    // Render our buffer
    dim3 blocks(IMAGE_WIDTH / tx + 1, IMAGE_HEIGHT / ty + 1);
    dim3 threads(tx, ty);
    Render<<<blocks, threads, 0 /*shared memory*/, hipStreamDefault>>> (fb_device, IMAGE_WIDTH, IMAGE_HEIGHT);

    // Check if the kernel launch was successful.
    HIP_CHECK(hipGetLastError());

    // Record the stop event and wait until the kernel execution finishes.
    HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
    HIP_CHECK(hipEventSynchronize(stop));

    HIP_CHECK(hipEventElapsedTime(&kernel_ms, start, stop));
    cerr << "Render time: " << kernel_ms / 1000.0f << " seconds" << endl;

    // Copy from device to cpu
    vector<float> fb_cpu(3 * numPixels);
    HIP_CHECK(hipMemcpy(fb_cpu.data(), fb_device, 3 * numPixels * sizeof(float), hipMemcpyDeviceToHost));

    // Save image
    SaveImage("image.ppm", IMAGE_WIDTH, IMAGE_HEIGHT, fb_cpu.data());

    // Free events variables and device memory.
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(fb_device));

    return 0;
}