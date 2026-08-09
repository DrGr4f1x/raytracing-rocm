
#include <hip/hip_runtime.h>

#include <fstream>
#include <iostream>
#include <time.h>
#include <vector>

#include "Ray.h"

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


__device__ Vec3 Color(const Ray& r)
{
    Vec3 unitDirection = Normalize(r.Direction());
    float t = 0.5f * (unitDirection.Y() + 1.0f);
    return (1.0f - t) * Vec3 { 1.0f, 1.0f, 1.0f } + t * Vec3{ 0.5f, 0.7f, 1.0f };
}


__global__ void Render(Vec3* fb, int maxX, int maxY, Vec3 lowerLeft, Vec3 horizontal, Vec3 vertical, Vec3 origin)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i >= maxX || j >= maxY)
    {
        return;
    }

    int pixelIndex = j * maxX + i;
    float u = (float)i / (float)maxX;
    float v = (float)j / (float)maxY;

    Ray r{ origin, lowerLeft + u * horizontal + v * vertical };

    fb[pixelIndex] = Color(r);
}

void SaveImage(const char* filename, const int width, const int height, const Vec3* const fb)
{
    ofstream outfile;
    outfile.open(filename, ios::out | ios::trunc);

    outfile << "P3\n" << width << " " << height << "\n255\n";

    for (int j = height - 1; j >= 0; --j)
    {
        for (int i = 0; i < width; ++i)
        {
            const size_t pixelIndex = j * IMAGE_WIDTH + i;
            const Vec3& pixel = fb[pixelIndex];
            const int ir = (int)(255.99 * pixel.R());
            const int ig = (int)(255.99 * pixel.G());
            const int ib = (int)(255.99 * pixel.B());
            outfile << ir << " " << ig << " " << ib << "\n";
        }
    }

    outfile.close();
}

int main()
{
    // Allocate framebuffer
    const int numPixels = IMAGE_WIDTH * IMAGE_HEIGHT;
    const size_t fbSize = numPixels * sizeof(Vec3);
    
    Vec3* fb = nullptr;
    HIP_CHECK(hipMallocManaged((void**)&fb, fbSize));

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
    Render<<<blocks, threads, 0 /*shared memory*/, hipStreamDefault>>> (fb, IMAGE_WIDTH, IMAGE_HEIGHT,
        Vec3{ -2.0f, -1.0f, -1.0f },
        Vec3{ 4.0f, 0.0f, 0.0f },
        Vec3{ 0.0f, 2.0f, 0.0f },
        Vec3{ 0.0f, 0.0f, 0.0f });

    // Check if the kernel launch was successful.
    HIP_CHECK(hipGetLastError());

    // Record the stop event and wait until the kernel execution finishes.
    HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
    HIP_CHECK(hipEventSynchronize(stop));

    HIP_CHECK(hipEventElapsedTime(&kernel_ms, start, stop));
    cerr << "Render time: " << kernel_ms / 1000.0f << " seconds" << endl;

    // Save image
    SaveImage("image.ppm", IMAGE_WIDTH, IMAGE_HEIGHT, fb);

    // Free events variables and device memory.
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(fb));

    return 0;
}