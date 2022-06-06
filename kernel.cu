#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <cassert>

#include "stb_image.h"
#include "stb_image_write.h"

struct Pixel{
	unsigned char r, g, b, a;
};

void RecolorImageCPU(unsigned char* imageRGBA, int width, int height){
	for(int y = 0; y < height; y++){
		for(int x = 0; x < width; x++){
			Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
            unsigned char pixelValue = (unsigned char)(ptrPixel->r * 0.5666f + ptrPixel->g * 0.5583f + ptrPixel->b * 0.7583f);
            ptrPixel->r = pixelValue;
            ptrPixel->g = pixelValue;
            ptrPixel->b = pixelValue;
            ptrPixel->a = 255;
		}
	}
}

//TODO FIXED RECOLOR IN CPU

void FixedImageCPU(unsigned char* imageRGBA, int width, int height){
	for(int y = 0; y < height; y++){
		for(int x = 0; x < width; x++){
			Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
            unsigned char pixelValue = (unsigned char)(ptrPixel->r * 0.5666f + ptrPixel->g * 0.5583f + ptrPixel->b * 0.7583f);
            ptrPixel->r = pixelValue;
            ptrPixel->g = pixelValue;
            ptrPixel->b = pixelValue;
            ptrPixel->a = 255;
		}
	}
}

__global__ void RecolorImageGPU(unsigned char* imageRGBA){
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t idx = y * blockDim.x * gridDim.x + x;

	Pixel* ptrPixel = (Pixel*)&imageRGBA[idx * 4];

    unsigned char redValue = (unsigned char)
        (ptrPixel->r * 0.56667f  + ptrPixel->g * 0.43333f + ptrPixel->b * 0.0f);
	unsigned char greenValue = (unsigned char)
        (ptrPixel->r * 0.55833f  + ptrPixel->g * 0.44167f + ptrPixel->b * 0.0f);
	unsigned char blueValue = (unsigned char)
        (ptrPixel->r * 0.0f  + ptrPixel->g * 0.24167f + ptrPixel->b * 0.75833f);

    ptrPixel->r = redValue;
    ptrPixel->g = greenValue;
    ptrPixel->b = blueValue;
    ptrPixel->a = 255;
}

//TODO FIXED RECOLOR IN GPU

__global__ void fixedImageGPU(unsigned char* imageRGBA){
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t idx = y * blockDim.x * gridDim.x + x;

	Pixel* ptrPixel = (Pixel*)&imageRGBA[idx * 4];

    unsigned char redValue = (unsigned char)
        (ptrPixel->r * 0.56667f  + ptrPixel->g * 0.43333f + ptrPixel->b * 0.0f);
	unsigned char greenValue = (unsigned char)
        (ptrPixel->r * 0.55833f  + ptrPixel->g * 0.44167f + ptrPixel->b * 0.0f);
	unsigned char blueValue = (unsigned char)
        (ptrPixel->r * 0.0f  + ptrPixel->g * 0.24167f + ptrPixel->b * 0.75833f);

    ptrPixel->r = redValue;
    ptrPixel->g = greenValue;
    ptrPixel->b = blueValue;
    ptrPixel->a = 255;
}

int main(int argc, char** argv){

	//Check The Arguments
	if(argc < 2){
		std::cout << "USAGE: image_to_grey <filename>" << std::endl;
		return -1;
	}

	//Opens The Image
	int width, height, componentCount;
    std::cout << "Loading png file..." << std::endl;
    unsigned char* imageData = stbi_load(argv[1], &width, &height, &componentCount, 4);

	if(!imageData){
		std::cout << "FAILED TO OPEN \"" << argv[1] << "\"";
		return -1;
	}
	std::cout << "DONE" << std::endl;

	//Validates Image Size
	/*if(width % 32 || height % 32){
		//NOTE: Leaked Memory Of "imageData"
		std::cout << "WIDTH AND/OR HEIGHT IS NOT DIVIDABLE BY 32!";
		return -1;
	}*/

	/*Process Image On CPU
	std::cout << "Image Processing..." << std::endl;
	RecolorImageCPU(imageData, width, height);
	std::cout << "DONE" << std::endl;
	*/

	//Copy Data to GPU
	std::cout << "Copy data to GPU...";
    unsigned char* ptrImageDataGpu = nullptr;
    assert(cudaMalloc(&ptrImageDataGpu, width * height * 4) == cudaSuccess);
    assert(cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    std::cout << " DONE" << std::endl;

	//Process Image On GPU
	std::cout << "Running CUDA Kernel...";
    dim3 blockSize(32, 32);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    RecolorImageGPU<<<gridSize, blockSize>>>(ptrImageDataGpu);
    auto err = cudaGetLastError();
    std::cout << " DONE" << std::endl; 

	// Copy data from the gpu
    std::cout << "Copy data from GPU...";
    assert(cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    std::cout << " DONE" << std::endl;

	//Build Output Filename
	std::string fileNameOut = argv[1];
	fileNameOut = fileNameOut.substr(0, fileNameOut.find_last_of(".")) + "_colorBlind.png";

	//Write Image Back To Disk
	std::cout << "Writting png to disk..." << std::endl;
	stbi_write_png(fileNameOut.c_str(), width, height, 4, imageData, 4 * width);
	std::cout << "DONE" << std::endl;

	//Free Memory
	cudaFree(ptrImageDataGpu);
	stbi_image_free(imageData);
}