Mi unidad
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <cassert>
#include <time.h>

#include "stb_image.h"
#include "stb_image_write.h"

struct Pixel {
	unsigned char r, g, b, a;
};

void RecolorImageCPU(unsigned char* imageRGBA, int width, int height) {
	float redTransform[3];
	float greenTransform[3];
	float blueTransform[3];

	redTransform[0] = 0.56667f;
	redTransform[1] = 0.55833f;
	redTransform[2] = 0.0f;

	greenTransform[0] = 0.43333f;
	greenTransform[1] = 0.44167f;
	greenTransform[2] = 0.24167f;

	blueTransform[0] = 0.0f;
	blueTransform[1] = 0.0f;
	blueTransform[2] = 0.75833f;


	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
			unsigned char newRedValue = (unsigned char)
				(ptrPixel->r * 0.0f + ptrPixel->g * 0.7f + ptrPixel->b * 0.3f);
			unsigned char newGreenValue = (unsigned char)
				(ptrPixel->r * 0.0f + ptrPixel->g * 1.1f + ptrPixel->b * 0.0f);
			unsigned char newBlueValue = (unsigned char)
				(ptrPixel->r * 0.0f + ptrPixel->g * 0.0f + ptrPixel->b * 1.1f);


			unsigned char filterRedValue = (unsigned char)
				(newRedValue * redTransform[0] + newGreenValue * greenTransform[0] + newBlueValue * blueTransform[0]);
			unsigned char filterGreenValue = (unsigned char)
				(newRedValue * redTransform[1] + newGreenValue * greenTransform[1] + newBlueValue * blueTransform[1]);
			unsigned char filterBlueValue = (unsigned char)
				(newRedValue * redTransform[2] + newGreenValue * greenTransform[2] + newBlueValue * blueTransform[2]);


			ptrPixel->r = filterRedValue;
			ptrPixel->g = filterGreenValue;
			ptrPixel->b = filterBlueValue;
			ptrPixel->a = 255;
		}
	}
}


__global__ void RecolorImageGPU(unsigned char* imageRGBA) {
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t idx = y * blockDim.x * gridDim.x + x;

	float redTransform[3];
	float greenTransform[3];
	float blueTransform[3];

	redTransform[0] = 0.56667f;
	redTransform[1] = 0.55833f;
	redTransform[2] = 0.0f;

	greenTransform[0] = 0.43333f;
	greenTransform[1] = 0.44167f;
	greenTransform[2] = 0.24167f;

	blueTransform[0] = 0.0f;
	blueTransform[1] = 0.0f;
	blueTransform[2] = 0.75833f;


	Pixel* ptrPixel = (Pixel*)&imageRGBA[idx * 4];

	unsigned char newRedValue = (unsigned char)
		(ptrPixel->r * 0.0f + ptrPixel->g * 0.7f + ptrPixel->b * 0.3f);
	unsigned char newGreenValue = (unsigned char)
		(ptrPixel->r * 0.0f + ptrPixel->g * 1.1f + ptrPixel->b * 0.0f);
	unsigned char newBlueValue = (unsigned char)
		(ptrPixel->r * 0.0f + ptrPixel->g * 0.0f + ptrPixel->b * 1.1f);
	

	unsigned char filterRedValue = (unsigned char)
		(newRedValue * redTransform[0] + newGreenValue * greenTransform[0] + newBlueValue * blueTransform[0]);
	unsigned char filterGreenValue = (unsigned char)
		(newRedValue * redTransform[1] + newGreenValue * greenTransform[1] + newBlueValue * blueTransform[1]);
	unsigned char filterBlueValue = (unsigned char)
		(newRedValue * redTransform[2] + newGreenValue * greenTransform[2] + newBlueValue * blueTransform[2]);
	

	ptrPixel->r = filterRedValue;
	ptrPixel->g = filterGreenValue;
	ptrPixel->b = filterBlueValue;
	ptrPixel->a = 255;
}



int main(int argc, char** argv) {

	//Check The Arguments
	if (argc < 2) {
		std::cout << "USAGE: EaseColor <filename> " << std::endl;
		return -1;
	}

	//Opens The Image
	int width, height, componentCount;
	std::cout << "Loading png file..." << std::endl;
	unsigned char* imageDataCPU = stbi_load(argv[1], &width, &height, &componentCount, 4);
	unsigned char* imageData = stbi_load(argv[1], &width, &height, &componentCount, 4);

	//int type = (int)argv[2];

	if (!imageData) {
		std::cout << "FAILED TO OPEN \"" << argv[1] << "\"";
		return -1;
	}
	std::cout << "DONE" << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;

	//Validates Image Size
	/*if(width % 32 || height % 32){
		//NOTE: Leaked Memory Of "imageData"
		std::cout << "WIDTH AND/OR HEIGHT IS NOT DIVIDABLE BY 32!";
		return -1;
	}*/

	std::cout << "Image Processing on CPU..." << std::endl;

	const clock_t begin_time = clock();
	RecolorImageCPU(imageDataCPU, width, height);
	double cpu_end = ((double)clock() - begin_time) / CLOCKS_PER_SEC;

	std::cout << "DONE" << std::endl;
	std::cout << std::endl;

	//Build Output Filename CPU
	std::string fileNameOutCPU = argv[1];
	fileNameOutCPU = fileNameOutCPU.substr(0, fileNameOutCPU.find_last_of(".")) + "_recolorCPU.png";

	//Write Image Back To Disk CPU
	std::cout << "Writting png to disk..." << std::endl;
	stbi_write_png(fileNameOutCPU.c_str(), width, height, 4, imageDataCPU, 4 * width);
	std::cout << "DONE" << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;


	//Copy Data to GPU
	std::cout << "Copy data to GPU..." << std::endl;
	unsigned char* ptrImageDataGpu = nullptr;
	assert(cudaMalloc(&ptrImageDataGpu, width * height * 4) == cudaSuccess);
	assert(cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice) == cudaSuccess);
	std::cout << " DONE" << std::endl;
	std::cout << std::endl;

	//Process Image On GPU
	std::cout << "Running CUDA Kernel..." << std::endl;
	dim3 blockSize(32, 32);
	dim3 gridSize(width / blockSize.x, height / blockSize.y);

	const clock_t gpu_start = clock();
	RecolorImageGPU << <gridSize, blockSize >> > (ptrImageDataGpu);
	double gpu_end = ((double)clock() - gpu_start) / CLOCKS_PER_SEC;

	auto err = cudaGetLastError();
	std::cout << " DONE" << std::endl;
	std::cout << std::endl;

	// Copy data from the gpu
	std::cout << "Copy data from GPU..." << std::endl;
	assert(cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
	std::cout << " DONE" << std::endl;
	std::cout << std::endl;

	//Build Output Filename
	std::string fileNameOut = argv[1];
	fileNameOut = fileNameOut.substr(0, fileNameOut.find_last_of(".")) + "_recolorGPU.png";

	//Write Image Back To Disk
	std::cout << "Writting png to disk..." << std::endl;
	stbi_write_png(fileNameOut.c_str(), width, height, 4, imageData, 4 * width);
	std::cout << "DONE" << std::endl;
	std::cout << std::endl;

	//TIME EXECUTIONS
	std::cout << "TIME EXECUTION IN CPU: \"" << cpu_end << "\"";
	std::cout << std::endl;
	
	std::cout << "TIME EXECUTION IN GPU: \"" << gpu_end << "\"";
	std::cout << std::endl;

	//Free Memory
	cudaFree(ptrImageDataGpu);
	stbi_image_free(imageData);
	stbi_image_free(imageDataCPU);
}