/*
*
* Authors: Fernando Amat
*  main_multiviewDeconvLR_multiGPU.cxx
*
*  Created on : July 27th, 2015
* Author : Fernando Amat
*
* \brief main executable to perform multiview deconvolution Lucy-Richardson with multi-GPU
*
*/

#include <iostream>
#include <string>
#include <chrono>
#include "multiGPUblockController.h"


using namespace std;
typedef std::chrono::high_resolution_clock Clock;

int main(int argc, const char** argv)
{
	auto tStart = Clock::now();
	auto t1 = Clock::now();
	auto t2 = Clock::now();

	//main inputs
	string filenameXML("C:/Users/Fernando/matlabProjects/deconvolution/CUDA/test/data/regDeconvParam.xml");
	int maxNumberGPU = -1;//default value

	if (argc > 1)
		filenameXML = string(argv[1]);
	if (argc > 2)
		maxNumberGPU = atoi(argv[2]);

	//--------------------------------------------------------
	//main object to control the deconvolution process
	cout << "Reading parameters from XML file" << endl;
	multiGPUblockController master(filenameXML);

	//check number of GPUs and the memory available for each of them
	master.queryGPUs(maxNumberGPU);
	master.debug_listGPUs();
		
	for (int ii = 0; ii < master.paramDec.Nviews; ii++)
	{
		t1 = Clock::now();
		cout << "Reading view " << ii << endl;
		master.full_img_mem.readImage(master.paramDec.fileImg[ii], -1);
		
		cout << "Reading PSF for view " << ii << endl;
		master.full_img_mem.readImage(master.paramDec.filePSF[ii], -1);

		t2 = Clock::now();
		std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

		t1 = Clock::now();
		cout << "Calculating constrast weights for each view" << endl;

		t2 = Clock::now();
		std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

		t1 = Clock::now();
		cout << "Applying affine transformation to view " << ii << endl;
		
		t2 = Clock::now();
		std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

		t1 = Clock::now();
		cout << "Applying affine transformation to weight array " << endl;

		t2 = Clock::now();
		std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

	}

	cout << "Calculating multiview deconvolution..." << endl;


	auto tEnd = Clock::now();
	std::cout << "Total process  took " << std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count() << " ms" << std::endl;

}