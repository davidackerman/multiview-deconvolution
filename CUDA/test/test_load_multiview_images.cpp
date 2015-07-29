/*
*
* Authors: Fernando Amat
*  test_load_multiview_images.cpp
*
*  Created on : June 5th, 2015
* Author : Fernando Amat
*
* \brief testing loading multiview image
*
*/

#include <iostream>
#include <cstdint>
#include "multiviewImage.h"


using namespace std;
int main(int argc, const char** argv)
{
	std::cout << "test load multiview images running..." << std::endl;

	//parameters
	int numViews = 4;
	string filepath("C:/Users/Fernando/matlabProjects/deconvolution/CUDA/test/data/");

	string filePattern( filepath + "imReg_?.klb" );

	//declare object
	multiviewImage<float> img;

	//read object
	for (int ii = 0; ii < numViews; ii++)
	{
		string filename = img.recoverFilenamePatternFromString(filePattern, ii+1);
		int err = img.readImage(filename, -1);

		if (err != 0)
		{
			cout << "ERROR: loading image " << filename << endl;
			return err;
		}
	}

	//release memory


	std::cout << "...OK" << endl;
	return 0;
}