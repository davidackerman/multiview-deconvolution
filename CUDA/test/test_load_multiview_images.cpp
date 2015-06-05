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
	string filePattern = "write something";

	//declare object
	multiviewImage<uint16_t> img;

	//read object
	for (int ii = 0; ii < numViews; ii++)
	{
		string filename = recoverFilenamePatternFromString(filePattern, ii);
		img.readImage(filename, -1);
	}

	//release memory

}