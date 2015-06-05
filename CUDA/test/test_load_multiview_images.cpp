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
		string filename = img.recoverFilenamePatternFromString(filePattern, ii);
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