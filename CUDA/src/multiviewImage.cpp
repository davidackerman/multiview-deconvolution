/*
* Copyright (C) 2015 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat
*  multiviewImage.cpp
*
*  Created on: June 5th, 2015
*      Author: Fernando Amat
*
* \brief 
*/

#include "multiviewImage.h"
#include "imgeIO.h"

using namespace std;

template<class imgType>
string multiviewImage<imgType>::recoverFilenamePatternFromString(const string& imgPath, int frame)
{

	string imgRawPath(imgPath);
	size_t found = imgRawPath.find_first_of("?");
	while (found != string::npos)
	{
		int intPrecision = 0;
		while ((imgRawPath[found] == '?') && found != string::npos)
		{
			intPrecision++;
			found++;
			if (found >= imgRawPath.size())
				break;

		}


		char bufferTM[16];
		switch (intPrecision)
		{
		case 1:
			sprintf(bufferTM, "%.1d", frame);
			break;
		case 2:
			sprintf(bufferTM, "%.2d", frame);
			break;
		case 3:
			sprintf(bufferTM, "%.3d", frame);
			break;
		case 4:
			sprintf(bufferTM, "%.4d", frame);
			break;
		case 5:
			sprintf(bufferTM, "%.5d", frame);
			break;
		case 6:
			sprintf(bufferTM, "%.6d", frame);
			break;
		case 7:
			sprintf(bufferTM, "%.7d", frame);
			break;
		case 8:
			sprintf(bufferTM, "%.28", frame);
			break;
		default:
			cout << "ERROR:recoverFilenamePatternFromString: not prepared for so many ??? in the file pattern"
		}
		string itoaTM(bufferTM);

		found = imgRawPath.find_first_of("?");
		imgRawPath.replace(found, intPrecision, itoaTM);


		//find next ???
		found = imgRawPath.find_first_of("?");
	}

	return imgRawPath;

}

//===========================================================================================
template<class imgType>
int multiviewImage<imgType>::readImage(const std::string& filename, int pos)
{
		
}