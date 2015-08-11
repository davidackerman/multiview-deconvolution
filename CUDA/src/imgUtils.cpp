/*
* Copyright(C) 2015 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat
*  imgUtils.cpp
*
*  Created on : June 11th, 2015
* Author : Fernando Amat
*
* \brief random functions that are generically useful
*/

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#endif

#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <cstring>
#include "imgUtils.h"


using namespace std;

template<class imgType>
imgType* fa_padArrayWithZeros(const imgType* im, const std::int64_t *dimsNow, const std::uint32_t *dimsAfterPad, int ndims)
{

	if (ndims != 3)
	{
		cout << "TODO:ERROR: padArrayWithZeros: function not ready for other values of ndims except for " << 3 << endl;
		exit(3);
	}

	int64_t nImg = 1;
	for (int ii = 0; ii < ndims; ii++)
	{
		if (dimsNow[ii] > dimsAfterPad[ii])
		{
			cout << "ERROR: padArrayWithZeros: new dimensions are smaller than current dimensions" << endl;
			return NULL;
		}
		nImg *= (int64_t)(dimsAfterPad[ii]);
	}



	imgType* p = new imgType[nImg];
	memset(p, 0, sizeof(imgType)* nImg);	

    //copy "lines" of x	
	size_t lineSize = dimsNow[0] * sizeof(imgType);
	int64_t idx = 0;
	int64_t count = 0;	
	for (int64_t zz = 0; zz < dimsNow[2]; zz++)
	{        
		idx = dimsAfterPad[0] * dimsAfterPad[1] * zz;		
		for (int64_t yy = 0; yy < dimsNow[1]; yy++)
		{
			//update for new array
			//idx = dimsAfterPad[0] * ( yy + dimsAfterPad[1] * zz);
			//update for new array
			//count = dimsNow[0] * (yy + dimsNow[1] * zz);

			//copy elements
			memcpy(&(p[idx]), &(im[count]), lineSize );
			
            //update counters
            idx += dimsAfterPad[0];
			count += dimsNow[0];
		}
	}

	return p;
}

//====================================================
std::string generateTempFilename(const char* prefix)
{
#ifdef _WIN32
	char s[MAX_PATH];
	if (GetTempPath(MAX_PATH, s)) {
		//std::cout << "GetTempPath() returned <" << s << ">\n";
	}
	else {
		std::cout << "ERROR: generateTempFilename: GetTempPath() failed with 0x" << std::hex << GetLastError() << "\n";
	}

	char* name = _tempnam("c:\\tmp", prefix);
	std::string nameS(name); 
	
#else //unix systems
	char *folder = getenv("TMPDIR");
	if (folder == 0)
		folder = "/tmp";

	char *name = tempnam(folder, prefix);

	std::string nameS(name);		
	free(folder);
#endif
	
	free(name);
	std::cout << "============DEBUGGING:Temporary filename is " << nameS << std::endl;
	
	return nameS;

}

//=============================================================
template float* fa_padArrayWithZeros<float>(const float* im, const std::int64_t *dimsNow, const std::uint32_t *dimsAfterPad, int ndims);
template std::uint16_t* fa_padArrayWithZeros<std::uint16_t>(const std::uint16_t* im, const std::int64_t *dimsNow, const std::uint32_t *dimsAfterPad, int ndims);
template std::uint8_t* fa_padArrayWithZeros<std::uint8_t>(const std::uint8_t* im, const std::int64_t *dimsNow, const std::uint32_t *dimsAfterPad, int ndims);
