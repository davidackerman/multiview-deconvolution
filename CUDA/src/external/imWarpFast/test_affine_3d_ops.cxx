#include <stdio.h>
#include "affine_transform_3d_single.h"

using namespace std;
int main(int argc, const char** argv)
{

	float A[AFFINE_3D_MATRIX_SIZE] = { 1.0f, 0.5f, 0.25f, -10.0f, 0.0f, 0.9f, 0.35f, 7.2f, -0.2f, -0.3f, 2.0f, 200.0f, 0.0f, 0.0f, 0.0f, 1.0f};
	float B[AFFINE_3D_MATRIX_SIZE] = { 3.0f, 0.75f, 0.25f, -11.0f, 0.0f, 1.9f, 0.45f, -5.2f, 0.2f, -0.35f, 1.0f, 20.0f, 0.0f, 0.0f, 0.0f, 1.0f };
	float C[AFFINE_3D_MATRIX_SIZE];

	if (!affine_3d_isAffine(A))
	{
		printf("ERROR: A is not an affine matrix\n");
		return 2;
	}
	if (!affine_3d_isAffine(B))
	{
		printf("ERROR: B is not an affine matrix\n");
		return 2;
	}

	printf("Matrix A:\n");
	affine3d_printMatrix(A);
	printf("\n\n");

	printf("Matrix B:\n");
	affine3d_printMatrix(B);


	affine_3d_compose(A, B, C);
	printf("Matrix A*B:\n");
	affine3d_printMatrix(C);
	printf("\n\n");



	affine_3d_inverse(A, C);
	printf("Matrix inv(A):\n");
	affine3d_printMatrix(C);
	printf("\n\n");


	affine_3d_transpose(A, C);
	printf("Matrix transpose(A):\n");
	affine3d_printMatrix(C);
	printf("\n\n");


	return 0;
}