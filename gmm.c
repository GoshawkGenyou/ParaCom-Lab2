#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// treat as 1 dimensional array indexing
int gemm(int *A, int *B, int *C, int m, int k, int n){
	for (int r = 0; r < m; r++) {
		for (int c = 0; c < n; c++) {
			for(int i = 0; i < k; i++) {
				// Cmn = Amk * Bkn
				C[r * n + c] += A[r * k + i] * B[i * n + c];
			}
		}
	}
	return 0;
}

void printMatrix(int *array, int m, int k) {
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < k; c++) {
            printf("%d ", array[r * k + c]);
        }
        printf("\n");
    }
}

int naivegemm(void){
	int *A, *B, *C;
	int m, n, k;
	m = 2;
	k = 3;
	n = 4;
	scanf("%d %d", &m, &k);
	A = (int *)malloc(m * k * sizeof(int));
    if (A == NULL) {
        printf("Memory allocation failed for A\n");
        return -1;
    }

	for (int i = 0; i < m * k; i++) {
		scanf("%d", &A[i]);
	}
	
	int temp = k;
	scanf("%d %d", &k, &n);
	if (temp != k) {printf("Invalid matrix sizes for multiplication\n"); return -1;}
	B = (int *)malloc(k * n * sizeof(int));
	
    if (B == NULL) {
        printf("Memory allocation failed for B\n");
        free(A);
        return -1;
    }
    
    for (int i = 0; i < k * n; i++){
		scanf("%d", &B[i]);
	}
	
    C = (int *)malloc(m * n * sizeof(int));
    if (C == NULL) {
        printf("Memory allocation failed for C\n");
        free(A);
        free(B);
        return -1;
    }
    
	memset(C, 0, m * n * sizeof(int));
	gemm(A, B, C, m, k, n);
	
	printMatrix(C, m, n);
	
	free(A);
	free(B);
	free(C);
	return 0;
}


int openMPGEMM(int *A, int *B, int *C, int rstart, int cstart, int kstart, int m, int k, int n) {
	#pragma omp parallel
	{
		int *localC = (int *)calloc(m * n, sizeof(int));
	
		#pragma omp for collapse(2)
		for (int r = rstart; r < m; r++) {
			for (int c = cstart; c < n; c++) {
				int sum = 0;

				for(int i = kstart; i < k; i++) {
					// Cmn = Amk * Bkn
					sum += A[r * k + i] * B[i * n + c];
				}
				localC[r * n + c] += sum;
			}

			
		}
		
		// no concurrent writes
		#pragma omp critical
		{
			for (int r = rstart; r < m; r++) {
	            for (int c = cstart; c < n; c++) {
	            	if (!localC[r * n + c]) continue;
	                C[r * n + c] += localC[r * n + c];
	            }
		    }
		}
	
		free(localC);
	}
	
	return 0;
}

int opmgemm(void){
	int *A, *B, *C;
	int m, n, k;
	m = 2;
	k = 3;
	n = 4;
	scanf("%d %d", &m, &k);
	A = (int *)malloc(m * k * sizeof(int));
    if (A == NULL) {
        printf("Memory allocation failed for A\n");
        return -1;
    }

	for (int i = 0; i < m * k; i++) {
		scanf("%d", &A[i]);
	}
	
	int temp = k;
	scanf("%d %d", &k, &n);
	if (temp != k) {printf("Invalid matrix sizes for multiplication\n"); return -1;}
	B = (int *)malloc(k * n * sizeof(int));
	
    if (B == NULL) {
        printf("Memory allocation failed for B\n");
        free(A);
        return -1;
    }
    
    for (int i = 0; i < k * n; i++){
		scanf("%d", &B[i]);
	}
	
    C = (int *)malloc(m * n * sizeof(int));
    if (C == NULL) {
        printf("Memory allocation failed for C\n");
        free(A);
        free(B);
        return -1;
    }
    
	memset(C, 0, m * n * sizeof(int));
	
	// add aditional handling of chunking if needed for very large arrays
	openMPGEMM(A, B, C, 0, 0, 0, m, k, n);
	
	printMatrix(C, m, n);
	
	free(A);
	free(B);
	free(C);
	return 0;
}

int main(void) {
	//naivegemm();
	opmgemm();
}
