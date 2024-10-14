#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define GENERATE 1
#define DEBUG 1

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
    	printf("Row %d: \n", r);
        for (int c = 0; c < k; c++) {
            printf("%d ", array[r * k + c]);
        }
        printf("\n");
    }
}

int readMatrixFromFile(const char* filename, int *A, int *B, int *C, int &m, int &k, int &n) {
// Open the file for reading
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return -1;
    }
    
	fscanf(file, "%d %d", &m, &k);
	A = (int *)malloc(m * k * sizeof(int));
    if (A == NULL) {
        printf("Memory allocation failed for A\n");
        fclose(file);
        return -1;
    }

	for (int i = 0; i < m * k; i++) {
		fscanf(file, "%d", &A[i]);
	}
	
	int temp = k;
	fscanf(file, "%d %d", &k, &n);
	if (temp != k) {printf("Invalid matrix sizes for multiplication\n"); return -1;}
	B = (int *)malloc(k * n * sizeof(int));
	
    if (B == NULL) {
        printf("Memory allocation failed for B\n");
        free(A);
        fclose(file);
        return -1;
    }
    
    for (int i = 0; i < k * n; i++){
		fscanf(file, "%d", &B[i]);
	}
	
    C = (int *)calloc(m * n, sizeof(int));
    if (C == NULL) {
        printf("Memory allocation failed for C\n");
        free(A);
        free(B);
        fclose(file);
        return -1;
    }
    fclose(file);
}


int naivegemm(const char* filename){
	int *A, *B, *C;
	int m, n, k;
	
	if (readMatrixFromFile(filename, A, B, C, m, n, k) == -1) return -1;
    
	struct timeval start, end;
	double elapsed_time;
	gettimeofday(&start, NULL); // Start timing
	gemm(A, B, C, m, k, n);
	gettimeofday(&end, NULL); // End timing
	elapsed_time = (end.tv_sec - start.tv_sec) + 
	                      (end.tv_usec - start.tv_usec) / 1e6;
	printf("Normal executed in %.6fs\n", elapsed_time);
	
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

int opmgemm(const char* filename){
	int *A, *B, *C;
	int m, n, k;
	
	
	if (readMatrixFromFile(filename, A, B, C, &m, &n, &k) == -1) return -1;
	
	// add aditional handling of chunking if needed for very large arrays
	openMPGEMM(A, B, C, 0, 0, 0, m, k, n);
	
	printMatrix(C, m, n);
	
	free(A);
	free(B);
	free(C);
	return 0;
}

// Function to write two random matrices to a file
void writeRandomMatricesToFile(const char *filename, int m, int k, int n) {
    // Seed the random number generator
    srand(time(NULL));

    // Open the file for writing
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    // Write first matrix (m x k)
    fprintf(file, "%d %d\n", m, k);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            int value = 25 - (rand() % 50); // Random value between 0 and 99
            fprintf(file, "%d ", value);
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");

    // Write second matrix (k x n)
    fprintf(file, "%d %d\n", k, n);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            int value = rand() % 100; // Random value between 0 and 99
            fprintf(file, "%d ", value);
        }
        fprintf(file, "\n");
    }

    // Close the file
    fclose(file);
}

int main(int argc, char *argv[]) {
	if (DEBUG) {
		if (argc != 4) {
		    fprintf(stderr, "Usage: %s <m> <k> <n>\n", argv[0]);
		    return 1;
		}
		
		const char *filename = "matrix.txt";

		// Convert command line arguments to integers
		int m = atoi(argv[1]);
		int k = atoi(argv[2]);
		int n = atoi(argv[3]);
		
		printf("Matrix dimensions: A (%d x %d), B (%d x %d)\n", m, k, k, n);
		writeRandomMatricesToFile(filename, m, k, n);
    }
	if (DEBUG) {
		naivegemm(filename);
		opmgemm(filename);
	}
}
