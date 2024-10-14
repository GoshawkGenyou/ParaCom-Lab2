#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#define DEBUG 1
#define SIZE 150000
#define PART 5000

int swap(int *a, int *b) {
	int t = *a;
	*a = *b;
	*b = t;
}

void shuffle(int *array, size_t n) {
    // Seed the random number generator
    srand((unsigned int)time(NULL));

    // Fisher-Yates shuffle
    for (size_t i = n - 1; i > 0; i--) {
        // Generate a random index from 0 to i
        size_t j = rand() % (i + 1);
        
        // Swap array[i] with array[j]
        unsigned long long temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

int printArr(int *arr) {
	for (int i = 0; i < 5; i++) {
		printf("%d ", arr[i]);
	}
	printf("... ");
	int mid = SIZE / 2;
	for(int i = mid - 5; i < mid + 5; i++) {
		printf("%d ", arr[i]);
	}
	printf("... ");
	for(int i = SIZE - 5; i < SIZE; i++) {
		printf("%d ", arr[i]);
	}
	printf("\n");
}

// Partition function
int partition(int arr[], int low, int high) {
// Choose the middle element as the pivot
    int mid = low + (high - low) / 2;
    int pivot = arr[mid];
    swap(&arr[mid], &arr[high]); // Move pivot to end for partitioning
    int i = low; // Pointer for the smaller element

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            swap(&arr[i], &arr[j]);
            i++;
        }
    }
    swap(&arr[i], &arr[high]); // Restore pivot
    return i; // Return the partition index
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        // Partitioning index
        int pi = partition(arr, low, high);
		//printf("low: %d, high: %d, pi: %d\n", low, high, pi);
        // Recursively sort elements before and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

// Parallel QuickSort function
void parallelQuickSort(int *arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        //printf("low: %d, high: %d, pi: %d\n", low, high, pi);
  		if (high - low > PART) {
            #pragma omp task shared(arr)
            parallelQuickSort(arr, low, pi - 1);
            
            #pragma omp task shared(arr)
            parallelQuickSort(arr, pi + 1, high);
            
            #pragma omp taskwait // Wait for tasks to complete
        } else {
            // If not creating tasks, sort sequentially
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }
}

void runParallel(int *arr, int n) {
	// low to high
	struct timeval start, end;
	double elapsed_time;
	gettimeofday(&start, NULL); // Start timing
	
	#pragma omp parallel
    {
    	#pragma omp single
    	{
        	parallelQuickSort(arr, 0, n - 1);
        }
    }
    
	gettimeofday(&end, NULL); // End timing
	elapsed_time = (end.tv_sec - start.tv_sec) + 
	                      (end.tv_usec - start.tv_usec) / 1e6;
	printArr(arr);
	printf("Parallel executed in %.6fs\n", elapsed_time);
}

void runNormal(int *arr, int n) {
	struct timeval start, end;
	double elapsed_time;
	gettimeofday(&start, NULL); // Start timing
	quickSort(arr, 0, n - 1);
	gettimeofday(&end, NULL); // End timing
	elapsed_time = (end.tv_sec - start.tv_sec) + 
	                      (end.tv_usec - start.tv_usec) / 1e6;
	printf("Normal executed in %.6fs\n", elapsed_time);
}

int generate_arrays(int *arr1, int *arr2, int n) {
	for (int i = 0; i < n; i++) {
		arr1[i] = i;
	}
	shuffle(arr1, n);
	for (int i = 0; i < n; i++) {
		arr2[i] = arr1[i];
	}
	return 0;
}

// Main function to execute QuickSort
int main() {
	omp_set_num_threads(12);
	int n;
	int *arr1, *arr2;
	if (!DEBUG) {
		scanf("%d", &n);
		arr1 = (int *)malloc(n * sizeof(int));
		for (int i = 0; i < n; i++) {
		    scanf("%d", &arr1[i]);
		}
    } else {
    	n = SIZE;
    	arr1 = (int *)malloc(n * sizeof(int));
    	if (arr1 == NULL) {
		    printf("Memory allocation failed for arr1\n");
		    return -1;
		}
    
    	arr2 = (int *)malloc(n * sizeof(int));
    	if (arr2 == NULL) {
		    printf("Memory allocation failed for arr2\n");
		    free(arr1);
		    return -1;
		}    	
		generate_arrays(arr1, arr2, n);
		printArr(arr1);
		printArr(arr2);
    }
    
    runParallel(arr1, n);
    if (DEBUG) {
    	runNormal(arr2, n);
    }
	printf("Size: %d\n", SIZE);
    free(arr1);
    if (DEBUG) free(arr2);
    return 0;
}
