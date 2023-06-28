/************************************************************************
*  Copyright 2015-2023 Lenovo
* 
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
* 
*      http://www.apache.org/licenses/LICENSE-2.0
* 
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
************************************************************************/

#include <stdio.h>
#include <string.h>
#include <mpi.h>

void Read_vector(double  local_a[], int local_n, int n, char vec_name[], int my_rank, MPI_Comm comm){
    double* a = NULL;
    int i;
    if(my_rank == 0){
        a = malloc(n*sizeof(double));
        printf("Enter the vector %s\n",vec_name);
        for(i = 0; i < n; i++){
            a[i] = i;
	}
        MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0, comm);
        free(a);
    } else{
        MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0, comm);
    }
}

void Print_vector(double local_b[], int local_n, int n, char title[], int my_rank, MPI_Comm comm){
    double* b = NULL;
    //int i;

    if(my_rank == 0){
        b = malloc(n*sizeof(double));
        MPI_Gather(local_b, local_n, MPI_DOUBLE, b, local_n, MPI_DOUBLE, 0, comm);
        printf("%s\n", title);printf("%f ", b[1126]);
        /*for(i = 0; i < n; i++)
            printf("%f ", b[i]);*/
        printf("\n");
        free(b);
    } else {
        MPI_Gather(local_b, local_n, MPI_DOUBLE, b, local_n, MPI_DOUBLE, 0, comm);
    }
}


void Parrel_vector_sum(double local_x[], double local_y[], double local_z[], int local_n){
    int local_i;
    
    for(local_i = 0; local_i < local_n; local_i++){
        local_z[local_i] = local_x[local_i] + local_y[local_i];
    }
}

int main(void){
  
  int comm_sz;
  int my_rank;
  int n = 1000;
  double *local_a,*local_b,*local_c;
  char A[100]="A";
  char B[100]="B";
  char C[500]="the 1126th element of'A + B' is ";
  sleep(180);
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  int local_n = n/comm_sz;
  local_a = malloc(local_n*sizeof(double));
  local_b = malloc(local_n*sizeof(double));
  local_c = malloc(local_n*sizeof(double));
  Read_vector(local_a, local_n, n, A, my_rank, MPI_COMM_WORLD);
  Read_vector(local_b, local_n, n, B, my_rank, MPI_COMM_WORLD);
  double start = MPI_Wtime();
  Parrel_vector_sum(local_a, local_b, local_c, local_n);
  Print_vector(local_c, local_n, n, C, my_rank, MPI_COMM_WORLD);
  double finish = MPI_Wtime();
  printf("the time needed is %e\n",finish-start);

  MPI_Finalize();
  return 0;
}
