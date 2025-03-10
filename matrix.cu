#include "matrix.h"
#include <stdlib.h>
#include <string.h>
#include "error.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

void destroy_matrix(matrix_t *m)
{
    //printf("free %p %p\n", m, m->m);

    CHECK_ERROR(cudaFree(m->m));
    CHECK_ERROR(cudaFree(m));
}

matrix_t * alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t * res;
    CHECK_ERROR(cudaMallocManaged((void**)&res, sizeof(matrix_t)));
    //matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );

    //res->m = (double *) calloc(columns * rows, sizeof(double));
    CHECK_ERROR(cudaMallocManaged((void**)&res->m, rows*columns*sizeof(double)));
    CHECK_ERROR(cudaMemset(res->m, 0, rows*columns*sizeof(double)));

    res->columns = columns;
    res->rows = rows;
    return res;
}

void print_matrix(matrix_t *m, bool is_short){
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    for (int row = 0; row < lim_rows; row ++)
    {
        for (int col = 0; col < lim_col; col ++)
        {
            printf("%.2lf ", m->m[col + row * m->columns]);
        }
        if (is_short && lim_col != m->columns) printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != m->rows) printf("...\n");
}

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
            res->m[idx] = m1->m[idx] * m2->m[idx];
    }
}

__global__
void hadamard_product_GPU(matrix_t *m1, matrix_t *m2, matrix_t *res) {

    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m1->rows && col < m2->columns) {
        res->m[row*res->columns+col] = m1->m[row*m1->columns+col] * m2->m[row*m2->columns+col];
    } 

}

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    { 
        res->m[idx] = m1->m[idx] + m2->m[idx];
    }
}

__global__
void matrix_sum_GPU(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*m1->columns+col;

    if (row < m1->rows && col < m1->columns) {
        res->m[idx] = m1->m[idx] + m2->m[idx];
    } 
}

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));
             
    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = m1->m[idx] - m2->m[idx];
    }
}


__global__
void matrix_minus_GPU(matrix_t *m1, matrix_t *m2, matrix_t *res) {

    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*m1->columns+col;

    if (row < m1->rows && col < m1->columns) {
        res->m[idx] = m1->m[idx] - m2->m[idx];
    } 

}

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->rows)  &&
             (m1->rows == res->rows)    &&
             (m2->columns == res->columns));

    for (int row = 0; row < m1->rows; row ++)
    {
        for (int col = 0; col < m2->columns; col ++)
        {
            int idx = col + row * m2->columns;
            double var = 0.0;

            for (int ii = 0; ii < m1->columns; ii++)
            {
                var += m1->m[ii + row * m1->columns] * m2->m[col + ii * m2->columns];
            }

            res->m[idx] = var;
        }
    }
}

__global__
void matrix_dot_GPU(matrix_t *m1, matrix_t *m2, matrix_t *res) {

    assert ( (m1->columns == m2->rows)  &&
    (m1->rows == res->rows)    &&
    (m2->columns == res->columns));

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m1->rows && col < m2->columns)
    {
        float sum = 0;
        for (int i = 0; i < m1->columns; i++)
        {
            sum += m1->m[row*m1->columns+i] * m2->m[i*m2->columns+col];
        }
        res->m[row*m2->columns+col] = sum;
    } 

}

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = f(m1->m[idx]);
    }
}

void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert ( (m1->columns == res->rows) &&             
             (m1->rows == res->columns));
    
    for (int row = 0; row < m1->rows; row++)
    {
        for (int col = 0; col < m1->columns; col ++)
        {
            res->m[row + col * m1->rows] = m1->m[col + row * m1->columns];
        }
    }
}

__global__
void matrix_transpose_GPU(matrix_t *m1, matrix_t *res) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m1->rows && col < m1->columns)
    {
        res->m[col*m1->rows+row] = m1->m[row*m1->columns+col];
    } 

}


void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    for (int idx = 0; idx < m1->columns*m1->rows; idx ++)
    {
        res->m[idx] = m1->m[idx] * s;
    }
}

__global__
void matrix_scalar_GPU(matrix_t *m1, double s, matrix_t *res) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*res->columns+col;

    if (row < m1->rows && col < m1->columns) {
        res->m[idx] = m1->m[idx] * s;
    } 

}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));     
}