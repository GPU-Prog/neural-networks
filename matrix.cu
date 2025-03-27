#include "matrix.h"
#include <stdlib.h>
#include <string.h>
#include "error.h"
#include <cublas_v2.h>


#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define TILE_SIZE 16

/**
 * Libère la mémoire GPU associée à une matrice. 
 * Désalloue d'abord les données, puis la structure.
 */
void destroy_matrix(matrix_t *m)
{
    //printf("free %p %p\n", m, m->m);

    CHECK_ERROR(cudaFree(m->m));
    CHECK_ERROR(cudaFree(m));
}

/**
 * Alloue une matrice en mémoire GPU et initialise ses valeurs à zéro.
 */
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

/**
 * Affiche une matrice avec une option d'affichage réduit.
 */
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

/**
 * CPU
 * Calcule le produit de Hadamard de deux matrices et stocke le résultat.
 */
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

/**
 * GPU en utilisant la parallélisation CUDA
 * Calcule le produit de Hadamard de deux matrices et stocke le résultat.
 */
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

/**
 * CPU
 * Calcule la somme élément par élément de deux matrices.
 */
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

/**
 * GPU en utilisant la parallélisation CUDA
 * Calcule la somme élément par élément de deux matrices.
 */
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

/**
 * CPU
 * Calcule la différence élément par élément de deux matrices.
 */
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

/**
 * GPU en utilisant la parallélisation CUDA
 * Calcule la différence élément par élément de deux matrices.
 */
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

/**
 * CPU
 * Effectue le produit matriciel classique entre deux matrices.
 */
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

/**
 * GPU en utilisant la parallélisation CUDA
 * Effectue le produit matriciel classique entre deux matrices.
 */
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

/**
 * GPU en utilisant la multiplication en tuiles
 * Effectue le produit matriciel classique entre deux matrices.
 */
__global__
void matrix_dot_GPU_shared(matrix_t *m1, matrix_t *m2, matrix_t *res) {
    assert((m1->columns == m2->rows) &&
           (m1->rows == res->rows) &&
           (m2->columns == res->columns));

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    int tileCount = (m1->columns + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < tileCount; t++) {
        if (row < m1->rows && t * TILE_SIZE + threadIdx.x < m1->columns)
            tileA[threadIdx.y][threadIdx.x] = m1->m[row * m1->columns + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < m2->columns && t * TILE_SIZE + threadIdx.y < m2->rows)
            tileB[threadIdx.y][threadIdx.x] = m2->m[(t * TILE_SIZE + threadIdx.y) * m2->columns + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m1->rows && col < m2->columns) {
        res->m[row * m2->columns + col] = sum;
    }
}

/**
 * Effectue le produit matriciel en utilisant la bibliothèque cuBLAS.
 */
void matrix_dot_cublas(matrix_t* m1, matrix_t* m2, matrix_t* res) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const double alpha = 1.0;
    const double beta = 0.0;

    cublasDgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m1->rows, m2->columns, m1->columns,
                &alpha,
                m1->m, m1->rows,
                m2->m, m2->rows,
                &beta,
                res->m, res->rows);

    cublasDestroy(handle);
}

/**
 * Applique une fonction scalaire à chaque élément d'une matrice.
 */
void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = f(m1->m[idx]);
    }
}

/**
 * CPU
 * Calcule la transposée d'une matrice.
 */
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

/**
 * GPU en utilisant la parallélisation CUDA
 * Calcule la transposée d'une matrice.
 */
__global__
void matrix_transpose_GPU(matrix_t *m1, matrix_t *res) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m1->rows && col < m1->columns)
    {
        res->m[col*m1->rows+row] = m1->m[row*m1->columns+col];
    } 
}

/**
 * CPU
 * Multiplie chaque élément d'une matrice par un scalaire.
 */
void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    for (int idx = 0; idx < m1->columns*m1->rows; idx ++)
    {
        res->m[idx] = m1->m[idx] * s;
    }
}

/**
 * GPU en utilisant la parallélisation CUDA
 * Multiplie chaque élément d'une matrice par un scalaire.
 */
__global__
void matrix_scalar_GPU(matrix_t *m1, double s, matrix_t *res) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*res->columns+col;

    if (row < m1->rows && col < m1->columns) {
        res->m[idx] = m1->m[idx] * s;
    } 

}

/**
 * Copie les éléments d'une matrice source vers une matrice destination.
 */
void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));     
}