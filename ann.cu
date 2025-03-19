#include "ann.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include "error.h"


double normalRand(double mu, double sigma);
void init_weight(matrix_t* w, unsigned nneurones_prev);
void print_layer(layer_t *layer);

double normalRand(double mu, double sigma)
{
	const double epsilon = DBL_MIN;
	const double two_pi = 2.0*M_PI;
    bool generate;
    double z1;

	generate = !generate;

	if (!generate)
	   return z1 * sigma + mu;

	double u1, u2;
	do
	 {
	   u1 = (double) rand() / RAND_MAX;
	   u2 = (double) rand() / RAND_MAX;
	 }
	while ( u1 <= epsilon );

	double z0;
	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}

void init_weight(matrix_t* w, unsigned nneurones_prev)
{
    for (int idx = 0; idx < w->columns * w->rows; idx ++)
    {
        w->m[idx] = normalRand(0, 1 / sqrt(nneurones_prev));
    }
}

ann_t * create_ann(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned* nneurons_per_layer)
{   
    //ann_t * nn = (ann_t *)malloc(sizeof(ann_t));
    ann_t * nn;
    //cudaMalloc((void**)&nn, sizeof(ann_t));
    
    CHECK_ERROR(cudaMallocManaged((void **)&nn, sizeof(ann_t)));

    //nn->layers = (layer_t **)malloc(number_of_layers * sizeof(layer_t *));
    CHECK_ERROR(cudaMallocManaged((void**)&nn->layers, number_of_layers * sizeof(layer_t *)));

    nn->number_of_layers = number_of_layers;
    nn->alpha = alpha;
    nn->minibatch_size = minibatch_size;

    nn->layers[0] = create_layer(0, nneurons_per_layer[0], minibatch_size, minibatch_size);
    for (int l = 1; l < number_of_layers; l++)
    {
        nn->layers[l] = create_layer(l, nneurons_per_layer[l], nneurons_per_layer[l-1], minibatch_size);
    }

    return nn;
}

layer_t * create_layer(unsigned layer_number, unsigned number_of_neurons, unsigned nneurons_previous_layer, unsigned minibatch_size)
{
    //layer_t * layer = (layer_t*) malloc(sizeof(layer_t));
    layer_t * layer;
    CHECK_ERROR(cudaMallocManaged((void**)&layer, sizeof(layer_t)));

    layer->number_of_neurons = number_of_neurons;
    layer->minibatch_size = minibatch_size;    
    layer->activations = alloc_matrix(number_of_neurons, minibatch_size);
    layer->z = alloc_matrix(number_of_neurons, minibatch_size);
    layer->delta = alloc_matrix(number_of_neurons, minibatch_size);
    layer->weights = alloc_matrix(number_of_neurons, nneurons_previous_layer);    
    layer->biases = alloc_matrix(number_of_neurons, 1);

    if (layer_number > 0)
    {
        init_weight(layer->weights, nneurons_previous_layer);
    }

    return layer;
}

void set_input(ann_t *nn, matrix_t* input){
    matrix_memcpy(nn->layers[0]->activations, input);
}

void print_layer(layer_t *layer)
{
    printf("-- neurons:%d, minibatch size:%d\n", layer->number_of_neurons, layer->minibatch_size);

    printf(">> Weighted inputs --\n");
    print_matrix(layer->z, true);
    printf(">> Activations --\n");
    print_matrix(layer->activations, true);
    
    printf(">> Weights --\n");
    print_matrix(layer->weights, true);
    printf(">> Biases --\n");
    print_matrix(layer->biases, true);

    printf(">> Delta --\n");
    print_matrix(layer->delta, true);
    
}

void print_nn(ann_t *nn)
{
    printf("ANN -- nlayers:%d, alpha:%lf, minibatch size: %d\n", nn->number_of_layers, nn->alpha, nn->minibatch_size);
    for (int l = 0; l < nn->number_of_layers; l++)
    {
        printf("Layer %d ", l);
        print_layer(nn->layers[l]);
    }
}

void forward(ann_t *nn, double (*activation_function)(double))
{
    for (int l = 1; l < nn->number_of_layers; l++)
    {
        matrix_t *z1 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        matrix_t *z2 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        matrix_t *one = alloc_matrix(1, nn->minibatch_size);
        for (int idx = 0; idx < one->columns*one->rows; idx++)
            one->m[idx] = 1.0;

        //matrix_dot(nn->layers[l]->weights, nn->layers[l-1]->activations, z1); // z1 <- w^l x a^(l-1)
        dim3 blockDim(16, 16);
        dim3 gridDim(ceil(((float)nn->layers[l-1]->activations->columns) / blockDim.x), ceil(((float)nn->layers[l]->weights->rows) / blockDim.y));
        matrix_dot_GPU<<<gridDim, blockDim>>>(nn->layers[l]->weights, nn->layers[l-1]->activations, z1);
        // matrix_dot_cublas(nn->layers[l]->weights, nn->layers[l-1]->activations, z1);
        CHECK_ERROR(cudaDeviceSynchronize());

        //matrix_dot(nn->layers[l]->biases, one, z2); // z2 <- b^l x 1        
        dim3 blockDim2(16, 16);
        dim3 gridDim2( ceil( ((float)one->columns) / blockDim2.x ), ceil( ((float)nn->layers[l]->biases->rows) / blockDim2.y ) );
        matrix_dot_GPU<<<gridDim2, blockDim2>>>(nn->layers[l]->biases, one, z2);
        // matrix_dot_cublas(nn->layers[l]->biases, one, z2);
        CHECK_ERROR(cudaDeviceSynchronize());
        
        
        //matrix_sum(z1, z2, nn->layers[l]->z); // z^l <- z1 + z2 <=> z^l <- w^l x a^(l-1) + b^l x 1   
        dim3 blockDim_sum(16, 16);
        dim3 gridDim_sum( ceil( ((float)z1->columns) / blockDim_sum.x ), ceil( ((float)z1->rows) / blockDim_sum.y ) );
        matrix_sum_GPU<<<gridDim_sum, blockDim_sum>>>(z1, z2, nn->layers[l]->z);
        cudaDeviceSynchronize();

        matrix_function(nn->layers[l]->z, activation_function, nn->layers[l]->activations); // a^l = f(z^l)
     
        destroy_matrix(z1);
        destroy_matrix(z2);
        destroy_matrix(one);

    }
}

/* void forward(ann_t *nn, double (*activation_function)(double)) {
    cudaStream_t stream1, stream2, stream3; 

    // Criando streams
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    for (int l = 1; l < nn->number_of_layers; l++) {
        matrix_t *z1 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        matrix_t *z2 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        matrix_t *one = alloc_matrix(1, nn->minibatch_size);

        for (int idx = 0; idx < one->columns * one->rows; idx++)
            one->m[idx] = 1.0;

        // Pré-carregamento para GPU
        int device;
        cudaGetDevice(&device);
        cudaMemPrefetchAsync(z1->m, z1->rows * z1->columns * sizeof(double), device, stream1);
        cudaMemPrefetchAsync(z2->m, z2->rows * z2->columns * sizeof(double), device, stream2);
        cudaMemPrefetchAsync(one->m, one->rows * one->columns * sizeof(double), device, stream2);
        cudaMemPrefetchAsync(nn->layers[l]->weights->m, nn->layers[l]->weights->rows * nn->layers[l]->weights->columns * sizeof(double), device, stream1);
        cudaMemPrefetchAsync(nn->layers[l-1]->activations->m, nn->layers[l-1]->activations->rows * nn->layers[l-1]->activations->columns * sizeof(double), device, stream1);
        cudaMemPrefetchAsync(nn->layers[l]->biases->m, nn->layers[l]->biases->rows * nn->layers[l]->biases->columns * sizeof(double), device, stream2);

         // z1 <- w^l x a^(l-1)
        dim3 blockDim(16, 16);
        dim3 gridDim(ceil(((float)nn->layers[l-1]->activations->columns) / blockDim.x), 
                     ceil(((float)nn->layers[l]->weights->rows) / blockDim.y));
        matrix_dot_GPU<<<gridDim, blockDim, 0, stream1>>>(nn->layers[l]->weights, nn->layers[l-1]->activations, z1);
        // cudaStreamSynchronize(stream1);

         // z1 <- w^l x a^(l-1)
        dim3 blockDim2(16, 16);
        dim3 gridDim2(ceil(((float)one->columns) / blockDim2.x), 
                      ceil(((float)nn->layers[l]->biases->rows) / blockDim2.y));
        matrix_dot_GPU<<<gridDim2, blockDim2, 0, stream2>>>(nn->layers[l]->biases, one, z2);

        // Sincroniza ambos antes de somar
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);

        // Pré-carregamento para soma
        cudaMemPrefetchAsync(z1->m, z1->rows * z1->columns * sizeof(double), device, stream3);
        cudaMemPrefetchAsync(z2->m, z2->rows * z2->columns * sizeof(double), device, stream3);
        cudaMemPrefetchAsync(nn->layers[l]->z->m, nn->layers[l]->z->rows * nn->layers[l]->z->columns * sizeof(double), device, stream3);

        // Executa a soma em stream3
        dim3 blockDim_sum(16, 16);
        dim3 gridDim_sum(ceil(((float)z1->columns) / blockDim_sum.x), 
                         ceil(((float)z1->rows) / blockDim_sum.y));
        matrix_sum_GPU<<<gridDim_sum, blockDim_sum, 0, stream3>>>(z1, z2, nn->layers[l]->z);

        // Sincroniza antes de aplicar a função de ativação
        cudaStreamSynchronize(stream3);

        matrix_function(nn->layers[l]->z, activation_function, nn->layers[l]->activations);

        destroy_matrix(z1);
        destroy_matrix(z2);
        destroy_matrix(one);
    }

    // Destroi os streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
} */


void backward(ann_t *nn, matrix_t *y, double (*derivative_actfunct)(double))
{
    unsigned L = nn->number_of_layers-1;

    matrix_t *dfzL = alloc_matrix(nn->layers[L]->number_of_neurons, nn->minibatch_size);

    //matrix_minus(nn->layers[L]->activations, y, nn->layers[L]->delta);  // delta^(L) = (a^L - y)
    dim3 blockDim_minus(16, 16);
    dim3 gridDim_minus(ceil(((float)nn->layers[L]->activations->columns) / blockDim_minus.x), ceil(((float)nn->layers[L]->activations->rows) / blockDim_minus.y));
    matrix_minus_GPU<<<gridDim_minus, blockDim_minus>>>(nn->layers[L]->activations, y, nn->layers[L]->delta);
    cudaDeviceSynchronize();

    matrix_function(nn->layers[L]->z, derivative_actfunct, dfzL); // f'(z^(L))

    //hadamard_product(nn->layers[L]->delta, dfzL, nn->layers[L]->delta); // delta^(L) = (a^L - y) o f'(z^(L))
    dim3 blockDim_had(16, 16);
    dim3 gridDim_had(ceil(((float)nn->layers[L]->delta->columns) / blockDim_had.x), ceil(((float)nn->layers[L]->delta->rows) / blockDim_had.y));
    hadamard_product_GPU<<< gridDim_had, blockDim_had >>> (nn->layers[L]->delta, dfzL, nn->layers[L]->delta); 
    cudaDeviceSynchronize();

    destroy_matrix(dfzL);

    for (int l = L; l > 1; l--)
    {
        matrix_t *tw, *delta_tmp, *dfz;
        tw = alloc_matrix(nn->layers[l-1]->number_of_neurons, nn->layers[l]->number_of_neurons);
        delta_tmp = alloc_matrix(nn->layers[l-1]->number_of_neurons, nn->minibatch_size);
        dfz = alloc_matrix(nn->layers[l-1]->number_of_neurons, nn->minibatch_size);

        //matrix_transpose(nn->layers[l]->weights, tw); // (w^l)T 
        dim3 blockDim_trans(16, 16);
        dim3 gridDim_trans(ceil(((float)nn->layers[l]->weights->columns) / blockDim_trans.x), ceil(((float)nn->layers[l]->weights->rows) / blockDim_trans.y));
        matrix_transpose_GPU<<<blockDim_trans, gridDim_trans>>>(nn->layers[l]->weights, tw);
        cudaDeviceSynchronize();

        //matrix_dot(tw, nn->layers[l]->delta, delta_tmp); // (w^l)T x delta^l
        dim3 blockDim_dot(16, 16);
        dim3 gridDim_dot(ceil(((float)nn->layers[l]->delta->columns) / blockDim_dot.x), ceil(((float)tw->rows) / blockDim_dot.y));
        matrix_dot_GPU<<<gridDim_dot, blockDim_dot>>>(tw, nn->layers[l]->delta, delta_tmp);
        // matrix_dot_cublas(tw, nn->layers[l]->delta, delta_tmp);
        cudaDeviceSynchronize();

        matrix_function(nn->layers[l-1]->z, derivative_actfunct, dfz); // f'(z^(l-1))

        //hadamard_product(delta_tmp, dfz, nn->layers[l-1]->delta); // delta^(l-1) = (w^l)T x delta^l o f'(z^(l-1))
        dim3 blockDim_had2(16, 16);
        dim3 gridDim_had2(ceil(((float)delta_tmp->columns) / blockDim_had2.x), ceil(((float)delta_tmp->rows) / blockDim_had2.y));
        hadamard_product_GPU<<< gridDim_had2, blockDim_had2 >>> (delta_tmp, dfz, nn->layers[l-1]->delta);
        cudaDeviceSynchronize();

        destroy_matrix(tw);
        destroy_matrix(delta_tmp);
        destroy_matrix(dfz);
    }

    for (int l = 1; l < nn->number_of_layers; l++)
    {
        matrix_t *w1, *ta;
        w1 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->layers[l-1]->number_of_neurons);
        ta = alloc_matrix(nn->minibatch_size, nn->layers[l-1]->number_of_neurons);
        
        //matrix_transpose(nn->layers[l-1]->activations, ta); // ta <- (a^(l-1))^T
        dim3 blockDim_trans2(16, 16);
        dim3 gridDim_trans2(ceil(((float)nn->layers[l-1]->activations->columns) / blockDim_trans2.x), ceil(((float)nn->layers[l-1]->activations->rows) / blockDim_trans2.y));
        matrix_transpose_GPU<<<blockDim_trans2, gridDim_trans2>>>(nn->layers[l-1]->activations, ta);
        cudaDeviceSynchronize();

        //matrix_dot(nn->layers[l]->delta, ta, w1); // w1 <- delta^l x (a^(l-1))^T
        dim3 blockDim_dot2(16, 16);
        dim3 gridDim_dot2(ceil(((float)ta->columns) / blockDim_dot2.x), ceil(((float)nn->layers[l]->delta->rows) / blockDim_dot2.y));
        matrix_dot_GPU<<<gridDim_dot2, blockDim_dot2>>>(nn->layers[l]->delta, ta, w1);
        // matrix_dot_cublas(nn->layers[l]->delta, ta, w1);
        cudaDeviceSynchronize();

        //matrix_scalar(w1, nn->alpha / nn->minibatch_size, w1); // w1 <- alpha /m . delta^l x (a^(l-1))^T
        dim3 blockDim_scalar(16, 16);
        dim3 gridDim_scalar(ceil(((float)w1->columns) / blockDim_scalar.x), ceil(((float)w1->rows) / blockDim_scalar.y));
        matrix_scalar_GPU<<<gridDim_scalar, blockDim_scalar>>>(w1, nn->alpha / nn->minibatch_size, w1);
        cudaDeviceSynchronize();

        //matrix_minus(nn->layers[l]->weights, w1, nn->layers[l]->weights); // w^l <- w^l - alpha /m . delta^l x (a^(l-1))^T
        dim3 blockDim_minus2(16, 16);
        dim3 gridDim_minus2((int)ceil((float)w1->columns / blockDim_minus2.x), (int)ceil((float)w1->rows / blockDim_minus2.y));
        matrix_minus_GPU<<<gridDim_minus2, blockDim_minus2>>>(nn->layers[l]->weights, w1, nn->layers[l]->weights);
        cudaDeviceSynchronize();

        destroy_matrix(w1);
        destroy_matrix(ta);

        matrix_t *one, *b1;
        b1 = alloc_matrix(nn->layers[l]->number_of_neurons, 1);
        one = alloc_matrix(nn->minibatch_size, 1);
        for (int idx = 0; idx < one->columns*one->rows; idx++)
            one->m[idx] = 1.0;

        //matrix_dot(nn->layers[l]->delta, one, b1); // b1 <- delta^l x 1^T
        dim3 blockDim_dot3(16, 16);
        dim3 gridDim_dot3(ceil(((float)one->columns) / blockDim_dot3.x), ceil(((float)nn->layers[l]->delta->rows) / blockDim_dot3.y));
        matrix_dot_GPU<<<gridDim_dot3, blockDim_dot3>>>(nn->layers[l]->delta, one, b1);
        // matrix_dot_cublas(nn->layers[l]->delta, one, b1);
        cudaDeviceSynchronize();

        //matrix_scalar(b1,  nn->alpha / nn->minibatch_size, b1); // b1 <- alpha / m . delta^l x 1^T
        dim3 blockDim_scalar2(16, 16);
        dim3 gridDim_scalar2(ceil(((float)b1->columns) / blockDim_scalar2.x), ceil(((float)b1->rows) / blockDim_scalar2.y));
        matrix_scalar_GPU<<<gridDim_scalar2, blockDim_scalar2>>>(b1, nn->alpha / nn->minibatch_size, b1);
        cudaDeviceSynchronize();

        //matrix_minus(nn->layers[l]->biases, b1, nn->layers[l]->biases); // b^l = b^l - alpha / m . delta^l x 1^T
        dim3 blockDim_minus3(16, 16);
        dim3 gridDim_minus3((int)ceil((float)b1->columns / blockDim_minus3.x), (int)ceil((float)b1->rows / blockDim_minus3.y));
        matrix_minus_GPU<<<gridDim_minus3, blockDim_minus3>>>(nn->layers[l]->biases, b1, nn->layers[l]->biases);
        cudaDeviceSynchronize();
        
        destroy_matrix(one);
        destroy_matrix(b1);
    }
}

/* void backward(ann_t *nn, matrix_t *y, double (*derivative_actfunct)(double))
{
    unsigned L = nn->number_of_layers - 1;
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    matrix_t *dfzL = alloc_matrix(nn->layers[L]->number_of_neurons, nn->minibatch_size);

    // Subtrai ativação de saída pelo rótulo esperado
    dim3 blockDim_minus(16, 16);
    dim3 gridDim_minus(ceil((float)nn->layers[L]->activations->columns / blockDim_minus.x),
                       ceil((float)nn->layers[L]->activations->rows / blockDim_minus.y));
    matrix_minus_GPU<<<gridDim_minus, blockDim_minus, 0, stream1>>>(nn->layers[L]->activations, y, nn->layers[L]->delta);

    printf("minus\n");
    fflush(stdout);

    // Aplica função de ativação derivada
    matrix_function(nn->layers[L]->z, derivative_actfunct, dfzL);

    printf("matrix function\n");
    fflush(stdout);

    // Produto de Hadamard
    dim3 blockDim_had(16, 16);
    dim3 gridDim_had(ceil((float)nn->layers[L]->delta->columns / blockDim_had.x),
                     ceil((float)nn->layers[L]->delta->rows / blockDim_had.y));
    hadamard_product_GPU<<<gridDim_had, blockDim_had, 0, stream2>>>(nn->layers[L]->delta, dfzL, nn->layers[L]->delta);

    printf("hadamar\n");
    fflush(stdout);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    destroy_matrix(dfzL);

    for (int l = L; l > 1; l--)
    {
        matrix_t *tw = alloc_matrix(nn->layers[l - 1]->number_of_neurons, nn->layers[l]->number_of_neurons);
        matrix_t *delta_tmp = alloc_matrix(nn->layers[l - 1]->number_of_neurons, nn->minibatch_size);
        matrix_t *dfz = alloc_matrix(nn->layers[l - 1]->number_of_neurons, nn->minibatch_size);

        dim3 blockDim_trans(16, 16);
        dim3 gridDim_trans(ceil((float)nn->layers[l]->weights->columns / blockDim_trans.x),
                           ceil((float)nn->layers[l]->weights->rows / blockDim_trans.y));
        matrix_transpose_GPU<<<gridDim_trans, blockDim_trans, 0, stream1>>>(nn->layers[l]->weights, tw);

        dim3 blockDim_dot(16, 16);
        dim3 gridDim_dot(ceil((float)nn->layers[l]->delta->columns / blockDim_dot.x),
                         ceil((float)tw->rows / blockDim_dot.y));
        matrix_dot_GPU<<<gridDim_dot, blockDim_dot, 0, stream2>>>(tw, nn->layers[l]->delta, delta_tmp);

        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);

        matrix_function(nn->layers[l - 1]->z, derivative_actfunct, dfz);

        hadamard_product_GPU<<<gridDim_dot, blockDim_dot, 0, stream1>>>(delta_tmp, dfz, nn->layers[l - 1]->delta);

        cudaStreamSynchronize(stream1);
        destroy_matrix(tw);
        destroy_matrix(delta_tmp);
        destroy_matrix(dfz);
    }

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
} */
