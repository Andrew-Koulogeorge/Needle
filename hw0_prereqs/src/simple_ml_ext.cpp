#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <cmath>

namespace py = pybind11;
using namespace std; 

float *elementwise_divide(float *A, size_t a, size_t b, size_t batch_size);
void normalize(float *expA, size_t a, size_t b);
float *construct_Iy(const unsigned char *y, size_t batch_size, size_t num_classes);
float *elementwise_exp(const float *A, size_t a, size_t b);

float *mat_mult(const float *A, const float *B, 
            const size_t a, const size_t c, const size_t b);

void inplace_sub(float *A, float *B, 
              const size_t a, const size_t b, float lr);

float *transpose(const float *A, const size_t a, const size_t b);
float *sub(float *A, float *B, const size_t a, const size_t b);

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE

    // intermediate values in computation

    float *preds; // raw predictions
    float *Z; // normalized logits
    float *Iy; // stacked 1 hot label vectors 
    float *grad; // gradient matrix 
    float *X_T; // transpose matrix

    size_t idx = 0; // keeping track of row we are at in the array
    size_t batch_size; // number of examples we consider at a time. batch_size*n floats per batch
    
    while(idx < m)
    {
      batch_size = (m-idx > batch) ? batch : m-idx;

      // compute logits
      preds = mat_mult(X, theta, batch_size, n, k);
      Z = elementwise_exp(preds, batch_size, k);
      normalize(Z, batch_size, k);

      // compute grad
      Iy = construct_Iy(y, batch_size, k);
      X_T = transpose(X, batch_size, n);
      grad = elementwise_divide(mat_mult(X_T, sub(Z, Iy, batch_size, k), n, batch_size, k), n, k, batch_size);

      // sgd step
      inplace_sub(theta, grad, n, k, lr);

      // update pointer to next batch of data (jumping down rows in the matrix)
      X += n*batch_size; 
      y += batch_size;

      idx += batch_size; 
    }
    // delete[] preds;
    // delete[] Z;
    // delete[] Iy;
    // delete[] grad;
    // delete[] X_T;

    /// END YOUR CODE
}

/* in place division of the rows */
float *elementwise_divide(float *A, size_t a, size_t b, size_t batch_size)
{
  for(size_t i = 0; i < a; i++)
  {
    for(size_t j = 0; j < b; j++)
    {
      A[i*b + j] /= batch_size;
    }
  }
  return A;
}


/* in place normalization of the rows */
void normalize(float *expA, size_t a, size_t b){
  for(size_t i = 0; i < a; i++)
  {
    float row_sum = 0.0;
    for(size_t j = 0; j < b; j++)
    {
      row_sum += expA[i*b + j];
    }
    for(size_t j = 0; j < b; j++)
    {
      expA[i*b + j] /= row_sum;
    }
  }
}
float *construct_Iy(const unsigned char *y, size_t batch_size, size_t num_classes)
{
  float *result = new float[batch_size*num_classes]();
  for(size_t i = 0; i < batch_size; i++)
  {
    for(size_t j = 0; j < num_classes; j++)
    {
      if(y[i] == j){
        result[i*num_classes + j] = 1.0;
      }
    }
  }
  return result;

}
float *elementwise_exp(const float *A, size_t a, size_t b){
  float *result = new float[a*b];
  for(size_t i = 0; i < a; i++)
  {
    for(size_t j = 0; j < b; j++)
    {
      result[i*b + j] = exp(A[i*b + j]);
    }
  }
  return result;
}

/*
Matmult takes in pointers to two arrays and multiplies them together,
returning a new array. Assuming that A is shape a x c and B is of type c x b
*/

float *mat_mult(const float *A, const float *B, 
            const size_t a, const size_t c, const size_t b)
{
  // declare a new array of size a x b
  // vector<float> result(a*b, 0);
  float *result = new float[a*b]();

  // loop over the rows of a
  for(size_t i = 0; i < a; i++)
  {
    for(size_t j = 0; j < b; j++)
    {
      float temp = 0.0;
      for(size_t k = 0; k < c; k++)
      {
        temp += A[i*c + k] * B[k*b + j];
      }
      result[i*b + j] = temp;
    }
  }
  return result;
}
/* add to matrixs of the same shape a x b together. if neg!=0, subtract them*/
void inplace_sub(float *A, float *B, 
              const size_t a, const size_t b, float lr)
{
    // loop over both index sets
  for(size_t i = 0; i < a; i++)
  {
    size_t row_index = i*b;
    for(size_t j = 0; j < b; j++)
    {
      A[row_index + j] -= lr*B[row_index + j];
    }
  }
}              
float *sub(float *A, float *B, 
              const size_t a, const size_t b)
{
  // vector<float> result(a*b, 0);
  float *result = new float[a*b];

  // loop over both index sets
  for(size_t i = 0; i < a; i++)
  {
    size_t row_index = i*b;
    for(size_t j = 0; j < b; j++)
    {
      result[row_index + j] = A[row_index + j] - B[row_index + j];
    }
  }
  return result;
}

/*Transpose takes in a pointer to an array and performs a transpose on it*/
float *transpose(const float *A, const size_t a, const size_t b){
  float *result = new float[a*b];
  for(size_t i = 0; i <a; i++)
  {
    for(size_t j = 0; j < b; j++)
    {
      result[i+j*a] = A[i*b+j];
    }
  }
  return result;
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
