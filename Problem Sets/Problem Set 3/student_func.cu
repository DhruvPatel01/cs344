/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <math.h>

#if __CUDA_ARCH__ < 600
  #define atomicAdd_block(X,Y) atomicAdd(X,Y)
#endif

__global__ void min(
   const float* const d_array,
   float *d_out, const size_t max_elems)
{
   extern __shared__ float sdata[];
   int tId = threadIdx.x;
   int gId = blockDim.x *  blockIdx.x + tId;

   if (gId < max_elems)
      sdata[tId] = d_array[gId];
   __syncthreads();

   for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
      if (tId < s && gId < max_elems && (gId+s) < max_elems && sdata[tId] > sdata[tId+s]) 
         sdata[tId] = sdata[tId+s];
      __syncthreads();
   }

   if (tId == 0)
      d_out[blockIdx.x] = sdata[0];
}

__global__ void my_cuMin(
   const float* const d_array,
   float *d_out, const size_t max_elems)
{
   extern __shared__ float sdata[];
   int tId = threadIdx.x;
   int gId = blockDim.x *  blockIdx.x + tId;

   if (gId < max_elems)
      sdata[tId] = d_array[gId];
   __syncthreads();

   for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
      if (tId < s && gId < max_elems && (gId+s) < max_elems && sdata[tId] > sdata[tId+s]) 
         sdata[tId] = sdata[tId+s];
      __syncthreads();
   }

   if (tId == 0)
      d_out[blockIdx.x] = sdata[0];
}

__global__ void my_cuMax(
   const float* const d_array,
   float *d_out, const size_t max_elems)
{
   extern __shared__ float sdata[];
   int tId = threadIdx.x;
   int gId = blockDim.x *  blockIdx.x + tId;

   if (gId < max_elems)
      sdata[tId] = d_array[gId];
   __syncthreads();

   for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
      if (tId < s && gId < max_elems && (gId+s) < max_elems && sdata[tId] < sdata[tId+s]) 
         sdata[tId] = sdata[tId+s];
      __syncthreads();
   }

   if (tId == 0)
      d_out[blockIdx.x] = sdata[0];

}

void min_or_max_driver(const float* const d_array, 
                       float* h_out, const size_t numElems,
                       bool is_max)
{
   float *d_intermediate, *d_prev;
      
   size_t numBlocks = (size_t) ceil(numElems/1024.0);
   size_t prevNumBlocks;

   checkCudaErrors(cudaMalloc((void **)&d_intermediate, sizeof(float)*numBlocks));

   if (is_max) {
      my_cuMax<<<numBlocks, 1024, 1024*sizeof(float)>>>(d_array, d_intermediate, numElems);
   } else {
      my_cuMin<<<numBlocks, 1024, 1024*sizeof(float)>>>(d_array, d_intermediate, numElems);
   }

   while (numBlocks > 1) {
      prevNumBlocks = numBlocks;
      numBlocks = (int) ceil(prevNumBlocks/1024.0);
      d_prev = d_intermediate;
      checkCudaErrors(cudaMalloc((void**) &d_intermediate, sizeof(float)*numBlocks));

      if (is_max) {
         my_cuMax<<<numBlocks, 1024, 1024*sizeof(float)>>>(d_prev, d_intermediate, prevNumBlocks);
      } else {
         my_cuMin<<<numBlocks, 1024, 1024*sizeof(float)>>>(d_prev, d_intermediate, prevNumBlocks);
      }

      checkCudaErrors(cudaFree(d_prev));
   }

   checkCudaErrors(cudaMemcpy(h_out, d_intermediate, sizeof(float), cudaMemcpyDeviceToHost));
   checkCudaErrors(cudaFree(d_intermediate));
}

const int NUM_SEGMENTS = 2;

__global__ void histogram_kernel(const float * const d_logLuminance, 
                                 unsigned int * const d_histogram,
                                 unsigned int numElems,
                                 size_t numBins,
                                 float range, float min_ll)
{
   extern __shared__ unsigned int local_hist[];
   int tIdx = threadIdx.x;
   
   for (int i = tIdx; i < numBins; i += blockDim.x)
      local_hist[i] = 0;
   __syncthreads();

   int gIdx = blockDim.x * blockIdx.x + tIdx;
   if (gIdx < numElems) {
      float x = d_logLuminance[gIdx];
      int bin = (int) ((x - min_ll)/range * numBins);
      if (bin == numBins)
         bin--;
      atomicAdd_block(&local_hist[bin], 1);
   }
   __syncthreads();


   int base = (blockIdx.x % NUM_SEGMENTS)*numBins;

   for (int i = tIdx; i < numBins; i += blockDim.x)
      atomicAdd(&d_histogram[base+i], local_hist[i]);
}       

/*
   Following implementation assumes that numBins <= blockDim.x;
*/
__global__ void ex_scan(unsigned int *d_cdf, 
                        const unsigned int * d_histogram, 
                        int numBins){
   int tId = threadIdx.x;
   int D = (int) ceil(log2((double)numBins)) - 1;
   unsigned int val = d_histogram[tId];
   unsigned int y;
   extern __shared__ unsigned int ex_scan_local[];

   if (tId < numBins)
      ex_scan_local[tId] = d_histogram[tId];

   for (int d = 0; d <= D; d++) {
      int p = pow(2, d);
      if (tId >= p && tId < numBins)
         y = ex_scan_local[(int)tId - p];
      __syncthreads();

      if (tId >= p && tId < numBins) {
         val += y;
         ex_scan_local[tId] = val;
      }
      __syncthreads();
   }

   if (tId < numBins)
      d_cdf[tId] = ex_scan_local[tId];


}

__global__ void shift_right(unsigned int *d_cdf) 
{
   int tId = threadIdx.x;
   unsigned int val;
   if (tId == 0)
      val = 0;
   else
      val = d_cdf[tId-1];
   __syncthreads();

   d_cdf[tId] = val;
}

__global__ void merge_segments(unsigned int * d_histogram, int numBins) 
{
   int tId = threadIdx.x;
   unsigned int local = 0;
   for (int i = 0; i < NUM_SEGMENTS; i++)
      local += d_histogram[i*numBins+tId];
   d_histogram[tId] = local;
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

   float tmp_max, tmp_min;
   min_or_max_driver(d_logLuminance, &tmp_max, numRows*numCols, true);
   min_or_max_driver(d_logLuminance, &tmp_min, numRows*numCols, false);

   float log_range = tmp_max - tmp_min;

   min_logLum = tmp_min;
   max_logLum = tmp_max;

   unsigned int* d_histogram;
   checkCudaErrors(cudaMalloc((void **) &d_histogram, 
                   NUM_SEGMENTS*numBins*sizeof(int)));
   checkCudaErrors(cudaMemset((void *) d_histogram, 0, numBins*sizeof(int)));

   float tpb = 1024.0;
   histogram_kernel<<<ceil(numRows*numCols/tpb), (int)tpb, numBins*sizeof(int)>>>(
      d_logLuminance, d_histogram, numRows*numCols, numBins, log_range, tmp_min
   );

   if (NUM_SEGMENTS > 1) {
      merge_segments<<<1, numBins>>>(d_histogram, numBins);
   }

   checkCudaErrors(cudaMemset((void *) d_cdf, 0, sizeof(int)*numBins));
   int nthreads = pow(2, ceil(log2(numBins)));
   ex_scan<<<1, nthreads, numBins*sizeof(int)>>>(d_cdf, d_histogram, numBins);
   shift_right<<<1, numBins>>>(d_cdf);
   checkCudaErrors(cudaFree(d_histogram));
}
