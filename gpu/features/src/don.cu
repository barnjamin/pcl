/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#include "internal.hpp"

#include "pcl/gpu/utils/safe_call.hpp"
#include "pcl/gpu/utils/device/warp.hpp"
#include "pcl/gpu/utils/device/funcattrib.hpp"

#include "pcl/gpu/features/device/eigen.hpp"
#include <stdio.h>

using namespace pcl::gpu;

namespace pcl
{
    namespace device
    {                 
        struct DifferenceOfNormalsEstimator
        {            
            enum
            {
                CTA_SIZE = 256,
                WAPRS = CTA_SIZE / Warp::WARP_SIZE,
            };

            struct plus 
            {              
                __forceinline__ __device__ float operator()(const float &lhs, const volatile float& rhs) const { return lhs + rhs; }
            }; 

            const PointType *points;
            const PtrSz<NormalType> small_normals;
            const PtrSz<NormalType> large_normals;
            PointType *output;

            PtrStep<int> indices;
            const int *sizes;

            __device__ __forceinline__ void operator()() const
            {                
                __shared__ float cov_buffer[6][CTA_SIZE + 1];

                int warp_idx = Warp::id();
                int idx = blockIdx.x * WAPRS + warp_idx;
                
                if (idx >= points.size)
                    return;               


                //perform DoN subtraction and return results
                for (size_t point_id = 0; point_id < input_->points.size (); ++point_id)
                {
                  output.points[point_id].getNormalVector3fMap () =  
                    (input_normals_small_->points[point_id].getNormalVector3fMap () - input_normals_large_->points[point_id].getNormalVector3fMap ()) / 2.0;

                  if(!pcl_isfinite (output.points[point_id].normal_x) || !pcl_isfinite (output.points[point_id].normal_y) || !pcl_isfinite (output.points[point_id].normal_z)){
                    output.points[point_id].getNormalVector3fMap () = Eigen::Vector3f(0,0,0);
                  }

                  output.points[point_id].curvature = output.points[point_id].getNormalVector3fMap ().norm();
                }



                int size = sizes[idx];
                int lane = Warp::laneId();

                if (size < MIN_NEIGHBOORS)
                {
                    const float NaN = numeric_limits<float>::quiet_NaN();
                    if (lane == 0)
                        normals.data[idx] = make_float4(NaN, NaN, NaN, NaN);
                }

                const int *ibeg = indices.ptr(idx);
                const int *iend = ibeg + size;

                //copmpute centroid
                float3 c = make_float3(0.f, 0.f, 0.f);
                for(const int *t = ibeg + lane; t < iend; t += Warp::STRIDE)                
                    c += fetch(*t);

                volatile float *buffer = &cov_buffer[0][threadIdx.x - lane];

                c.x = Warp::reduce(buffer, c.x, plus());
                c.y = Warp::reduce(buffer, c.y, plus());
                c.z = Warp::reduce(buffer, c.z, plus());                                
                c *= 1.f/size;                                                  

                //nvcc bug workaround. if comment this => c.z == 0 at line: float3 d = fetch(*t) - c;
                __threadfence_block();

                //compute covariance matrix        
                int tid = threadIdx.x;

                for(int i = 0; i < 6; ++i)
                    cov_buffer[i][tid] = 0.f;                

                for(const int *t = ibeg + lane; t < iend; t += Warp::STRIDE)   
                {
                    //float3 d = fetch(*t) - c;

                    float3 p = fetch(*t);
                    float3 d = p - c;

                    cov_buffer[0][tid] += d.x * d.x; //cov (0, 0) 
                    cov_buffer[1][tid] += d.x * d.y; //cov (0, 1) 
                    cov_buffer[2][tid] += d.x * d.z; //cov (0, 2) 
                    cov_buffer[3][tid] += d.y * d.y; //cov (1, 1) 
                    cov_buffer[4][tid] += d.y * d.z; //cov (1, 2) 
                    cov_buffer[5][tid] += d.z * d.z; //cov (2, 2)                     
                }

                Warp::reduce(&cov_buffer[0][tid - lane], plus());
                Warp::reduce(&cov_buffer[1][tid - lane], plus());
                Warp::reduce(&cov_buffer[2][tid - lane], plus());
                Warp::reduce(&cov_buffer[3][tid - lane], plus());
                Warp::reduce(&cov_buffer[4][tid - lane], plus());
                Warp::reduce(&cov_buffer[5][tid - lane], plus());

                volatile float *cov = &cov_buffer[0][tid-lane];
                if (lane < 6)
                    cov[lane] = cov_buffer[lane][tid-lane];

                //solvePlaneParameters
                if (lane == 0)
                {
                    // Extract the eigenvalues and eigenvectors
                    typedef Eigen33::Mat33 Mat33;
                    Eigen33 eigen33(&cov[lane]);

                    Mat33&     tmp = (Mat33&)cov_buffer[1][tid - lane];
                    Mat33& vec_tmp = (Mat33&)cov_buffer[2][tid - lane];
                    Mat33& evecs   = (Mat33&)cov_buffer[3][tid - lane];
                    float3 evals;

                    eigen33.compute(tmp, vec_tmp, evecs, evals);
                    //evecs[0] - eigenvector with the lowerst eigenvalue

                    // Compute the curvature surface change
                    float eig_sum = evals.x + evals.y + evals.z;
                    float curvature = (eig_sum == 0) ? 0 : fabsf( evals.x / eig_sum );

                    NormalType output;
                    output.w = curvature;

                    // The normalization is not necessary, since the eigenvectors from Eigen33 are already normalized
                    output.x = evecs[0].x;
                    output.y = evecs[0].y;
                    output.z = evecs[0].z;                    

                    normals.data[idx] = output;
                }
            }                  

            __device__ __forceinline__ float3 fetch(int idx) const 
            {
                return *(float3*)&points[idx];                
            }

        };

        __global__ void EstimateDifferenceOfNormalsKernel(const DifferenceOfNormalsEstimator est) { est(); }

    }
}

void pcl::device::computeDifference(const PointCloud& cloud, const Indices& indices, const Normals& small_norm, const Normals& large_norm, PointCloud& output)
{

    DifferenceOfNormalsEstimator est;

    est.points = cloud;    
    est.small_norm = small_norm;
    est.large_norm = large_norm;
    est.out = output;

    est.indices = nn_indices;    
    est.sizes = nn_indices.sizes;

    int block = DifferenceOfNormalsEstimator::CTA_SIZE;
    int grid = divUp((int)cloud.size(), DifferenceOfNormalsEstimator::WAPRS);

    EstimateDifferenceOfNormalsKernel<<<grid, block>>>(est);

    cudaSafeCall(cudaGetLastError());        
    cudaSafeCall(cudaDeviceSynchronize());
}
