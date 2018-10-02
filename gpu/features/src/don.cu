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
            PtrSz<NormalType> small_normals;
            PtrSz<NormalType> large_normals;
            PtrSz<PointType>  output;

            PtrStep<int> indices;
            const int *sizes;

            __device__ __forceinline__ void operator()() const
            {                
                //__shared__ float cov_buffer[6][CTA_SIZE + 1];

                int warp_idx = Warp::id();
                int idx = blockIdx.x * WAPRS + warp_idx;
                
                if (idx >= output.size)
                    return;               


                ////perform DoN subtraction and return results
                //for (size_t point_id = 0; point_id < input_->points.size (); ++point_id)
                //{
                //  output.points[point_id].getNormalVector3fMap () =  
                //    (input_normals_small_->points[point_id].getNormalVector3fMap () - input_normals_large_->points[point_id].getNormalVector3fMap ()) / 2.0;

                //  if(!pcl_isfinite (output.points[point_id].normal_x) || !pcl_isfinite (output.points[point_id].normal_y) || !pcl_isfinite (output.points[point_id].normal_z)){
                //    output.points[point_id].getNormalVector3fMap () = Eigen::Vector3f(0,0,0);
                //  }

                //  output.points[point_id].curvature = output.points[point_id].getNormalVector3fMap ().norm();
                //}

            }                  

            __device__ __forceinline__ float3 fetch(int idx) const 
            {
                return *(float3*)&points[idx];                
            }

        };

        __global__ void EstimateDifferenceOfNormalsKernel(const DifferenceOfNormalsEstimator est) { est(); }

    }
}

void pcl::device::computeDifference(const PointCloud& cloud, const NeighborIndices& indices, const Normals& small_norm, const Normals& large_norm, PointCloud& output)
{

    DifferenceOfNormalsEstimator est;

    est.points = cloud;    
    est.small_normals = small_norm;
    est.large_normals = large_norm;
    est.output = output;

    est.indices = indices;    
    est.sizes = indices.sizes;

    int block = DifferenceOfNormalsEstimator::CTA_SIZE;
    int grid = divUp((int)cloud.size(), DifferenceOfNormalsEstimator::WAPRS);

    EstimateDifferenceOfNormalsKernel<<<grid, block>>>(est);

    cudaSafeCall(cudaGetLastError());        
    cudaSafeCall(cudaDeviceSynchronize());
}
