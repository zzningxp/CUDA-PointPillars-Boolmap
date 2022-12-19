/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <iostream>

#include <cuda_runtime_api.h>

#include "BoolVFE_kernels.h"

__global__ void make_pillar_histo_kernel(
    const float* dev_points, 
    float* pillar_count_histo, const int num_points,
    const int grid_x_size, const int grid_y_size, const int grid_z_size, 
    const float min_x_range, const float min_y_range, const float min_z_range, 
    const float pillar_x_size, const float pillar_y_size, const float pillar_z_size,
    const int input_point_feature) {
  int th_i = blockIdx.x * blockDim.x +  threadIdx.x ;
  if (th_i >= num_points) {
    return;
  }
  int x_coor = floor((dev_points[th_i * input_point_feature + 0] - min_x_range) / pillar_x_size);
  int y_coor = floor((dev_points[th_i * input_point_feature + 1] - min_y_range) / pillar_y_size);
  int z_coor = floor((dev_points[th_i * input_point_feature + 2] - min_z_range) / pillar_z_size);

  if (x_coor >= 0 && x_coor < grid_x_size && y_coor >= 0 &&
      y_coor < grid_y_size && z_coor >= 0 && z_coor < grid_z_size) {
    pillar_count_histo[z_coor * grid_x_size * grid_y_size + y_coor * grid_x_size + x_coor] = 1;
  }
}


cudaError_t boolVFE_kernel_launcher(const float *pillar_features_data,
        size_t points_size,
        float *spatial_feature_data,
        cudaStream_t stream)
{

  int  num_threads_ = 64;
  int num_block = (points_size/ num_threads_);

    const int grid_x_size_=512; 
    const int grid_y_size_=512;
    const int grid_z_size_=40;
    
    const float pillar_x_size_=0.2; 
    const float pillar_y_size_=0.2;
    const float pillar_z_size_=0.2;
    const float min_x_range_=-51.2; 
    const float min_y_range_=-51.2;
    const float min_z_range_= -5.;
    const int input_point_feature_ = 4;

  make_pillar_histo_kernel<<<num_block , num_threads_>>>(
      pillar_features_data, spatial_feature_data, points_size, 
      grid_x_size_, grid_y_size_, grid_z_size_, 
      min_x_range_, min_y_range_, min_z_range_, 
      pillar_x_size_, pillar_y_size_, pillar_z_size_, 
      input_point_feature_);

  return cudaGetLastError();
}
