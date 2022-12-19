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
#ifndef PARAMS_H_
#define PARAMS_H_
#define MAX_VOXELS 30000
class Params
{
  public:
    static const int num_classes = 10;
    const char *class_name [num_classes] = { "car","truck","construction_vehicle","bus","trailer","barrier","motorcycle","bicycle","pedestrian","traffic_cone",};
    const float min_x_range = -51.2;
    const float max_x_range = 51.2;
    const float min_y_range = -51.2;
    const float max_y_range = 51.2;
    const float min_z_range = -5.0;
    const float max_z_range = 3.0;
    // the size of a pillar
    const float pillar_x_size = 0.2;
    const float pillar_y_size = 0.2;
    const float pillar_z_size = 0.2;
    const int max_num_points_per_pillar = 20;
    const int num_point_values = 4;
    // the number of feature maps for pillar scatter
    const int num_feature_scatter = 40;
    const float dir_offset = 0.78539;
    const float dir_limit_offset = 0.0;
    // the num of direction classes(bins)
    const int num_dir_bins = 2;
    // anchors decode by (x, y, z, dir)
    static const int num_anchors = num_classes * 2;
    static const int len_per_anchor = 4;
    const float anchors[num_anchors * len_per_anchor] = {
      4.63,1.97,1.74,0.0,
      4.63,1.97,1.74,1.57,
      6.93,2.51,2.84,0.0,
      6.93,2.51,2.84,1.57,
      6.37,2.85,3.19,0.0,
      6.37,2.85,3.19,1.57,
      10.5,2.94,3.47,0.0,
      10.5,2.94,3.47,1.57,
      12.29,2.9,3.87,0.0,
      12.29,2.9,3.87,1.57,
      0.5,2.53,0.98,0.0,
      0.5,2.53,0.98,1.57,
      2.11,0.77,1.47,0.0,
      2.11,0.77,1.47,1.57,
      1.7,0.6,1.28,0.0,
      1.7,0.6,1.28,1.57,
      0.73,0.67,1.77,0.0,
      0.73,0.67,1.77,1.57,
      0.41,0.41,1.07,0.0,
      0.41,0.41,1.07,1.57,
      };
    const float anchor_bottom_heights[num_classes] = {-0.95,-0.6,-0.225,-0.085,0.115,-1.33,-1.085,-1.18,-0.935,-1.285,};
    // the score threshold for classification
    const float score_thresh = 0.1;
    const float nms_thresh = 0.2;
    const int max_num_pillars = MAX_VOXELS;
    const int pillarPoints_bev = max_num_points_per_pillar * max_num_pillars;
    // the detected boxes result decode by (x, y, z, w, l, h, yaw)
    const int num_box_values = 7;
    // the input size of the 2D backbone network
    const int grid_x_size = (max_x_range - min_x_range) / pillar_x_size;
    const int grid_y_size = (max_y_range - min_y_range) / pillar_y_size;
    const int grid_z_size = (max_z_range - min_z_range) / pillar_z_size;
    // the output size of the 2D backbone network
    const int feature_x_size = grid_x_size / 4;
    const int feature_y_size = grid_y_size / 4;
    Params() {};
};
#endif
