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
#include <cassert>
#include <iostream>
#include "BoolVFE.h"
#include "BoolVFE_kernels.h"
/**
 * For the usage of those member function, please refer to the
 * offical api doc.
 * https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_ext.html
 */

#ifndef CUTEDEBUG
#define CUTEDEBUG 0 // set debug mode, if you want to see the api call, set it to 1
#endif

#if CUTEDEBUG
#define cutelog(...) {\
    char str[100];\
    sprintf(str, __VA_ARGS__);\
    std::cout << " (๑¯◡¯๑) noexcept CUSTOM PLUGIN TRACE----> call " << "[" << __FILE__ << "][" \
              << __FUNCTION__ << "][Line " << __LINE__ << "] " << str << std::endl;\
    }
#else
#define cutelog(...)
#endif

using namespace nvinfer1;
using nvinfer1::plugin::BoolVfePlugin;
using nvinfer1::plugin::BoolVfePluginCreator;

static const char* PLUGIN_VERSION{"1"};
static const char* PLUGIN_NAME{"BoolVFE"};

// Static class fields initialization
PluginFieldCollection BoolVfePluginCreator::mFC{};
std::vector<PluginField> BoolVfePluginCreator::mPluginAttributes;

BoolVfePlugin::BoolVfePlugin()
{
    cutelog("wow I run to here now");
}

BoolVfePlugin::BoolVfePlugin(const void* data, size_t length)
{
    cutelog("wow I run to here now");
}
// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* BoolVfePlugin::clone() const noexcept
{
    cutelog("wow I run to here now");
    auto* plugin = new BoolVfePlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs BoolVfePlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    cutelog("wow I run to here now");
    assert(outputIndex == 0);
    // for(int i=0;i<nbInputs;i++) {
    //     printf("input[%d]: ", i);
    //     for(int j=0;j<inputs[i].nbDims;j++) {
    //         printf("%d ", inputs[i].d[j]->getConstantValue());
    //     }
    //     printf("\n");
    // }
    nvinfer1::DimsExprs output;
    output.nbDims = 4;
    output.d[0] = exprBuilder.constant(1);
    // output.d[1] = exprBuilder.constant(featureNum_);
    // output.d[2] = exprBuilder.constant(feature_y_size_);
    // output.d[3] = exprBuilder.constant(feature_x_size_);
    // output.d[1] = exprBuilder.constant(60);
    // output.d[2] = exprBuilder.constant(320);
    // output.d[3] = exprBuilder.constant(320);
    output.d[1] = exprBuilder.constant(40);
    output.d[2] = exprBuilder.constant(512);
    output.d[3] = exprBuilder.constant(512);



    return output;
}

bool BoolVfePlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{

    cutelog("wow I run to here now");
    assert(nbInputs == 2);
    assert(nbOutputs == 1);


    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 2)
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    // if (pos == 3)
    // {
    //     return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    // }
    // if (pos == 4)
    // {
    //     return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    // }
    return false;
}

void BoolVfePlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
  cutelog("wow I run to here now");
}

size_t BoolVfePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
  cutelog("wow I run to here now");
  unsigned int cacheBEVSize = inputs[0].dims.d[0]
                                * inputs[0].dims.d[2] * sizeof(float);
  return cacheBEVSize;
}

int BoolVfePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
  cutelog("wow I run to here now");

  unsigned int batch = 1;
//   unsigned int featureNum = 60;
//   unsigned int featureY = 320;
//   unsigned int featureX = 320;
  unsigned int featureNum = 40;
  unsigned int featureY = 512;
  unsigned int featureX = 512;

  const float *in = (const float *)inputs[0];
//   const float *coords_data = (const float *)(inputs[1]);
//   const unsigned int *params_data = (const unsigned int *)(inputs[2]);
  const int *point_size = (const int *)inputs[1];



  float *spatial_feature_data = (float *)(outputs[0]);

//   unsigned int count = inputDesc[0].dims.d[0];
//   cacheBEV_ = workspace;
//   const float *pillar_features_data = (const float *)(cacheBEV_);

  //cudaMemcpyAsync(paramsPtr, params_data, 5*sizeof(int), cudaMemcpyDefault, stream);

  checkCudaErrors(cudaMemsetAsync(spatial_feature_data, 0, batch*featureNum_*feature_y_size_*feature_y_size_ * sizeof(float), stream));
//   checkCudaErrors(reduceMax_kernel_launcher((const float*)in, (float*)pillar_features_data, count, stream));
  checkCudaErrors(boolVFE_kernel_launcher((const float*)in, *point_size, spatial_feature_data, stream));

  return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType BoolVfePlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    cutelog("wow I run to here now");
    return inputTypes[0];
}

// IPluginV2 Methods

const char* BoolVfePlugin::getPluginType() const noexcept
{
    cutelog("wow I run to here now");
    return PLUGIN_NAME;
}

const char* BoolVfePlugin::getPluginVersion() const noexcept
{
    cutelog("wow I run to here now");
    return PLUGIN_VERSION;
}

int BoolVfePlugin::getNbOutputs() const noexcept
{
    cutelog("wow I run to here now");
    return 1;
}

int BoolVfePlugin::initialize() noexcept
{
    cutelog("wow I run to here now");
    return 0;
}

void BoolVfePlugin::terminate() noexcept
{
    cutelog("wow I run to here now");
}

size_t BoolVfePlugin::getSerializationSize() const noexcept
{
    cutelog("wow I run to here now");
    return 0;
}

void BoolVfePlugin::serialize(void* buffer) const noexcept
{
    cutelog("wow I run to here now");
}

void BoolVfePlugin::destroy() noexcept
{
    cutelog("wow I run to here now");
    delete this;
}

void BoolVfePlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    cutelog("wow I run to here now");
    mNamespace = libNamespace;
}

const char* BoolVfePlugin::getPluginNamespace() const noexcept
{
    cutelog("wow I run to here now");
    return mNamespace.c_str();
}

///////////////

BoolVfePluginCreator::BoolVfePluginCreator()
{
    cutelog("wow I run to here now");
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* BoolVfePluginCreator::getPluginName() const noexcept
{
    cutelog("wow I run to here now");
    return PLUGIN_NAME;
}

const char* BoolVfePluginCreator::getPluginVersion() const noexcept
{
    cutelog("wow I run to here now");
    return PLUGIN_VERSION;
}

const PluginFieldCollection* BoolVfePluginCreator::getFieldNames() noexcept
{
    cutelog("wow I run to here now");
    return &mFC;
}

IPluginV2* BoolVfePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    cutelog("wow I run to here now");
    return new BoolVfePlugin();
}

IPluginV2* BoolVfePluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    cutelog("wow I run to here now");
    return new BoolVfePlugin(serialData, serialLength);
}

void BoolVfePluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    cutelog("wow I run to here now");
    mNamespace = libNamespace;
}

const char* BoolVfePluginCreator::getPluginNamespace() const noexcept
{
    cutelog("wow I run to here now");
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(BoolVfePluginCreator);
