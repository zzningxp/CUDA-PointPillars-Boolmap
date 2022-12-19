# PointPillars with BOOLVFE inference in TensorRT

This repository implemnts the inference of [PointPillars-ROS with BOOLMAP VFE](https://github.com/zzningxp/PointPillars-ROS) using TensorRT. The struct of the model is [PointPillars_MultiHead_40FPS](https://github.com/hova88/PointPillars_MultiHead_40FPS) combined with [boolVFE](https://github.com/Livox-SDK/livox_detection).

The original inference time of pointpillars in [PointPillars-ROS with BOOLMAP VFE](https://github.com/zzningxp/PointPillars-ROS) is around 42 ms(backbone) in Xavier. In this reposity, the inferece time can reach 16ms in Xavier by using fp16, though there is an unsovling problem in this impl, which makes the output indexes of bbox mismatches with the indexes of cls and dir. This might be caused by a bug of TensorRT. If we enable the fp32 in Xavier, all outputs will math correctly, but the inference time will extend to 46 ms. we are going to fix this problem soon.



## Data
The demo use the data from KITTI Dataset and more data can be downloaded following the linker
[GETTING_STARTED](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md)

## Model
The onnx file can be converted from a model trainned by [PointPillars-ROS with BOOLMAP VFE](https://github.com/zzningxp/PointPillars-ROS) with the tool in the demo.

## Build

### Prerequisites
To build the pointpillars inference, **TensorRT** with BoolVFE layer and **CUDA** are needed. BoolVFE layer plugin is already implemented as a plugin for TRT in the demo.

- Jetpack 4.5
- TensorRT v7.1.3
- CUDA-10.2 + cuDNN-8.0.0
- PCL is optinal to store pcd pointcloud file

### Compile
---

```shell
$ cd test
$ mkdir build
$ cd build
$ make -j$(nproc)
```

## Run
```shell
$ ./demo
```
## Enviroments

- Jetpack 4.5
- Cuda10.2 + cuDNN8.0.0 + TensorRT 7.1.3
- Nvidia Jetson AGX Xavier

#### Performance
- FP16
```
|                   | GPU/ms | 
| ----------------- | ------ |
| Inference         | 16.41  |
| Postprocessing    | 9.61   |
```

- FP32
```
|                   | GPU/ms | 
| ----------------- | ------ |
| Inference         | 42.40  |
| Postprocessing    | 6.14   |
```
## Note
1. The demo will cache the onnx file to improve performance.
If a new onnx will be used, please remove the cache file in "./model"

## References

- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [Autoware-AI/core_perception](https://github.com/Autoware-AI/core_perception)

