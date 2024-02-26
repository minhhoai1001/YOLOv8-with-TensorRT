# Lice detection with Yolov8 and TensorRT on Jetson Xavier
## Environment:

- Jetpack 4.6.3
- CUDA-10.2
- CUDNN-8.2.1
- TensorRT-8.2.1
- DeepStream-6.0.1
- OpenCV-4.1.1
- CMake-3.12

## Export model
### Export YOLOv8 to ONNX
```
yolo task=detect mode=export model=yolov8l.pt simplify=true format=onnx opset=11 imgsz=640
```
### Export ONNX to TensorRT
```
/usr/src/tensorrt/bin/trtexec \
--onnx=yolov8l.onnx \
--saveEngine=yolov8l.engine
```

## Inference with c++
```
mkdir build_xavier
cd build_xavier
cmake ..
make
./lice_det ../input.h265
```