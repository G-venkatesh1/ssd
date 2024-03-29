# SSD: Single Shot MultiBox Detector

#### Pretrained Weights - https://drive.google.com/drive/folders/1_DYYDJUfwLIvGBDnM3hMFNgkVRZW6MgX?usp=sharing
 Place the Pretrained Weights inside the path Trained_models/ 
 
 ### Accuracy

| Dataset        | Dtype| Runtime |  100 Images | Full Dataset |
| -------------- | ---- | -------- | ------- | ----- |
| COCO  | FP32 | Pytorch |**mAPtest** - 0.805| **mAPtest** - 25.048
| COCO  | FP32 | ONNX Runtime | **mAPtest** - 0.805 |**mAPtest** - 25.048
| COCO  | FP16 | ONNX Runtime | **mAPtest**- 0.808 |**mAPtest**-25.062|
| COCO  | INT8 | ONNX Runtime(Quantize_static) |**mAPtest** - 0.810| **mAPtest** - 23.826|
| COCO  | INT8 | TVM          | **mAPtest** - 0.811| **mAPtest**- 23.827|
-


 Make sure to put the files as the following structure (The root folder names **coco**):
  ```
  coco
  ├── annotations
  │   ├── instances_train2017.json
  │   └── instances_val2017.json
  │── train2017
  └── val2017 
  ```
## Docker

Build:

`./devtools/build.sh`

Run:

`./devtools/run.sh`

### Pytorch Baseline validation :
#### Please use the below command to run validation for pytorch model


```python
python -m torch.distributed.launch --nproc_per_node=1  --master_port=25678 train.py --model ssd --batch-size 1 --data-path datasets_coco  -c config/config.yaml --rt val --subset 5000
```

#### You will get the results:

```python
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.25048
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.42364
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.25828
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.07166
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.27042
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.39630
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.23743
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.34527
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.36238
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.11909
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.39848
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.54719
```

### FP32 ONNX Export:
#### Please use the below command to export FP32 ONNX model:

```python
python test_dataset.py --pretrained-model trained_models/SSD.pth --data-path datasets_coco -c config/config.yaml --type fp32_model
```

### FP32 ONNX Runtime:
#### Please use the below command to run inference for FP32 ONNX model:


```python
python -m torch.distributed.launch --nproc_per_node=1 --master_port=25678 train.py --model ssd --batch-size 1 --data-path datasets_coco --rt Onnxruntime -c config/config.yaml --mtype fp32_model --subset 5000
```

#### You will get the results:
```python
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.25048
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.42364
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.25828
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.07166
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.27042
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.39630
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.23743
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.34527
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.36238
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.11909
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.39848
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.54719
```


### FP16 ONNX Export:
#### Please use the below command to export FP16 ONNX model:


```python
python test_dataset.py --pretrained-model trained_models/SSD.pth --data-path datasets_coco -c config/config.yaml --type fp16_model
```

### FP16 ONNX runtime:
#### Please use the below command to run inference for FP16 ONNX model:

```python
python -m torch.distributed.launch --nproc_per_node=1 --master_port=25678 train.py --model ssd --batch-size 1 --data-path datasets_coco --rt Onnxruntime -c config/config.yaml --mtype fp16_model --subset 5000
```
#### You will get the results:

```python
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.25062
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.42369
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.25866
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.07168
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.27035
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.39675
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.23746
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.34524
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.36226
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.11905
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.39844
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.54740
```

### INT8 ONNX Export:
#### Please use the below command to Convert FP32 ONNX model into Int8 ONNX model:

```python
python test_dataset.py --pretrained-model trained_models/SSD.pth --data-path datasets_coco -c config/config.yaml --type Quant_ONNX_Export --subset 100
```

### INT8 ONNX Runtime:
#### Please use the below command to run inference for Int8 ONNX model:

```python
python -m torch.distributed.launch --nproc_per_node=1 --master_port=25678 train.py --model ssd --batch-size 1 --data-path datasets_coco --rt Onnxruntime -c config/config.yaml --mtype QDQ_model --subset 5000
```
#### You will get the results:

```python
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.23826
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.40493
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.24463
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.06377
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.25429
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.38037
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.23168
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.33475
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.35305
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.11472
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.38767
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.53644 
```

### TVM CPU Convertion :

#### Please use the below command to convert Int8 model into TVM format:
```python
 python test_dataset.py --pretrained-model trained_models/SSD.pth --data-path datasets_coco -c config/config.yaml --type Convert_tvm
```

### TVM CPU Evaluation :
#### Please use the below command to run inference for TVM CPU:

```python
python -m torch.distributed.launch --nproc_per_node=1 --master_port=25678 train.py --model ssd --batch-size 1 --data-path datasets_coco --rt tvm -c config/config.yaml --mtype evaluate_tvm --deploy_cfg config/tvm_cpu.py  --subset 5000
```
#### You will get the results:

 ```python
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.24424                                                                                                    
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.41253                                                                                                    
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.25204
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.06570
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.26109
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.38912
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.23609
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.34253
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.36159
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.11826
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.39927
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.54424
 ```