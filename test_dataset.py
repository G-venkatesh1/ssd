"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os
import numpy as np
import argparse
import logging
from src.dataset import CocoDataset
from src.transform import SSDTransformer
from src.model_optimise import SSD, ResNet 
import cv2
import shutil
import yaml
import onnx
from onnxsim import simplify
import torch
import torch.onnx
import onnxruntime as ort
from onnxconverter_common import float16
from onnxruntime.quantization.calibrate import (CalibrationDataReader,
                                                CalibrationMethod)
from onnxruntime.quantization.quantize import quantize_static, quantize_dynamic
from onnxruntime.quantization.preprocess import quant_pre_process
from onnxruntime.quantization.quant_utils import QuantType
from onnxsim import simplify
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from pathlib import Path
import sys
from multiprocessing import Process
import timm
from PIL import Image

from src.utils_optimise import generate_dboxes, Encoder, colors
from src.model_optimise import SSD, ResNet
from onnxruntime.quantization import QuantFormat, quantize_static
from onnxruntime.quantization.quant_utils import QuantType

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)
global input_shape

COMMON_TRANSFORMS = [
    relay.transform.InferType(),
    relay.transform.SimplifyInference(),
    relay.transform.FakeQuantizationToInteger(
        hard_fail=False, optional_qnn_ops=["nn.softmax"]
    ),
]


def read_config(config_path):
    with open(config_path, "r") as file:
        code = file.read()
    cfg = {}
    exec(code, cfg)
    return cfg

def read_yaml(config_path):
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: The file '{config_path}' does not exist.")
    except yaml.YAMLError as exc:
        print(f"Error while reading '{config_path}': {exc}")

def get_args():
    parser = argparse.ArgumentParser("Implementation of SSD")
    parser.add_argument("--data-path", type=str, default="/coco", help="the root folder of dataset")
    parser.add_argument("--cls-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--pretrained-model", type=str, default="trained_models/SSD.pth")
    parser.add_argument("--output", type=str, default="predictions")
    parser.add_argument("-c","--config_path", type=str, required=True, help="Path to yaml configuration")
    parser.add_argument("--type", choices=["fp16_model","fp32_model","Quant_ONNX_Export","Convert_tvm","fp16_MP"])
    parser.add_argument("--subset", default=10, type=int, help="Number of images to use from each folder")
    parser.add_argument("--deploy_cfg", default="config/tvm_cpu.py", type=str, help="deploy config path")
    
    
    args = parser.parse_args()
    return args


def get_tvm_targets(targets):
    tvm_targets = []
    transforms = []
    for target in targets:
        if "llvm" in target:
            tvm_targets.append(tvm.target.Target(target))
        else:
            raise ValueError("Unknown tvm target:", target)
    return tvm_targets, transforms


def convert_tvm(model, deploy_cfg):
    logger.info("Start onnx2tvm")
    print("start tvm convertion")
    model = os.path.abspath(model)
    print(os.path.basename(model).replace(".onnx", ".tar"))
    tvm_config = deploy_cfg["tvm_config"]
    tvm_config["out"] = os.path.basename(model).replace(".onnx", ".tar")
    original_workdir = os.getcwd()
    print(f"original workdir = {original_workdir}")
    onnx2tvm_workdir = os.path.join(os.getcwd(), "tvm")
    if Path(onnx2tvm_workdir).exists():
        shutil.rmtree(onnx2tvm_workdir)
    Path(onnx2tvm_workdir).mkdir(parents=True, exist_ok=True)
    os.chdir(onnx2tvm_workdir)
    stderr = os.dup(sys.stderr.fileno())
    log_stderr = open("onnx2tvm_stderr.txt", "wb")
    os.dup2(log_stderr.fileno(), sys.stderr.fileno())
    print("running conversion")
    p = Process(target=_convert, args=(model, deploy_cfg, onnx2tvm_workdir))
    p.start()
    p.join(timeout=tvm_config["timeout"])
    log_stderr.close()
    os.dup2(stderr, sys.stderr.fileno())
    with open("onnx2tvm_stderr.txt") as f:
        print(f.read())
    sys.stderr.flush()
    os.chdir(original_workdir)
    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(
            f"TVM model convert: timeout after {tvm_config['timeout']} sec"
        )
    if p.exitcode is None or p.exitcode > 0:
        raise RuntimeError("TVM compile failed:", p.exitcode)
    tvm_config = deploy_cfg["tvm_config"]
    deploy_cfg[model[0]] = os.path.join(onnx2tvm_workdir, tvm_config["out"])

    if not os.path.exists(deploy_cfg[model[0]]):
        raise RuntimeError(f"TVM model didn't generated to {deploy_cfg.model[0]}")

    logger.info(
        "Successfully exported TVM model for %s: %s",
        tvm_config["compiler"],
        model,
    )
    print("finished tvm convertion")


def _convert(model, deploy_cfg, onnx2tvm_workdir: str):
    assert Path(model).exists(), model

    tvm_config = deploy_cfg["tvm_config"]

    onnx_model = onnx.load(model)
    onnx_input = onnx_model.graph.input[0]
    input_name = onnx_input.name
    input_shape = [d.dim_value for d in onnx_input.type.tensor_type.shape.dim]

    shape_dict = {input_name: input_shape}
    mod, params = relay.frontend.from_onnx(
        onnx_model, shape_dict, convert_config={"no_ort_dequantize": True}
    )
    with open("tvm_onnx_model_relay.txt", "w") as f:
        print(mod, file=f)

    targets, transforms = get_tvm_targets(
        tvm_config["targets"]
    )
    with tvm.transform.PassContext(opt_level=tvm_config["opt_level"]):
        relay.backend.te_compiler.get().clear()
        mod = tvm.transform.Sequential(COMMON_TRANSFORMS)(mod)
        for fn in transforms:
            mod = fn(mod)
        with open(f"tvm_{tvm_config['compiler']}_model_relay.txt", "w") as f:
            print(mod, file=f)
    lib = relay.build(mod, target=targets, params=params)
    lib.export_library(os.path.join(onnx2tvm_workdir, tvm_config["out"]))
    print(os.path.join(onnx2tvm_workdir, tvm_config["out"]))
    return lib

def test(opt):
    model = SSD(backbone=ResNet())
    dummy_input = torch.randn(1, 3, 300, 300).to("cuda")
    checkpoint = torch.load(opt.pretrained_model)
    model.load_state_dict(checkpoint["model_state_dict"])
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    dboxes = generate_dboxes()
    test_set = CocoDataset(opt.data_path, 2017, "val", SSDTransformer(dboxes, (300, 300), val=True))
    encoder = Encoder(dboxes)    
    if os.path.isdir(opt.output):
        shutil.rmtree(opt.output)
    os.makedirs(opt.output)

    for img, img_id, img_size, _, _ in test_set:
        print(img_size)
        if img is None:
            continue
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():            
            if opt.type == "fp32_model":
                print("\n exporting fp32 onnx ")
                torch.onnx.export(
                    model, dummy_input, config["Model_onnx"], opset_version=11)
                simplified_onnx_model, check = simplify(config["Model_onnx"])
                onnx.save(simplified_onnx_model, config["Model_onnx"])    
                print("fp32 onnx exported")
            if opt.type == "fp16_model":
                print("\n exporting fp16 onnx: ")
                model = onnx.load(config["Model_onnx"])
                model_fp16 = float16.convert_float_to_float16(model)
                onnx.save(model_fp16, config["model_fp16"])
                print("fp16 onnx exported")
            if opt.type == "Quant_ONNX_Export":
                Quant_ONNX_Export(config, opt.subset)
            if opt.type =="Convert_tvm":
                convert_tvm(config["Model_quant"], deploy_cfg) 
            exit()   

def Quant_ONNX_Export(config, subset=100):
    print("quant ONNX Exporting")
    model_fp32 = config["Model_onnx"]
    model_int8 = config["Model_quant"]
    preprocessed_name = model_fp32 + ".pre_static.onnx"
    image_directory = config["image_dir"]
    image_files = os.listdir(image_directory)
    image_files = [
        f
        for f in image_files
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))
    ]
   
    dboxes = generate_dboxes()
    test_set = CocoDataset(opt.data_path, 2017, "val", SSDTransformer(dboxes, (300, 300), val=True))
    
    class SSD_data_Reader:
        def __init__(self, fp32_onnx_path, calibration_loader, sample=None) -> None:
            self.fp32_onnx_path = fp32_onnx_path
            self.calibration_loader = calibration_loader
            self.enum_data = None
            self.sample = sample
               
        def get_next(self, EP_list = ['CPUExecutionProvider']):
            if self.enum_data is None:
                session = ort.InferenceSession(self.fp32_onnx_path, providers=EP_list)
                input_name = session.get_inputs()[0].name
                calib_list = []
                count = 0
                for nhwc_data in self.calibration_loader:
                    images = nhwc_data[0]
                    images_with_batch = images.unsqueeze(0)
                    calib_list.append({input_name: images_with_batch.numpy()}) 
                    if self.sample is not None and self.sample == count:
                        break
                    count += 1
                self.enum_data = iter(calib_list)
            return next(self.enum_data, None)    

    dr = SSD_data_Reader(model_fp32, test_set, sample=100)
    quant_pre_process(model_fp32, preprocessed_name)        
    quantize_static(
            preprocessed_name,
            model_int8,
            calibration_data_reader=dr,
            calibrate_method=CalibrationMethod.MinMax,
            quant_format=QuantFormat.QDQ,
            weight_type=QuantType.QInt16,
            activation_type=QuantType.QInt8,        
            per_channel=False, reduce_range=False,
            extra_options={
                 "CalibMovingAverageConstant": 0.1, "CalibMovingAverage": True, 
            },
        )
    print("Int8 model exported")
    exit()
    
if __name__ == "__main__":
    opt = get_args()
    config = read_yaml(opt.config_path)
    if opt.deploy_cfg:
        deploy_cfg = read_config(opt.deploy_cfg)
    test(opt)
    