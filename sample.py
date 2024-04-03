from src.model_baseline import SSD, ResNet 
from src.utils import generate_dboxes, Encoder, coco_classes
import torch

model = SSD(backbone=ResNet(), num_classes=len(coco_classes))
if torch.cuda.is_available():
        print("cuda available")
        model.cuda()
# model.load_state_dict(torch.load('E:/Mcw/SSDResnet50/SSD.pth'))
# model.eval()
# dummy_input = torch.randn(1, 3, 300, 300).cuda()
# torch.onnx.export(model, dummy_input,"ssd_res50_un_optimised.onnx",constant_folding=False,opset_version=11)
