"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
from tqdm.autonotebook import tqdm
import torch
from pycocotools.cocoeval import COCOeval
# from apex import amp
import onnxruntime as ort
import yaml
import argparse
import tvm
from tvm import relay
from tvm.contrib import graph_executor


def read_config(config_path):
    with open(config_path, "r") as file:
        code = file.read()
    cfg = {}
    exec(code, cfg)
    return cfg        

def train(model, train_loader, epoch, writer, criterion, optimizer, scheduler, is_amp):
    model.train()
    num_iter_per_epoch = len(train_loader)
    progress_bar = tqdm(train_loader)
    scheduler.step()
    for i, (img, _, _, gloc, glabel) in enumerate(progress_bar):
        if torch.cuda.is_available():
            img = img.cuda()
            gloc = gloc.cuda()
            glabel = glabel.cuda()

        ploc, plabel = model(img)
        ploc, plabel = ploc.float(), plabel.float()
        gloc = gloc.transpose(1, 2).contiguous()
        loss = criterion(ploc, plabel, gloc, glabel)

        progress_bar.set_description("Epoch: {}. Loss: {:.5f}".format(epoch + 1, loss.item()))

        writer.add_scalar("Train/Loss", loss.item(), epoch * num_iter_per_epoch + i)

        if is_amp:
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate(model, test_loader, epoch, writer, encoder, nms_threshold, mtype,rt,config,deploy_cfg,subset):
    model.eval()
    if rt == "Onnxruntime":
        if mtype == "fp32_model":
            print("fp32 rt")
            providers = ["CPUExecutionProvider"]
            model  =  ort.InferenceSession(config["Model_onnx"], providers=providers)
        elif mtype == "fp16_model":   
            print("fp16 rt")
            providers = ["CPUExecutionProvider"]
            model  =  ort.InferenceSession(config["model_fp16"], providers=providers) 
        elif mtype == "fp16_MP":   
            print("fp16_MP rt")
            providers = ["CPUExecutionProvider"]
            model  =  ort.InferenceSession(config["model_fp16_mp"], providers=providers) 
        elif mtype == "QDQ_model":   
            print("QDQ rt")
            providers = ["CPUExecutionProvider"]
            model  =  ort.InferenceSession(config["Model_quant"], providers=providers)  
    elif rt == "tvm":
        if mtype == "evaluate_tvm":   
            print("TVM EVALUATION")
            if deploy_cfg:
                deploy_cfg = read_config(deploy_cfg)
            host = deploy_cfg["tvm_config"].get("host")
            port = deploy_cfg["tvm_config"].get("port")
            tvm_config = deploy_cfg["tvm_config"]
            if not (host and port):
                dev = tvm.device(str(tvm_config["targets"][0]), 0)
                print(dev)
                lib: tvm.runtime.Module = tvm.runtime.load_module("tvm/SSDRes50_quant.tar")
                module = graph_executor.GraphModule(lib["default"](dev))
            else:
                remote = tvm.rpc.connect(host, port)
                dev = remote.cpu()
                remote.upload("tvm/yolor_quant.tar", target="model.tar")
                mod = remote.load_module("model.tar")
                module = graph_executor.GraphModule(mod["default"](dev))      
    detections = []
    category_ids = test_loader.dataset.coco.getCatIds()
    for nbatch, (img, img_id, img_size, _) in enumerate(test_loader):   
        print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
        if nbatch==subset:
            break
        if torch.cuda.is_available():
            if img is None: #change
                continue
            img = img.cuda()
        with torch.no_grad():
            if rt == "Onnxruntime":
                print("ONNXRT")
                if mtype == "fp32_model":
                    img = img.cpu()
                    ort_inputs = model.get_inputs()[0].name
                    ort_outs = model.run([], {ort_inputs: img.numpy()})
                elif mtype == "fp16_model":
                    img_np_float16 = img.cpu().numpy().astype(np.float16)
                    ort_inputs = {model.get_inputs()[0].name: img_np_float16}
                    ort_outs = model.run(None, ort_inputs) 
                elif mtype == "fp16_MP":
                    img_np_float16 = img.cpu().numpy()
                    ort_inputs = {model.get_inputs()[0].name: img_np_float16}
                    ort_outs = model.run(None, ort_inputs) 
                elif mtype == "QDQ_model":
                    img = img.cpu()
                    ort_inputs = model.get_inputs()[0].name
                    ort_outs = model.run([], {ort_inputs: img.numpy()})    
                ploc = torch.tensor(ort_outs[0]).cuda()
                plabel = torch.tensor(ort_outs[1]).cuda()
            elif rt == "tvm":
                if mtype == "evaluate_tvm":
                    img = img.cpu()
                    module.set_input("input.1", img)
                    module.run()
                    output_shape1 = (1,4,8732)
                    output_shape2 = (1,81,8732)
                    output1 = tvm.nd.empty(output_shape1)
                    output2 = tvm.nd.empty(output_shape2)
                    module.get_output(0, out=output1)
                    module.get_output(1, out=output2)
                    output1_np = output1.numpy()
                    output2_np = output2.numpy()
                    ploc = torch.tensor(output1_np).cuda()
                    plabel = torch.tensor(output2_np).cuda()
            elif rt == "val":
                ploc, loc,label,prob = model(img)
                loc = torch.stack(loc)
                label = torch.stack(label)
                prob=torch.stack(prob)
                print(ploc)
                # print(ploc,len(label[0]),len(prob[0]),len(loc[0]))
                # ploc, plabel = ploc.float(), plabel.float()
            # print(ploc,plabel)
            for idx in range(len(ploc)):
                ploc_i = loc[idx, :, :].unsqueeze(0)
                plabel_i = label[idx, :, :].unsqueeze(0)
                pprob_i = prob[idx, :, :].unsqueeze(0)
    #             # try:
    #             result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
    #             # except:
    #                 # print("No object detected in idx: {}".format(idx))
    #                 # continue
    #             if img_size is None: #change
    #                 print("no image")
    #                 continue     
                height, width = img_size[idx]                
    #             loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(ploc_i, plabel_i, pprob_i):
                    detections.append([img_id[idx], loc_[0] * width, loc_[1] * height, (loc_[2] - loc_[0]) * width,
                                    (loc_[3] - loc_[1]) * height, prob_,
                                    category_ids[label_ - 1]])

    # detections = np.array(detections, dtype=np.float32)

    # coco_eval = COCOeval(test_loader.dataset.coco, test_loader.dataset.coco.loadRes(detections), iouType="bbox")
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()

    # writer.add_scalar("Test/mAP", coco_eval.stats[0], epoch)