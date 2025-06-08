import math
import os
import argparse

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import transforms
import videotransforms

import numpy as np

import torch.nn.functional as F
from pytorch_i3d import InceptionI3d

# from nslt_dataset_all import NSLT as Dataset
from datasets.nslt_dataset_all import NSLT as Dataset
import cv2
import json


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()
parser.add_argument("-mode", type=str, help="rgb or flow")
parser.add_argument("-save_model", type=str)
parser.add_argument("-root", type=str)

args = parser.parse_args()


def load_rgb_frames_from_video(video_path, start=0, num=-1):
    vidcap = cv2.VideoCapture(video_path)
    # vidcap = cv2.VideoCapture('/home/dxli/Desktop/dm_256.mp4')

    frames = []

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    if num == -1:
        num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    for offset in range(num):
        success, img = vidcap.read()

        # w, h, c = img.shape
        # sc = 224 / w
        # img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        img = cv2.resize(img, (224, 224))  # fixed 224x224
        img = (img / 255.0) * 2 - 1

        frames.append(img)

    frames = torch.Tensor(np.asarray(frames, dtype=np.float32))
    return frames


def run(
    init_lr=0.1,
    max_steps=64e3,
    mode="rgb",
    root="/ssd/Charades_v1_rgb",
    train_split="charades/charades.json",
    batch_size=3 * 15,
    save_model="",
    weights=None,
):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    val_dataset = Dataset(train_split, "test", root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False
    )

    dataloaders = {"test": val_dataloader}
    datasets = {"test": val_dataset}

    # setup the model
    if mode == "flow":
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load("weights/flow_imagenet.pt"))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load("weights/rgb_imagenet.pt"))
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(
        torch.load(weights)
    )  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    correct = 0
    correct_5 = 0
    correct_10 = 0

    top1_fp = np.zeros(num_classes, dtype=np.int)
    top1_tp = np.zeros(num_classes, dtype=np.int)

    top5_fp = np.zeros(num_classes, dtype=np.int)
    top5_tp = np.zeros(num_classes, dtype=np.int)

    top10_fp = np.zeros(num_classes, dtype=np.int)
    top10_tp = np.zeros(num_classes, dtype=np.int)

    for data in dataloaders["test"]:
        inputs, labels, video_id = data  # inputs: b, c, t, h, w

        per_frame_logits = i3d(inputs)

        predictions = torch.max(per_frame_logits, dim=2)[0]
        out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
        out_probs = np.sort(predictions.cpu().detach().numpy()[0])

        if labels[0].item() in out_labels[-5:]:
            correct_5 += 1
            top5_tp[labels[0].item()] += 1
        else:
            top5_fp[labels[0].item()] += 1
        if labels[0].item() in out_labels[-10:]:
            correct_10 += 1
            top10_tp[labels[0].item()] += 1
        else:
            top10_fp[labels[0].item()] += 1
        if torch.argmax(predictions[0]).item() == labels[0].item():
            correct += 1
            top1_tp[labels[0].item()] += 1
        else:
            top1_fp[labels[0].item()] += 1
        print(
            video_id,
            float(correct) / len(dataloaders["test"]),
            float(correct_5) / len(dataloaders["test"]),
            float(correct_10) / len(dataloaders["test"]),
        )

        # per-class accuracy
    top1_per_class = np.mean(top1_tp / (top1_tp + top1_fp))
    top5_per_class = np.mean(top5_tp / (top5_tp + top5_fp))
    top10_per_class = np.mean(top10_tp / (top10_tp + top10_fp))
    print(
        "top-k average per class acc: {}, {}, {}".format(
            top1_per_class, top5_per_class, top10_per_class
        )
    )


def ensemble(mode, root, train_split, weights, num_classes):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    # test_transforms = transforms.Compose([])

    val_dataset = Dataset(train_split, "test", root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False
    )

    dataloaders = {"test": val_dataloader}
    datasets = {"test": val_dataset}

    # setup the model
    if mode == "flow":
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load("weights/flow_imagenet.pt"))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load("weights/rgb_imagenet.pt"))
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(
        torch.load(weights)
    )  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    correct = 0
    correct_5 = 0
    correct_10 = 0
    # confusion_matrix = np.zeros((num_classes,num_classes), dtype=np.int)

    top1_fp = np.zeros(num_classes, dtype=np.int)
    top1_tp = np.zeros(num_classes, dtype=np.int)

    top5_fp = np.zeros(num_classes, dtype=np.int)
    top5_tp = np.zeros(num_classes, dtype=np.int)

    top10_fp = np.zeros(num_classes, dtype=np.int)
    top10_tp = np.zeros(num_classes, dtype=np.int)

    for data in dataloaders["test"]:
        inputs, labels, video_id = data  # inputs: b, c, t, h, w

        t = inputs.size(2)
        num = 64
        if t > num:
            num_segments = math.floor(t / num)

            segments = []
            for k in range(num_segments):
                segments.append(inputs[:, :, k * num : (k + 1) * num, :, :])

            segments = torch.cat(segments, dim=0)
            per_frame_logits = i3d(segments)

            predictions = torch.mean(per_frame_logits, dim=2)

            if predictions.shape[0] > 1:
                predictions = torch.mean(predictions, dim=0)

        else:
            per_frame_logits = i3d(inputs)
            predictions = torch.mean(per_frame_logits, dim=2)[0]

        out_labels = np.argsort(predictions.cpu().detach().numpy())

        if labels[0].item() in out_labels[-5:]:
            correct_5 += 1
            top5_tp[labels[0].item()] += 1
        else:
            top5_fp[labels[0].item()] += 1
        if labels[0].item() in out_labels[-10:]:
            correct_10 += 1
            top10_tp[labels[0].item()] += 1
        else:
            top10_fp[labels[0].item()] += 1
        if torch.argmax(predictions).item() == labels[0].item():
            correct += 1
            top1_tp[labels[0].item()] += 1
        else:
            top1_fp[labels[0].item()] += 1
        print(
            video_id,
            float(correct) / len(dataloaders["test"]),
            float(correct_5) / len(dataloaders["test"]),
            float(correct_10) / len(dataloaders["test"]),
        )

    top1_per_class = np.mean(top1_tp / (top1_tp + top1_fp))
    top5_per_class = np.mean(top5_tp / (top5_tp + top5_fp))
    top10_per_class = np.mean(top10_tp / (top10_tp + top10_fp))
    print(
        "top-k average per class acc: {}, {}, {}".format(
            top1_per_class, top5_per_class, top10_per_class
        )
    )


def run_on_tensor(weights, ip_tensor, num_classes):
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(num_classes)
    # Use map_location to load weights on CPU
    i3d.load_state_dict(torch.load(weights, map_location=torch.device("cpu")))
    # Remove or conditionally execute cuda() if no GPU is available
    if torch.cuda.is_available():
        i3d.cuda()
        i3d = nn.DataParallel(i3d)
    i3d.eval()

    t = ip_tensor.shape[2]
    ip_tensor = ip_tensor.to(torch.device("cpu"))
    per_frame_logits = i3d(ip_tensor)

    print("Input tensor shape:", ip_tensor.shape)
    print(per_frame_logits.shape)
    predictions = torch.mean(per_frame_logits, dim=2)[0]
    print(predictions.shape)
    out_labels = np.argsort(predictions.cpu().detach().numpy())

    return out_labels


# def run_on_tensor(weights, ip_tensor, num_classes):
#     i3d = InceptionI3d(400, in_channels=3)
#     i3d.replace_logits(num_classes)
#     i3d.load_state_dict(torch.load(weights))
#     i3d.cuda()
#     i3d = nn.DataParallel(i3d)
#     i3d.eval()

#     t = ip_tensor.size(2)
#     num = 64
#     if t > num:
#         num_segments = math.floor(t / num)

#         segments = []
#         for k in range(num_segments):
#             segments.append(ip_tensor[:, :, k*num: (k+1)*num, :, :])

#         segments = torch.cat(segments, dim=0)
#         per_frame_logits = i3d(segments)

#         predictions = torch.mean(per_frame_logits, dim=2)

#         if predictions.shape[0] > 1:
#             predictions = torch.mean(predictions, dim=0)

#     else:
#         per_frame_logits = i3d(ip_tensor)
#         predictions = torch.mean(per_frame_logits, dim=2)[0]

#     print("Input tensor shape:", ip_tensor.shape)
#     print(per_frame_logits.shape)
#     print(predictions.shape)
#     out_labels = np.argsort(predictions.cpu().detach().numpy())


#     return out_labels


def get_slide_windows(frames, window_size, stride=1):
    indices = torch.arange(0, frames.shape[0])
    window_indices = indices.unfold(0, window_size, stride)

    return frames[window_indices, :, :, :].transpose(1, 2)


# if _name_ == '_main_':
#     # ================== test i3d on a dataset ==============
#     # need to add argparse
#     mode = 'rgb'
#     num_classes = 2000
#     save_model = './checkpoints/'

#     root = '../../data/WLASL2000'

#     train_split = 'preprocess/nslt_{}.json'.format(num_classes)
#     weights = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'

#     run(mode=mode, root=root, save_model=save_model, train_split=train_split, weights=weights)


if __name__ == "__main__":
    # ========== Run inference on a single video ==========
    print("-" * 100)
    import sys
    import torch

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("Python executable:", sys.executable)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print(
        "Device name:",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    )
    print("-" * 100)
    print(sys.executable)

    mode = "rgb"
    num_classes = 2000

    video_path = (
        r"C:\Users\Ashwa\hackathon\ash_testvid.mp4"  # << CHANGE THIS to your test video
    )
    weights = r"C:\Users\Ashwa\final_hackathon\WLASL\code\I3D\archived\asl2000\FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt"

    print("Loading video and preprocessing...")
    frames = load_rgb_frames_from_video(video_path)

    # print(f"Loaded {frames.shape[0]} frames.")

    # # Pad or trim to multiple of 64 if needed
    # if frames.shape[0] < 64:
    #     pad = torch.zeros(64 - frames.shape[0], 224, 224, 3)
    #     frames = torch.cat((frames, pad), dim=0)
    # else:
    #     frames = frames[:64]  # Trim if longer than 64

    # # Shape: T x H x W x C --> 1 x 3 x T x H x W

    frames = frames.permute(0, 3, 1, 2)  # T x C x H x W
    frames = frames.unsqueeze(0)  # 1 x T x C x H x W
    frames = frames.permute(0, 2, 1, 3, 4)  # 1 x C x T x H x W

    print("Running model...")
    predicted_labels = run_on_tensor(weights, frames, num_classes)

    with open(r"C:\Users\Ashwa\hackathon\WLASL\start_kit\WLASL_v0.3.json", "r") as f:
        data = json.load(f)

    index_to_gloss = {i: entry["gloss"] for i, entry in enumerate(data)}

    print("Predicted glosses:")
    for idx in predicted_labels[-5:]:
        gloss = index_to_gloss.get(idx, "Unknown")
        print(f"Class {idx}: {gloss}")
