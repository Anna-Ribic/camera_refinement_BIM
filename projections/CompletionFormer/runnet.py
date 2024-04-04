import argparse
from config import args as args_config
import time
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = args_config.address
os.environ["MASTER_PORT"] = args_config.port
import gc

import torch
from torch import nn

torch.autograd.set_detect_anomaly(True)

from model.completionformer import CompletionFormer

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def complete(args):
    root = "../../ConSLAM/data/"
    image_dir = os.path.join(root, "rgb", "rgb_2058x1533")
    concat_dir = os.path.join(root, "rgb", "rgb-concat-np-cf")
    velodyne_dir = os.path.join(root, "lidar", "lidar-pcd")
    result_dir = os.path.join(root, "lidar", "depth-np")
    #result_dir = os.path.join(root, "lidar", "lidar-depth-dense-2", "lidar-depth-dense-2")
    final_dir = os.path.join(root, "lidar", "lidar-cf-np")
    final_dir_test = os.path.join(root, "lidar", "lidar-test")


    cur_id = int(args.cur[75:-4]) #1650963758997008  # 1650963708569466 #1650963715635127 #1650963708569466 #1650963758997008 #1650963683657809

    if not os.path.isfile(os.path.join(final_dir, "depth_map_dense_%06d.png" % cur_id)):

        print(os.path.join(result_dir, "depth_map_%16d.png" % cur_id))

        net = CompletionFormer(args)
        net.cuda()

        path_to_weights = "pretrained/KITTIDC_L1.pt"  # "pretrained/NYUv2.pt" #"pretrained/KITTIDC_L1.pt"
        checkpoint = torch.load(path_to_weights)
        key_m, key_u = net.load_state_dict(checkpoint['net'], strict=False)
        if key_u:
            print('Unexpected keys :')
            print(key_u)

        if key_m:
            print('Missing keys :')
            print(key_m)
            raise KeyError
        print('Checkpoint loaded from {}!'.format(path_to_weights))

        net = nn.DataParallel(net)
        net.eval()

        depth = np.load(os.path.join(result_dir, "%16d.npy" % cur_id)).astype(np.float32) #cv2.imread(os.path.join(result_dir, "depth_map_%16d.png" % cur_id), cv2.IMREAD_GRAYSCALE)
        #depth = image_depth.astype(np.float32) / 256.0

        rgb = Image.open(os.path.join(image_dir, "%16d.png" % cur_id))
        rgb = np.array(rgb).astype(np.float32) / 256.0
        depth = np.array(depth)
        print('mean_before: ',depth.mean())


        factor = 0.2
        rgb = cv2.resize(rgb, (0, 0), fx=factor, fy=factor)
        depth = cv2.resize(depth, (0, 0), fx=factor, fy=factor)

        rgb = TF.to_tensor(rgb)
        rgb_n = TF.normalize(rgb, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=False)
        depth = TF.to_tensor(depth)
        print('mean_before2: ',depth.mean())

        batch_data = {'rgb': rgb_n.unsqueeze(0).cuda(), 'dep': depth.unsqueeze(0).cuda()}

        output = net(batch_data)

        print(output['pred'].shape)
        print('mean_after: ',output['pred'].mean())

        np.save(os.path.join(final_dir, "%06d.npy" % cur_id), output['pred'].squeeze().detach().cpu().numpy())

        pred = output['pred'].squeeze().detach().cpu() * 256.0
        plt.figure(figsize=(20, 40))
        plt.imsave(os.path.join(final_dir_test, "depth_map_dense_%06d.png" % cur_id), pred)

        heatmap = cv2.imread(os.path.join(final_dir_test, "depth_map_dense_%06d.png" % cur_id))

        concat = cv2.hconcat([np.array((rgb * 256.0).movedim(0, -1)).astype(np.uint8), heatmap])
        cv2.imwrite(os.path.join(concat_dir, "depth_map_%06d.png" % cur_id), concat)

        del rgb
        del rgb_n
        del depth
        del pred
        #del image_depth
        del heatmap
        del concat
        del output
        torch.cuda.empty_cache()
        gc.collect()

"""root = "../../ConSLAM/data/"
image_dir = os.path.join(root, "rgb", "rgb_2058x1533")
concat_dir = os.path.join(root, "rgb", "rgb_concat_cf")
velodyne_dir = os.path.join(root, "lidar", "lidar-pcd")
#result_dir = os.path.join(root, "lidar", "lidar-depth")
result_dir = os.path.join(root, "lidar", "lidar-depth-dense-2", "lidar-depth-dense-2")
final_dir = os.path.join(root, "lidar", "lidar-depth-dense-cf")


cur_id = 1650963758997008 #1650963708569466 #1650963715635127 #1650963708569466 #1650963758997008 #1650963683657809
print(os.path.join(result_dir, "depth_map_%16d.png" % cur_id))

for images in os.listdir(result_dir):

    if (images.endswith(".png")):
        print(images)
        # Data id
        cur_id = int(images[10:-4])

        net = CompletionFormer(args)
        net.cuda()

        path_to_weights = "pretrained/KITTIDC_L1.pt"  # "pretrained/NYUv2.pt" #"pretrained/KITTIDC_L1.pt"
        checkpoint = torch.load(path_to_weights)
        key_m, key_u = net.load_state_dict(checkpoint['net'], strict=False)
        if key_u:
            print('Unexpected keys :')
            print(key_u)

        if key_m:
            print('Missing keys :')
            print(key_m)
            raise KeyError
        print('Checkpoint loaded from {}!'.format(path_to_weights))

        net = nn.DataParallel(net)
        net.eval()

        image_depth = cv2.imread(os.path.join(result_dir, "depth_map_%16d.png" % cur_id), cv2.IMREAD_GRAYSCALE)
        depth = image_depth.astype(np.float32) / 256.0

        rgb = Image.open(os.path.join(image_dir, "%16d.png" % cur_id))
        rgb = np.array(rgb).astype(np.float32) / 256.0
        depth = np.array(depth)

        factor = 0.2
        rgb = cv2.resize(rgb, (0,0), fx = factor, fy = factor)
        depth = cv2.resize(depth, (0, 0), fx = factor, fy = factor)

        rgb = TF.to_tensor(rgb)
        rgb_n = TF.normalize(rgb, (0.485, 0.456, 0.406),(0.229, 0.224, 0.225), inplace=False)
        depth = TF.to_tensor(depth)

        batch_data = {'rgb': rgb_n.unsqueeze(0).cuda(), 'dep': depth.unsqueeze(0).cuda()}

        output= net(batch_data)

        pred = output['pred'].squeeze().detach().cpu() * 256.0
        plt.figure(figsize=(20,40))
        plt.imsave(os.path.join(final_dir, "depth_map_dense_%06d.png" % cur_id), pred)

        heatmap = cv2.imread(os.path.join(final_dir, "depth_map_dense_%06d.png" % cur_id))

        concat = cv2.hconcat([np.array((rgb*256.0).movedim(0, -1)).astype(np.uint8), heatmap])
        cv2.imwrite(os.path.join(concat_dir, "depth_map_%06d.png" % cur_id), concat)

        del rgb
        del rgb_n
        del depth
        del pred
        del image_depth
        del heatmap
        del concat
        del output
        torch.cuda.empty_cache()
        gc.collect()"""

"""pred = output['pred'].detach()

pred = torch.clamp(pred, min=0)

pred = pred[0, 0, :, :].data.cpu().numpy()

pred = (pred*256.0).astype(np.uint16)
pred = Image.fromarray(pred)
pred.save("depth_map_dense_2_completionformer_%06d.png"% cur_id)"""

"""
depth = cv2.imread(os.path.join(result_dir, "depth_map_%16d.png" % cur_id), cv2.IMREAD_GRAYSCALE) / 256.0
print(depth.shape)
#cv2.imshow('depth',depth)
#cv2.waitKey(0)
oh, ow = depth.shape

rgb = cv2.imread(os.path.join(image_dir, "%16d.png" % cur_id)) / 256.0
#cv2.imshow('rgb',rgb)
#cv2.waitKey(0)

print(rgb)
print(np.max(depth))

factor = 0.2
rgb = cv2.resize(rgb, (0,0), fx = factor, fy = factor)
depth = cv2.resize(depth, (0, 0), fx = factor, fy = factor)

rgb = torch.Tensor(rgb)
rgb = TF.normalize(rgb, (0.485, 0.456, 0.406),(0.229, 0.224, 0.225), inplace=True)


batch_data = {'rgb': torch.Tensor(rgb).unsqueeze(0).movedim(-1, 1).cuda(), 'dep': torch.Tensor(depth).unsqueeze(0).unsqueeze(0).cuda()}


net = CompletionFormer(args)
net.cuda()

path_to_weights = "pretrained/KITTIDC_L1.pt" #"pretrained/NYUv2.pt" #"pretrained/KITTIDC_L1.pt"
checkpoint = torch.load(path_to_weights)
key_m, key_u = net.load_state_dict(checkpoint['net'], strict=False)
if key_u:
    print('Unexpected keys :')
    print(key_u)

if key_m:
    print('Missing keys :')
    print(key_m)
    raise KeyError
print('Checkpoint loaded from {}!'.format(path_to_weights))

net = nn.DataParallel(net)
net.eval()

output= net(batch_data)

print(output['pred'].shape)

pred = output['pred'].squeeze().detach().cpu() * 256.0
plt.figure(figsize=(20,40))
plt.imsave("depth_map_dense_completionformernn_%06d.png" % cur_id, pred)

pred = output['pred'].detach()

pred = torch.clamp(pred, min=0)

pred = pred[0, 0, :, :].data.cpu().numpy()

pred = (pred*256.0).astype(np.uint16)
pred = Image.fromarray(pred)
pred.save("depth_map_dense_2_completionformernn_%06d.png"% cur_id)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--prop_time',
                        type=int,
                        default=6,
                        help='number of propagation')
    parser.add_argument('--prop_kernel',
                        type=int,
                        default=3,
                        help='propagation kernel size')
    parser.add_argument('--conf_prop',
                        action='store_true',
                        default=True,
                        help='confidence for propagation')
    parser.add_argument('--affinity',
                        type=str,
                        default='TGASS',
                        choices=('AS', 'ASS', 'TC', 'TGASS'),
                        help='affinity type (dynamic pos-neg, dynamic pos, '
                             'static pos-neg, static pos, none')
    parser.add_argument('--affinity_gamma',
                        type=float,
                        default=0.5,
                        help='affinity gamma initial multiplier '
                             '(gamma = affinity_gamma * number of neighbors')
    parser.add_argument('--legacy',
                        action='store_true',
                        default=False,
                        help='legacy code support for pre-trained models')
    parser.add_argument('--preserve_input',
                        action='store_true',
                        default=False,
                        help='preserve input points by replacement')
    parser.add_argument('--cur',
                        type=str,
                        default='TGASS')


    args = parser.parse_args()
    #print(args.cur[75:-4])
    complete(args)