# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import warnings

# -----------------------------------------------------------------------------
# NumPy>=1.24 compatibility for imgaug (imgaug still uses deprecated aliases)
# -----------------------------------------------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]
    if not hasattr(np, "object"):
        np.object = object  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "complex"):
        np.complex = complex  # type: ignore[attr-defined]
    if not hasattr(np, "str"):
        np.str = str  # type: ignore[attr-defined]
    if not hasattr(np, "long"):
        np.long = int  # type: ignore[attr-defined]
    if not hasattr(np, "unicode"):
        np.unicode = str  # type: ignore[attr-defined]

import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import ft_net, two_view_net, three_view_net
from utils import load_network
from image_folder import customData, customData_one, customData_style, ImageFolder_iaa
import imgaug.augmenters as iaa
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='./data/test',type=str, help='./test_data')
parser.add_argument('--name', default='three_view_long_share_d0.75_256_s1_google', type=str, help='save model path')
parser.add_argument('--pool', default='avg', type=str, help='avg|max')
parser.add_argument('--style', default='none', type=str, help='select image style: e.g. night, nightfall, NightLight, shadow, StrongLight, all')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='views')
parser.add_argument('--pad', default=0, type=int, help='padding')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--LPN', action='store_true', help='use LPN' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--scale_test', action='store_true', help='scale test' )
parser.add_argument('--iaa', action='store_true', help='iaa image augmentation' )
parser.add_argument('--modes', default='both', type=str, choices=['d2s', 's2d', 'both'],
                    help='Eval directions: d2s=query_drone->gallery_satellite, s2d=query_satellite->gallery_drone, both=run both (d2s then s2d)')
parser.add_argument(
    '--weather',
    default='all',
    type=str,
    help=(
        "When --iaa is enabled, which weather(s) to evaluate. "
        "Use 'all' (default) or a comma-separated subset, e.g. 'fog' or 'fog,rain'. "
        "Valid: normal,dark,rain,snow,fog,fog_rain,fog_snow,rain_snow,light,wind"
    ),
)
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

opt = parser.parse_args()
###load config###

#debug
# opt.iaa = True
# opt.name = 'three_view_long_share_d0.75_256_s1_google_lr0.005_spade_v24.5_210ep_weather_1010000_5std'
# opt.test_dir = '/data/test'
# opt.batchsize = 4




# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
opt.fp16 = config['fp16'] 
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']
opt.views = config['views']
opt.LPN = config['LPN']
opt.block = config['block']
scale_test = opt.scale_test
style = opt.style
if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']
print('------------------------------',opt.h)
if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 729 

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,gpu_ids))
    # torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Transform used after pixel-level translation.
transform_move_list = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


if opt.LPN:
    data_transforms = transforms.Compose([
        # transforms.Resize((384,192), interpolation=3),
        transforms.Resize((opt.h,opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])
# Build imgaug augmentation from a weather key.
def build_iaa_transform(weather, h, w):
    aug_list = []

    if weather in ['normal', 'none']:
        pass
    elif weather == 'dark':
        aug_list += [
            iaa.BlendAlpha(0.5, foreground=iaa.Add(100), background=iaa.Multiply(0.2), seed=31),
            iaa.MultiplyAndAddToBrightness(mul=0.2, add=(-30, -15), seed=1991),
        ]
    elif weather == 'rain':
        aug_list += [
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=38),
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=73),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=93),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=95),
        ]
    elif weather == 'snow':
        aug_list += [
            iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=38),
            iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=94),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=96),
        ]
    elif weather == 'fog':
        aug_list += [
            iaa.CloudLayer(
                intensity_mean=225,
                intensity_freq_exponent=-2,
                intensity_coarse_scale=2,
                alpha_min=1.0,
                alpha_multiplier=0.9,
                alpha_size_px_max=10,
                alpha_freq_exponent=-2,
                sparsity=0.9,
                density_multiplier=0.5,
                seed=35,
            ),
        ]
    elif weather in ['fog_rain', 'rain_fog']:
        aug_list += [
            iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0,
                           alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
                           density_multiplier=0.5, seed=35),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=36),
        ]
    elif weather in ['fog_snow', 'snow_fog']:
        aug_list += [
            iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0,
                           alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
                           density_multiplier=0.5, seed=35),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=35),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=36),
        ]
    elif weather == 'rain_snow':
        aug_list += [
            iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=92),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=91),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
        ]
    elif weather in ['light', 'overexposure']:
        aug_list += [
            iaa.MultiplyAndAddToBrightness(mul=1.6, add=(0, 30), seed=1992),
        ]
    elif weather == 'wind':
        aug_list += [
            iaa.MotionBlur(15, seed=17),
        ]
    else:
        print(f"[WARN] Unknown weather '{weather}', fallback to no weather augmentation.")

    aug_list.append(iaa.Resize({"height": h, "width": w}, interpolation=3))
    return iaa.Sequential(aug_list)


# using iaa image augmentation
if opt.iaa:
    # Initialize a default weather transform; each weather loop overwrites it.
    iaa_transform = build_iaa_transform('normal', opt.h, opt.w)
    data_transforms_iaa = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
data_dir = test_dir

if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
elif opt.iaa:
    print('------------------processing images using iaa----------------------')
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x), data_transforms) for x in ['gallery_satellite','query_satellite']}
    for x in ['query_drone', 'gallery_drone']:
        image_datasets[x] = ImageFolder_iaa(os.path.join(data_dir,x), data_transforms_iaa, iaa_transform=iaa_transform)

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=8) for x in ['gallery_satellite', 'gallery_drone', 'query_satellite', 'query_drone']}





else:
    # image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery_satellite','gallery_drone', 'gallery_street', 'query_satellite', 'query_drone', 'query_street']}
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery_satellite','gallery_drone', 'gallery_street']}
    # image_datasets = {}
    # for x in ['gallery_satellite','gallery_drone', 'gallery_street', 'gallery_satellite_usa_un']:
    #     image_datasets[x] = customData( os.path.join(data_dir,x) ,data_transforms, rotate=0)
    if scale_test:
        for x in ['query_drone']:
            print('----------scale test--------------')
            image_datasets[x] = customData_one( os.path.join(data_dir,x) ,data_transforms, rotate=0, reverse=False)
    else:
        for x in ['query_satellite', 'query_drone', 'query_street']:
            if opt.pad > 0:
                print('-----------move pixel test-----------')
                image_datasets[x] = customData( os.path.join(data_dir,x) ,transform_move_list, rotate=0, pad=opt.pad)
            else: 
                print('----------rotation test--------------')   
                image_datasets[x] = customData( os.path.join(data_dir,x) ,data_transforms, rotate=0)
    if style != 'none':
        for x in ['query_drone_style', 'gallery_drone_style']:
            image_datasets[x] = customData_style( os.path.join(data_dir,x) ,data_transforms, style=style)

    print(image_datasets.keys())
    # image_datasets = {x: customData( os.path.join(data_dir,x) ,data_transforms, rotate=0) for x in ['query_satellite', 'query_drone', 'query_street']}
    if scale_test:
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery_satellite', 'gallery_drone','gallery_street', 'query_drone']}
    else:
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=8) for x in ['gallery_satellite', 'gallery_drone','gallery_street', 'query_satellite', 'query_drone']}
    if style != 'none':
        print('using style is-----------------:', style)
        dataloaders['query_drone_style'] =  torch.utils.data.DataLoader(image_datasets['query_drone_style'], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16)
        dataloaders['gallery_drone_style'] =  torch.utils.data.DataLoader(image_datasets['gallery_drone_style'], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16)
use_gpu = torch.cuda.is_available()

######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def which_view(name):
    if 'satellite' in name:
        return 1
    elif 'street' in name:
        return 2
    elif 'drone' in name:
        return 3
    else:
        print('unknown view')
    return -1

def extract_feature(model,dataloaders, view_index = 1):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,512).zero_().cuda()
        if opt.LPN:
            # ff = torch.FloatTensor(n,2048,6).zero_().cuda()
            ff = torch.FloatTensor(n,512,opt.block).zero_().cuda()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
                if opt.views ==2:
                    if view_index == 1:
                        outputs, _ = model(input_img, None) 
                    elif view_index ==2:
                        _, outputs = model(None, input_img) 
                elif opt.views ==3:
                    if view_index == 1:
                        outputs, _, _ = model(input_img, None, None)
                    elif view_index ==2:
                        _, outputs, _ = model(None, input_img, None)
                    elif view_index ==3:
                        _, _, outputs = model(None, None, input_img)
                ff += outputs
        # norm feature
        if opt.LPN:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(opt.block) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features




def get_id(img_path):
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        # print(path, v)
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths





######################################################################
# Load Collected data Trained model
print('-------test-----------')

model, _, epoch = load_network(opt.name, opt)
if opt.LPN:
    print('use LPN')
    # model = three_view_net_test(model)
    for i in range(opt.block):
        cls_name = 'classifier'+str(i)
        c = getattr(model, cls_name)
        c.classifier = nn.Sequential()
else:
    model.classifier.classifier = nn.Sequential()
model = model.eval()
if use_gpu:
    model = model.cuda()
# print(model)
# Extract feature
since = time.time()

# gallery_name = 'gallery_street'

WEATHER_LIST = [
    'normal', 'dark', 'rain', 'snow',
    'fog', 'fog_rain', 'fog_snow', 'rain_snow',
    'light', 'wind'
]

if __name__ == "__main__":
    if not opt.iaa:
        weather_list = ['normal']
    else:
        w = (opt.weather or "all").strip().lower()
        if w in {"all", "*"}:
            weather_list = WEATHER_LIST
        else:
            weather_list = [x.strip().lower() for x in w.split(",") if x.strip()]
            unknown = [x for x in weather_list if x not in WEATHER_LIST]
            if unknown:
                raise ValueError(f"Unknown --weather: {unknown}. Valid: {WEATHER_LIST} or 'all'")
    last_mat_path = None
    last_dir_tag = None
    last_weather_key = None

    model_dir = os.path.join('./model', opt.name)
    os.makedirs(model_dir, exist_ok=True)

    # Run directions automatically: d2s then s2d (as requested).
    directions_all = [
        ('d2s', 'query_drone', 'gallery_satellite'),
        ('s2d', 'query_satellite', 'gallery_drone'),
    ]
    if opt.modes == 'd2s':
        direction_list = [directions_all[0]]
    elif opt.modes == 's2d':
        direction_list = [directions_all[1]]
    else:
        direction_list = directions_all

    for dir_tag, query_name, gallery_name in direction_list:
        which_gallery = which_view(gallery_name)
        which_query = which_view(query_name)
        print(f"\n================ Direction: {dir_tag} ({query_name} -> {gallery_name}) ================")
        print('%d -> %d:' % (which_query, which_gallery))

        gallery_path = image_datasets[gallery_name].imgs
        with open(os.path.join(model_dir, f'gallery_name_{dir_tag}.txt'), 'w') as f:
            for p in gallery_path:
                f.write(p[0] + '\n')
        query_path = image_datasets[query_name].imgs
        with open(os.path.join(model_dir, f'query_name_{dir_tag}.txt'), 'w') as f:
            for p in query_path:
                f.write(p[0] + '\n')

        gallery_label, gallery_path = get_id(gallery_path)
        query_label, query_path = get_id(query_path)

        for weather_key in weather_list:
            print(f"\n================ Weather: {weather_key} ================")

            # Reconfigure weather augmentation for each weather condition on drone views.
            if opt.iaa:
                new_iaa = build_iaa_transform(weather_key, opt.h, opt.w)
                if 'gallery_drone' in image_datasets:
                    image_datasets['gallery_drone'].iaa_trans = new_iaa
                if 'query_drone' in image_datasets:
                    image_datasets['query_drone'].iaa_trans = new_iaa

            since = time.time()
            with torch.no_grad():
                query_feature = extract_feature(model, dataloaders[query_name], which_query)
                gallery_feature = extract_feature(model, dataloaders[gallery_name], which_gallery)

            time_elapsed = time.time() - since
            print('Test [{}|{}] complete in {:.0f}m {:.0f}s'.format(
                dir_tag, weather_key, time_elapsed // 60, time_elapsed % 60))

            # Save to Matlab for check (evaluate_gpu.py reads the mat file)
            result = {
                'gallery_f': gallery_feature.numpy(),
                'gallery_label': gallery_label,
                'gallery_path': gallery_path,
                'query_f': query_feature.numpy(),
                'query_label': query_label,
                'query_path': query_path,
            }
            # Save per-run/per-model mat to avoid races when multiple processes run in parallel.
            mat_path = os.path.join(model_dir, f'pytorch_result_{dir_tag}_{weather_key}.mat')
            scipy.io.savemat(mat_path, result)
            last_mat_path = mat_path
            last_dir_tag = dir_tag
            last_weather_key = weather_key

            print(opt.name, f'Direction: {dir_tag}', 'Weather:', weather_key)
            # Write one result file for each direction and weather condition.
            result_txt = os.path.join(model_dir, f'result_{dir_tag}_{weather_key}.txt')
            os.system('CUDA_VISIBLE_DEVICES=%d python evaluate_gpu.py --mat %s | tee -a %s'
                      % (gpu_ids[0], mat_path, result_txt))

    # Evaluate function to get the top 10 matches
    def evaluate_top10_matches(qf, ql, gf, gl, g_paths):
        query = qf.view(-1,1)
        score = torch.mm(gf, query)
        score = score.squeeze(1).cpu()
        score = score.numpy()
        index = np.argsort(score)[::-1][:10]  # Get the top 10 indices
        # Extracting the matched image paths
        match_paths = [g_paths[idx] for idx in index]
        return match_paths

    # Load features and labels
    # Use the last saved mat file (or fallback to the legacy path).
    result = scipy.io.loadmat(last_mat_path or 'pytorch_result.mat')
    query_feature = torch.FloatTensor(result['query_f'])
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_label = result['gallery_label'][0]
    query_paths = result['query_path']
    gallery_paths = result['gallery_path']

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    # Store all query top 10 match results
    all_matches = {}

    for i in range(len(query_label)):
        matches = evaluate_top10_matches(query_feature[i], query_label[i], gallery_feature, gallery_label, gallery_paths)
        all_matches[query_paths[i]] = matches

    # Write the match results into a text file
    top10_name = 'top10_matches.txt'
    if last_dir_tag and last_weather_key:
        top10_name = f'top10_matches_{last_dir_tag}_{last_weather_key}.txt'
    with open(os.path.join(model_dir, top10_name), 'w') as f:
        for query, matches in all_matches.items():
            f.write(f'Query: {query}\n')
            for match in matches:
                f.write(f'{match}\n')
            f.write('\n')

    print(f"Matching results have been saved to '{top10_name}' in model dir.")

    # os.system('python evaluate_gpu.py | tee -a %s'%result)
    #test single part and combination
    '''
    # for i in range(7):
    #     if i == 0:
    #         query_feature_ = query_feature[:,0:512]
    #         gallery_feature_ = gallery_feature[:,0:512]
    #         print('-------------- 1 -----------------')
    #     if i == 1:
    #         query_feature_ = query_feature[:,512:1024]
    #         gallery_feature_ = gallery_feature[:,512:1024]
    #         print('-------------- 2 -----------------')
    #     if i == 2:
    #         query_feature_ = query_feature[:,1024:1536]
    #         gallery_feature_ = gallery_feature[:,1024:1536]
    #         print('-------------- 3 -----------------')
    #     if i == 3:
    #         query_feature_ = query_feature[:,1536:2048]
    #         gallery_feature_ = gallery_feature[:,1536:2048]
    #         print('-------------- 4 -----------------')
    #     if i == 4:
    #         query_feature_ = query_feature[:,0:1024]
    #         gallery_feature_ = gallery_feature[:,0:1024]
    #         print('-------------- 1+2 -----------------')
    #     if i == 5:
    #         query_feature_ = query_feature[:,0:1536]
    #         gallery_feature_ = gallery_feature[:,0:1536]
    #         print('-------------- 1+2+3 -----------------')
    #     if i == 6:
    #         query_feature_ = query_feature[:,0:2048]
    #         gallery_feature_ = gallery_feature[:,0:2048]
    #         print('-------------- 1+2+3+4 -----------------')
    #     result = {'gallery_f':gallery_feature_.numpy(),'gallery_label':gallery_label,'gallery_path':gallery_path,'query_f':query_feature_.numpy(),'query_label':query_label, 'query_path':query_path}
    #     scipy.io.savemat('pytorch_result.mat',result)
    #     print(opt.name)
    #     result = './model/%s/result.txt'%opt.name
    #     os.system('CUDA_VISIBLE_DEVICES=%d python evaluate_gpu.py | tee -a %s'%(gpu_ids[0],result))
    '''
    # query_feature_ = query_feature[:,0:1536]
    # gallery_feature_ = gallery_feature[:,512:2048]
    # print('-------------- （1+2+3，2+3+4） -----------------')
    # result = {'gallery_f':gallery_feature_.numpy(),'gallery_label':gallery_label,'gallery_path':gallery_path,'query_f':query_feature_.numpy(),'query_label':query_label, 'query_path':query_path}
    # scipy.io.savemat('pytorch_result.mat',result)
    # print(opt.name)
    # result = './model/%s/result.txt'%opt.name
    # os.system('CUDA_VISIBLE_DEVICES=%d python evaluate_gpu.py | tee -a %s'%(gpu_ids[0],result))
