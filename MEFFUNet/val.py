import argparse
import os
from glob import glob

import torch.nn as nn

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
import albumentations as albu
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

import archs
from dataset import Dataset
from metrics import iou_score, dice_coef, recall_score, precision_score, multi_class_iou_score, hausdorff_distance_95
from utils import AverageMeter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    with open('models/%s/config.yml' % (args.name), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    model_name = config['arch'] + '/' + config['name']
    print("=> creating model %s" % config['arch'])

    model = archs.__dict__[config['arch']](num_classes = config['num_classes'],
                                       input_channels = config['input_channels'],
                                       deep_supervision = config['deep_supervision'],
                                       deep_supervision_unet = config['deep_supervision_unet'],
                                       ds_unetmnx = config['ds_unetmnx'],
                                       kernel_size = config.get('mnx_kernel_size', 3)   # 使用 get 方法，如果 'mnx_kernel_size' 不存在，则返回默认值3
                                       )

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    #summary(model, (3, 256, 256), device="cuda")
    print('#######################################')


    # Data loading code
    # if config['dataset'] == 'spine1k':
    #     img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*_CT', '*' + config['img_ext']))
    #     img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    # else :
    #     img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    #     img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    #img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*_CT'))

    img_ids = []
    if config['dataset'] == 'spine1k_without_pre_more':
        img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*_CT'))
    elif config['dataset'] == 'totalsegmentator_slices' or 'totalsegmentator_slices_resize':
        img_ids = glob(os.path.join('inputs', config['dataset'], 'images', 's*'))
        img_ids = img_ids[:200]  # 选数据集中的一部分
    #print(len(img_ids))
    
    ###########################################


    # if 'valsize' in config.keys():
    #     _, val_img_ids = train_test_split(img_ids, test_size=config['valsize'], random_state=27)  #seed 原来是41
    #     print('--valsize loaded.')
    # else:
    #     _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=27)  #seed 原来是41
    #     print('* [Model Configuration] Validation split size not specified in this old config, using default value of 0.2.')

    _, val_img_path = train_test_split(img_ids, test_size=config['valsize'], random_state=27) #随机seed

    val_img_ids = []
    ###############################

    for patient in val_img_path:
        patient_path = glob(os.path.join(patient, '*' + config['img_ext']))
        for p in patient_path:
            val_img_ids.append(os.path.splitext(os.path.basename(p))[0])

    model.load_state_dict(torch.load(f'models/{model_name}/model.pth', weights_only=False))
    # checkpoint = torch.load(f'models/{model_name}/checkpoint.pth', weights_only=False)
    # model.load_state_dict(checkpoint['model'])
    model.eval()

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    img_dir=os.path.join('inputs', config['dataset'], 'images')
    mask_dir=os.path.join('inputs', config['dataset'], 'masks')
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_meter = AverageMeter()
    dice_meter = AverageMeter()
    recall_meter = AverageMeter()
    precision_meter = AverageMeter()
    hd_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision'] or config['deep_supervision_unet']:
                output = model(input)[-1]
            elif config['ds_unetmnx']:
                output = model(input)[0]
            else:
                output = model(input)


            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            recall = recall_score(output, target)
            precision = precision_score(output, target)
            hd = hausdorff_distance_95(output, target)

            iou_meter.update(iou, input.size(0))
            dice_meter.update(dice, input.size(0))
            recall_meter.update(recall, input.size(0))
            precision_meter.update(precision, input.size(0))
            hd_meter.update(hd, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            # 在多分类时这里是分别将不同的分类取出来分别放到不同的文件夹中
            for i in range(len(output)): #len(tensor) == tensor.shape[0]
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('Evaluation Metrics:')
    print('------------------')
    print(f'IoU:        {iou_meter.avg:.4f}')
    print(f'Dice:       {dice_meter.avg:.4f}')
    print(f'Recall:     {recall_meter.avg:.4f}')
    print(f'Precision:  {precision_meter.avg:.4f}')
    print(f'HD:  {hd_meter.avg:.4f}')
    print('------------------')

    # Save evaluation metrics to a file
    results_file_local = os.path.join('outputs', config['name'], 'evaluation_results.txt')
    with open(results_file_local, 'w') as f:
        f.write('Evaluation Metrics:\n')
        f.write('------------------\n')
        f.write(f'IoU:        {iou_meter.avg:.4f}\n')
        f.write(f'Dice:       {dice_meter.avg:.4f}\n')
        f.write(f'Recall:     {recall_meter.avg:.4f}\n')
        f.write(f'Precision:  {precision_meter.avg:.4f}\n')
        f.write(f'HD:  {hd_meter.avg:.4f}\n')
        f.write('------------------\n')
    
    print(f"* Results saved to {results_file_local}")


    # Append results to the results file
    results_file = os.path.join('val_outputs','val_results.csv')
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
    else:
        df = pd.DataFrame(columns=['Model', 'IoU', 'Dice', 'Recall', 'Precision'])

    
    new_row = pd.DataFrame({
        'Model': [model_name],
        'IoU': [iou_meter.avg],
        'Dice': [dice_meter.avg],
        'Recall': [recall_meter.avg],
        'Precision': [precision_meter.avg],
        'HD': [hd_meter.avg]
    })

    # Ensure column types are consistent
    new_row = new_row.astype(df.dtypes)

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(results_file, index=False)

    print(f"* Results appended to {results_file}")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
