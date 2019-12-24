import torch
import glob
import os
from argparse import ArgumentParser
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm
import numpy as np

from utilities.print_utils import *
from transforms.classification.data_transforms import MEAN, STD
from utilities.utils import model_parameters, compute_flops, AverageMeter

from utilities.metrics.segmentation_miou import MIOU
from utilities.train_eval_seg import val_seg as val
from loss_fns.segmentation_loss import SegmentationLoss

from pruning.methods import weight_prune, filter_prune

# ===========================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
# ============================================


def data_transform(img, im_size):
    img = img.resize(im_size, Image.BILINEAR)
    img = F.to_tensor(img)  # convert to tensor (values between 0 and 1)
    img = F.normalize(img, MEAN, STD)  # normalize the tensor
    return img


def evaluate(args, model, image_list, seg_classes, device):
    im_size = tuple(args.im_size)

    # get color map for pascal dataset
    if args.dataset == 'pascal':
        from utilities.color_map import VOCColormap
        cmap = VOCColormap().get_color_map_voc()
    else:
        cmap = None

    model.eval()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()

    miou_class = MIOU(num_classes=seg_classes)

    for i, imgName in tqdm(enumerate(image_list)):
        img = Image.open(imgName).convert('RGB')
        w, h = img.size

        img = data_transform(img, im_size)
        img = img.unsqueeze(0)  # add a batch dimension
        img = img.to(device)
        img_out = model(img)
        img_out = img_out.squeeze(0)  # remove the batch dimension
        img_out = img_out.max(0)[1].byte()  # get the label map
        img_out = img_out.to(device='cpu').numpy()

        img_out = Image.fromarray(img_out)
        # resize to original size
        img_out = img_out.resize((w, h), Image.NEAREST)

        # pascal dataset accepts colored segmentations
        if args.dataset == 'pascal':
            img_out.putpalette(cmap)

        # save the segmentation mask
        name = imgName.split('/')[-1]
        img_extn = imgName.split('.')[-1]
        name = '{}/{}'.format(args.savedir, name.replace(img_extn, 'png'))
        img_out.save(name)


def main(args):
    # read all the images in the folder
    from data_loader.segmentation.voc import VOCSegmentation, VOC_CLASS_LIST
    val_dataset = VOCSegmentation(root=args.data_path, train=False, crop_size=(256, 256), scale=args.s)
    seg_classes = len(VOC_CLASS_LIST)
    class_wts = torch.ones(seg_classes)

    if len(val_dataset) == 0:
        print_error_message('No files in directory: {}'.format(image_path))

    print_info_message('# of images for testing: {}'.format(len(val_dataset)))

    from model.segmentation.prune_espnetv2 import espnetv2_seg
    args.classes = seg_classes
    model = espnetv2_seg(args)

    # model information
    num_params = model_parameters(model)
    flops = compute_flops(model, input=torch.Tensor(1, 3, args.im_size[0], args.im_size[1]))
    print_info_message('FLOPs for an input of size {}x{}: {:.2f} million'.format(args.im_size[0], args.im_size[1], flops))
    print_info_message('# of parameters: {}'.format(num_params))

    print_info_message('Loading model weights')
    weight_dict = torch.load(args.weights_test, map_location=torch.device('cpu'))
    model.load_state_dict(weight_dict)
    print_info_message('Weight loaded successfully')

    # masks = weight_prune(model, args.prune_percentage)
    masks = filter_prune(model, args.prune_percentage)

    # for name, param in model.named_parameters():
    # 	if len(param.data.size()) != 1:
    # 		# print(name, param.data.size())
    # 		print(name)
    
    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu'
    model = model.to(device=device)

    model.set_masks(masks)    

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=40, shuffle=False,
                                             pin_memory=True, num_workers=4)

    criterion = SegmentationLoss(n_classes=seg_classes, loss_type='ce',
                                 device=device, ignore_idx=255,
                                 class_wts=class_wts.to(device))
    if num_gpus >= 1:
        if num_gpus == 1:
            # for a single GPU, we do not need DataParallel wrapper for Criteria.
            # So, falling back to its internal wrapper
            from torch.nn.parallel import DataParallel
            model = DataParallel(model)
            model = model.cuda()
            criterion = criterion.cuda()
        else:
            from utilities.parallel_wrapper import DataParallelModel, DataParallelCriteria
            model = DataParallelModel(model)
            model = model.cuda()
            criterion = DataParallelCriteria(criterion)
            criterion = criterion.cuda()

        if torch.backends.cudnn.is_available():
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            cudnn.deterministic = True

    # # evaluate(args, model, image_list, seg_classes, device=device)
    miou_val, val_loss = val(model, val_loader, criterion, seg_classes, device=device)
    # print_info_message('mIOU: {}'.format(miou_val))


if __name__ == '__main__':
    from commons.general_details import segmentation_models, segmentation_datasets

    parser = ArgumentParser()
    # mdoel details
    parser.add_argument('--model', default="espnetv2", choices=segmentation_models, help='Model name')
    parser.add_argument('--weights-test', default='', help='Pretrained weights directory.')
    parser.add_argument('--s', default=2.0, type=float, help='scale')
    # dataset details
    parser.add_argument('--data-path', default="", help='Data directory')
    parser.add_argument('--dataset', default='city', choices=segmentation_datasets, help='Dataset name')
    # input details
    parser.add_argument('--im-size', type=int, nargs="+", default=[512, 256], help='Image size for testing (W x H)')
    parser.add_argument('--split', default='val', choices=['val', 'test'], help='data split')
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')
    parser.add_argument('--num-classes', default=1000, type=int,
                        help='ImageNet classes. Required for loading the base network')

    parser.add_argument('--prune-percentage', default=0.0, type=float)

    args = parser.parse_args()

    args.weights_test = 'model/segmentation/model_zoo/espnetv2/espnetv2_s_2.0_pascal_384x384.pth'
    args.data_path = 'vision_datasets/pascal_voc/VOCdevkit'
    args.savedir = 'pruning'
    args.dataset = 'pascal'

    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    # This key is used to load the ImageNet weights while training. So, set to empty to avoid errors
    args.weights = ''

    main(args)
