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

# ===========================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
# ============================================


def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    '''
    img[img == 19] = 255
    img[img == 18] = 33
    img[img == 17] = 32
    img[img == 16] = 31
    img[img == 15] = 28
    img[img == 14] = 27
    img[img == 13] = 26
    img[img == 12] = 25
    img[img == 11] = 24
    img[img == 10] = 23
    img[img == 9] = 22
    img[img == 8] = 21
    img[img == 7] = 20
    img[img == 6] = 19
    img[img == 5] = 17
    img[img == 4] = 13
    img[img == 3] = 12
    img[img == 2] = 11
    img[img == 1] = 8
    img[img == 0] = 7
    img[img == 255] = 0
    return img


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

        if args.dataset == 'city':
            # cityscape uses different IDs for training and testing
            # so, change from Train IDs to actual IDs
            img_out = relabel(img_out)

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
    if args.dataset == 'city':
        # image_path = os.path.join(args.data_path, "leftImg8bit", args.split, "*", "*.png")
        # image_list = glob.glob(image_path)
        # from data_loader.segmentation.cityscapes import CITYSCAPE_CLASS_LIST
        # seg_classes = len(CITYSCAPE_CLASS_LIST)
        from data_loader.segmentation.cityscapes import CityscapesSegmentation, CITYSCAPE_CLASS_LIST
        val_dataset = CityscapesSegmentation(root=args.data_path, train=False, size=(256, 256), scale=args.s,
                                             coarse=False)
        seg_classes = len(CITYSCAPE_CLASS_LIST)
        class_wts = torch.ones(seg_classes)
        class_wts[0] = 2.8149201869965
        class_wts[1] = 6.9850029945374
        class_wts[2] = 3.7890393733978
        class_wts[3] = 9.9428062438965
        class_wts[4] = 9.7702074050903
        class_wts[5] = 9.5110931396484
        class_wts[6] = 10.311357498169
        class_wts[7] = 10.026463508606
        class_wts[8] = 4.6323022842407
        class_wts[9] = 9.5608062744141
        class_wts[10] = 7.8698215484619
        class_wts[11] = 9.5168733596802
        class_wts[12] = 10.373730659485
        class_wts[13] = 6.6616044044495
        class_wts[14] = 10.260489463806
        class_wts[15] = 10.287888526917
        class_wts[16] = 10.289801597595
        class_wts[17] = 10.405355453491
        class_wts[18] = 10.138095855713
        class_wts[19] = 0.0
    elif args.dataset == 'pascal':
        # from data_loader.segmentation.voc import VOC_CLASS_LIST
        # seg_classes = len(VOC_CLASS_LIST)
        # data_file = os.path.join(args.data_path, 'VOC2012', 'list', '{}.txt'.format(args.split))
        # if not os.path.isfile(data_file):
        #     print_error_message('{} file does not exist'.format(data_file))
        # image_list = []
        # with open(data_file, 'r') as lines:
        #     for line in lines:
        #         rgb_img_loc = '{}/{}/{}'.format(args.data_path, 'VOC2012', line.split()[0])
        #         if not os.path.isfile(rgb_img_loc):
        #             print_error_message('{} image file does not exist'.format(rgb_img_loc))
        #         image_list.append(rgb_img_loc)
        from data_loader.segmentation.voc import VOCSegmentation, VOC_CLASS_LIST
        val_dataset = VOCSegmentation(root=args.data_path, train=False, crop_size=(256, 256), scale=args.s)
        seg_classes = len(VOC_CLASS_LIST)
        class_wts = torch.ones(seg_classes)
    elif args.dataset == 'hockey':
        from data_loader.segmentation.hockey import HockeySegmentationDataset, HOCKEY_DATASET_CLASS_LIST
        train_dataset = HockeySegmentationDataset(root=args.data_path, train=True, crop_size=(256, 256), scale=args.s)
        val_dataset = HockeySegmentationDataset(root=args.data_path, train=False, crop_size=(256, 256), scale=args.s)
        seg_classes = len(HOCKEY_DATASET_CLASS_LIST)
        class_wts = torch.ones(seg_classes)
    elif args.dataset == 'hockey_rink_seg':
        from data_loader.segmentation.hockey_rink_seg import HockeyRinkSegmentationDataset, HOCKEY_DATASET_CLASS_LIST
        train_dataset = HockeyRinkSegmentationDataset(root=args.data_path, train=True, crop_size=(256, 256), scale=args.s)
        val_dataset = HockeyRinkSegmentationDataset(root=args.data_path, train=False, crop_size=(256, 256), scale=args.s)
        seg_classes = len(HOCKEY_DATASET_CLASS_LIST)
        class_wts = torch.ones(seg_classes)
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))

    if len(val_dataset) == 0:
        print_error_message('No files in directory: {}'.format(image_path))

    print_info_message('# of images for testing: {}'.format(len(val_dataset)))

    if args.model == 'espnetv2':
        from model.segmentation.espnetv2 import espnetv2_seg
        args.classes = seg_classes
        model = espnetv2_seg(args)
    elif args.model == 'dicenet':
        from model.segmentation.dicenet import dicenet_seg
        model = dicenet_seg(args, classes=seg_classes)
    else:
        print_error_message('{} network not yet supported'.format(args.model))
        exit(-1)

    # model information
    num_params = model_parameters(model)
    flops = compute_flops(model, input=torch.Tensor(1, 3, args.im_size[0], args.im_size[1]))
    print_info_message('FLOPs for an input of size {}x{}: {:.2f} million'.format(args.im_size[0], args.im_size[1], flops))
    print_info_message('# of parameters: {}'.format(num_params))

    if args.weights_test:
        print_info_message('Loading model weights')
        weight_dict = torch.load(args.weights_test, map_location=torch.device('cpu'))

        if isinstance(weight_dict, dict) and 'state_dict' in weight_dict:
            model.load_state_dict(weight_dict['state_dict'])
        else:
            model.load_state_dict(weight_dict)

        print_info_message('Weight loaded successfully')
    else:
        print_error_message('weight file does not exist or not specified. Please check: {}', format(args.weights_test))

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu'
    model = model.to(device=device)

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

    # evaluate(args, model, image_list, seg_classes, device=device)
    miou_val, val_loss = val(model, val_loader, criterion, seg_classes, device=device)
    print_info_message('mIOU: {}'.format(miou_val))


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

    args = parser.parse_args()

    if not args.weights_test:
        from model.weight_locations.segmentation import model_weight_map

        model_key = '{}_{}'.format(args.model, args.s)
        dataset_key = '{}_{}x{}'.format(args.dataset, args.im_size[0], args.im_size[1])
        assert model_key in model_weight_map.keys(), '{} does not exist'.format(model_key)
        assert dataset_key in model_weight_map[model_key].keys(), '{} does not exist'.format(dataset_key)
        args.weights_test = model_weight_map[model_key][dataset_key]['weights']
        if not os.path.isfile(args.weights_test):
            print_error_message('weight file does not exist: {}'.format(args.weights_test))

    # set-up results path
    if args.dataset == 'city':
        args.savedir = '{}_{}_{}/results'.format('results', args.dataset, args.split)
    elif args.dataset == 'pascal':
        args.savedir = '{}_{}/results/VOC2012/Segmentation/comp6_{}_cls'.format('results', args.dataset, args.split)
    elif args.dataset == 'hockey':
        args.savedir = '{}_{}_{}/results'.format('results', args.dataset, args.split)
    elif args.dataset == 'hockey_rink_seg':
        args.savedir = '{}_{}_{}/results'.format('results', args.dataset, args.split)
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))

    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    # This key is used to load the ImageNet weights while training. So, set to empty to avoid errors
    args.weights = ''

    main(args)
