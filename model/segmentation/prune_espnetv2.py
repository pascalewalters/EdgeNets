# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

import torch
from torch.nn import init
from nn_layers.prune_espnet_utils import *
from nn_layers.prune_efficient_pyramid_pool import EfficientPyrPool
from nn_layers.prune_efficient_pt import EfficientPWConv
from model.classification.prune_espnetv2 import EESPNet
from utilities.print_utils import *
from torch.nn import functional as F


class ESPNetv2Segmentation(nn.Module):
    '''
    This class defines the ESPNetv2 architecture for the Semantic Segmenation
    '''

    def __init__(self, args, classes=21, dataset='pascal'):
        super().__init__()

        # =============================================================
        #                       BASE NETWORK
        # =============================================================
        self.base_net = EESPNet(args) #imagenet model
        del self.base_net.classifier
        del self.base_net.level5
        del self.base_net.level5_0
        config = self.base_net.config

        #=============================================================
        #                   SEGMENTATION NETWORK
        #=============================================================
        dec_feat_dict={
            'pascal': 16,
            'city': 16,
            'hockey': 16,
            'hockey_rink_seg': 16,
            'coco': 32
        }
        base_dec_planes = dec_feat_dict[dataset]
        dec_planes = [4*base_dec_planes, 3*base_dec_planes, 2*base_dec_planes, classes]
        pyr_plane_proj = min(classes //2, base_dec_planes)

        self.bu_dec_l1 = EfficientPyrPool(in_planes=config[3], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[0])
        self.bu_dec_l2 = EfficientPyrPool(in_planes=dec_planes[0], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[1])
        self.bu_dec_l3 = EfficientPyrPool(in_planes=dec_planes[1], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[2])
        self.bu_dec_l4 = EfficientPyrPool(in_planes=dec_planes[2], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[3], last_layer_br=False)

        self.merge_enc_dec_l2 = EfficientPWConv(config[2], dec_planes[0])
        self.merge_enc_dec_l3 = EfficientPWConv(config[1], dec_planes[1])
        self.merge_enc_dec_l4 = EfficientPWConv(config[0], dec_planes[2])

        self.bu_br_l2 = nn.Sequential(nn.BatchNorm2d(dec_planes[0]),
                                      nn.PReLU(dec_planes[0])
                                      # nn.SELU(dec_planes[0])
                                      )
        self.bu_br_l3 = nn.Sequential(nn.BatchNorm2d(dec_planes[1]),
                                      nn.PReLU(dec_planes[1])
                                      # nn.SELU(dec_planes[1])
                                      )
        self.bu_br_l4 = nn.Sequential(nn.BatchNorm2d(dec_planes[2]),
                                      nn.PReLU(dec_planes[2])
                                      # nn.SELU(dec_planes[2])
                                      )

        #self.upsample =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.init_params()

    def upsample(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def get_basenet_params(self):
        modules_base = [self.base_net]
        for i in range(len(modules_base)):
            for m in modules_base[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.PReLU):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_segment_params(self):
        modules_seg = [self.bu_dec_l1, self.bu_dec_l2, self.bu_dec_l3, self.bu_dec_l4,
                       self.merge_enc_dec_l4, self.merge_enc_dec_l3, self.merge_enc_dec_l2,
                       self.bu_br_l4, self.bu_br_l3, self.bu_br_l2]
        for i in range(len(modules_seg)):
            for m in modules_seg[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.PReLU):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def forward(self, x):
        '''
        :param x: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''
        x_size = (x.size(2), x.size(3))
        enc_out_l1 = self.base_net.level1(x)  # 112
        if not self.base_net.input_reinforcement:
            del x
            x = None

        enc_out_l2 = self.base_net.level2_0(enc_out_l1, x)  # 56

        enc_out_l3_0 = self.base_net.level3_0(enc_out_l2, x)  # down-sample
        for i, layer in enumerate(self.base_net.level3):
            if i == 0:
                enc_out_l3 = layer(enc_out_l3_0)
            else:
                enc_out_l3 = layer(enc_out_l3)

        enc_out_l4_0 = self.base_net.level4_0(enc_out_l3, x)  # down-sample
        for i, layer in enumerate(self.base_net.level4):
            if i == 0:
                enc_out_l4 = layer(enc_out_l4_0)
            else:
                enc_out_l4 = layer(enc_out_l4)

        # bottom-up decoding
        bu_out = self.bu_dec_l1(enc_out_l4)

        # Decoding block
        bu_out = self.upsample(bu_out)
        enc_out_l3_proj = self.merge_enc_dec_l2(enc_out_l3)
        bu_out = enc_out_l3_proj + bu_out
        bu_out = self.bu_br_l2(bu_out)
        bu_out = self.bu_dec_l2(bu_out)

        #decoding block
        bu_out = self.upsample(bu_out)
        enc_out_l2_proj = self.merge_enc_dec_l3(enc_out_l2)
        bu_out = enc_out_l2_proj + bu_out
        bu_out = self.bu_br_l3(bu_out)
        bu_out = self.bu_dec_l3(bu_out)

        # decoding block
        bu_out = self.upsample(bu_out)
        enc_out_l1_proj = self.merge_enc_dec_l4(enc_out_l1)
        bu_out = enc_out_l1_proj + bu_out
        bu_out = self.bu_br_l4(bu_out)
        bu_out  = self.bu_dec_l4(bu_out)

        return F.interpolate(bu_out, size=x_size, mode='bilinear', align_corners=True)

    def set_masks(self, masks):
        # Apply masks
        self.base_net.level1.conv.set_mask(masks[0])
        self.base_net.level2_0.eesp.proj_1x1.conv.set_mask(masks[1])
        self.base_net.level2_0.eesp.spp_dw[0].conv.set_mask(masks[2])
        self.base_net.level2_0.eesp.spp_dw[1].conv.set_mask(masks[3])
        self.base_net.level2_0.eesp.spp_dw[2].conv.set_mask(masks[4])
        # model.base_net.level2_0.eesp.spp_dw[3].conv.set_mask(masks[5])
        # model.base_net.level2_0.eesp.conv_1x1_exp.conv.set_mask(masks[6])
        # model.base_net.level2_0.inp_reinf[0].conv.set_mask(masks[7])
        # model.base_net.level2_0.inp_reinf[1].conv.set_mask(masks[8])
        # model.base_net.level3_0.eesp.proj_1x1.conv.set_mask(masks[9])
        # model.base_net.level3_0.eesp.spp_dw[0].conv.set_mask(masks[10])
        # model.base_net.level3_0.eesp.spp_dw[1].conv.set_mask(masks[11])
        # model.base_net.level3_0.eesp.spp_dw[2].conv.set_mask(masks[12])
        # model.base_net.level3_0.eesp.spp_dw[3].conv.set_mask(masks[13])
        # model.base_net.level3_0.eesp.conv_1x1_exp.conv.set_mask(masks[14])
        # model.base_net.level3_0.inp_reinf[0].conv.set_mask(masks[15])
        # model.base_net.level3_0.inp_reinf[1].conv.set_mask(masks[16])
        # model.base_net.level3[0].proj_1x1.conv.set_mask(masks[17])
        # model.base_net.level3[0].spp_dw[0].conv.set_mask(masks[18])
        # model.base_net.level3[0].spp_dw[1].conv.set_mask(masks[19])
        # model.base_net.level3[0].spp_dw[2].conv.set_mask(masks[20])
        # model.base_net.level3[0].spp_dw[3].conv.set_mask(masks[21])
        # model.base_net.level3[0].conv_1x1_exp.conv.set_mask(masks[22])
        # model.base_net.level3[1].proj_1x1.conv.set_mask(masks[23])
        # model.base_net.level3[1].spp_dw[0].conv.set_mask(masks[24])
        # model.base_net.level3[1].spp_dw[1].conv.set_mask(masks[25])
        # model.base_net.level3[1].spp_dw[2].conv.set_mask(masks[26])
        # model.base_net.level3[1].spp_dw[3].conv.set_mask(masks[27])
        # model.base_net.level3[1].conv_1x1_exp.conv.set_mask(masks[28])
        # model.base_net.level3[2].proj_1x1.conv.set_mask(masks[29])
        # model.base_net.level3[2].spp_dw[0].conv.set_mask(masks[30])
        # model.base_net.level3[2].spp_dw[1].conv.set_mask(masks[31])
        # model.base_net.level3[2].spp_dw[2].conv.set_mask(masks[32])
        # model.base_net.level3[2].spp_dw[3].conv.set_mask(masks[33])
        # model.base_net.level3[2].conv_1x1_exp.conv.set_mask(masks[34])
        # model.base_net.level4_0.eesp.proj_1x1.conv.set_mask(masks[35])
        # model.base_net.level4_0.eesp.spp_dw[0].conv.set_mask(masks[36])
        # model.base_net.level4_0.eesp.spp_dw[1].conv.set_mask(masks[37])
        # model.base_net.level4_0.eesp.spp_dw[2].conv.set_mask(masks[38])
        # model.base_net.level4_0.eesp.spp_dw[3].conv.set_mask(masks[39])
        # model.base_net.level4_0.eesp.conv_1x1_exp.conv.set_mask(masks[40])
        # model.base_net.level4_0.inp_reinf[0].conv.set_mask(masks[41])
        # model.base_net.level4_0.inp_reinf[1].conv.set_mask(masks[42])
        # model.base_net.level4[0].proj_1x1.conv.set_mask(masks[43])
        # model.base_net.level4[0].spp_dw[0].conv.set_mask(masks[44])
        # model.base_net.level4[0].spp_dw[1].conv.set_mask(masks[45])
        # model.base_net.level4[0].spp_dw[2].conv.set_mask(masks[46])
        # model.base_net.level4[0].spp_dw[3].conv.set_mask(masks[47])
        # model.base_net.level4[0].conv_1x1_exp.conv.set_mask(masks[48])
        # model.base_net.level4[1].proj_1x1.conv.set_mask(masks[49])
        # model.base_net.level4[1].spp_dw[0].conv.set_mask(masks[50])
        # model.base_net.level4[1].spp_dw[1].conv.set_mask(masks[51])
        # model.base_net.level4[1].spp_dw[2].conv.set_mask(masks[52])
        # model.base_net.level4[1].spp_dw[3].conv.set_mask(masks[53])
        # model.base_net.level4[1].conv_1x1_exp.conv.set_mask(masks[54])
        # model.base_net.level4[2].proj_1x1.conv.set_mask(masks[55])
        # model.base_net.level4[2].spp_dw[0].conv.set_mask(masks[56])
        # model.base_net.level4[2].spp_dw[1].conv.set_mask(masks[57])
        # model.base_net.level4[2].spp_dw[2].conv.set_mask(masks[58])
        # model.base_net.level4[2].spp_dw[3].conv.set_mask(masks[59])
        # model.base_net.level4[2].conv_1x1_exp.conv.set_mask(masks[60])
        # model.base_net.level4[3].proj_1x1.conv.set_mask(masks[61])
        # model.base_net.level4[3].spp_dw[0].conv.set_mask(masks[62])
        # model.base_net.level4[3].spp_dw[1].conv.set_mask(masks[63])
        # model.base_net.level4[3].spp_dw[2].conv.set_mask(masks[64])
        # model.base_net.level4[3].spp_dw[3].conv.set_mask(masks[65])
        # model.base_net.level4[3].conv_1x1_exp.conv.set_mask(masks[66])
        # model.base_net.level4[4].proj_1x1.conv.set_mask(masks[67])
        # model.base_net.level4[4].spp_dw[0].conv.set_mask(masks[68])
        # model.base_net.level4[4].spp_dw[1].conv.set_mask(masks[69])
        # model.base_net.level4[4].spp_dw[2].conv.set_mask(masks[70])
        # model.base_net.level4[4].spp_dw[3].conv.set_mask(masks[71])
        # model.base_net.level4[4].conv_1x1_exp.conv.set_mask(masks[72])
        # model.base_net.level4[5].proj_1x1.conv.set_mask(masks[73])
        # model.base_net.level4[5].spp_dw[0].conv.set_mask(masks[74])
        # model.base_net.level4[5].spp_dw[1].conv.set_mask(masks[75])
        # model.base_net.level4[5].spp_dw[2].conv.set_mask(masks[76])
        # model.base_net.level4[5].spp_dw[3].conv.set_mask(masks[77])
        # model.base_net.level4[5].conv_1x1_exp.conv.set_mask(masks[78])
        # model.base_net.level4[6].proj_1x1.conv.set_mask(masks[79])
        # model.base_net.level4[6].spp_dw[0].conv.set_mask(masks[80])
        # model.base_net.level4[6].spp_dw[1].conv.set_mask(masks[81])
        # model.base_net.level4[6].spp_dw[2].conv.set_mask(masks[82])
        # model.base_net.level4[6].spp_dw[3].conv.set_mask(masks[83])
        # model.base_net.level4[6].conv_1x1_exp.conv.set_mask(masks[84])
        # model.bu_dec_l1.stages[0].set_mask(masks[85])
        # model.bu_dec_l1.stages[1].set_mask(masks[86])
        # model.bu_dec_l1.stages[2].set_mask(masks[87])
        # model.bu_dec_l1.stages[3].set_mask(masks[88])
        # model.bu_dec_l1.stages[4].set_mask(masks[89])
        # model.bu_dec_l1.projection_layer.cbr[0].set_mask(masks[90])
        # model.bu_dec_l1.merge_layer[2].cbr[0].set_mask(masks[91])
        # model.bu_dec_l1.merge_layer[3].set_mask(masks[92])
        # model.bu_dec_l2.stages[0].set_mask(masks[93])
        # model.bu_dec_l2.stages[1].set_mask(masks[94])
        # model.bu_dec_l2.stages[2].set_mask(masks[95])
        # model.bu_dec_l2.stages[3].set_mask(masks[96])
        # model.bu_dec_l2.stages[4].set_mask(masks[97])
        # model.bu_dec_l2.projection_layer.cbr[0].set_mask(masks[98])
        # model.bu_dec_l2.merge_layer[2].cbr[0].set_mask(masks[99])
        # model.bu_dec_l2.merge_layer[3].set_mask(masks[100])
        # model.bu_dec_l3.stages[0].set_mask(masks[101])
        # model.bu_dec_l3.stages[1].set_mask(masks[102])
        # model.bu_dec_l3.stages[2].set_mask(masks[103])
        # model.bu_dec_l3.stages[3].set_mask(masks[104])
        # model.bu_dec_l3.stages[4].set_mask(masks[105])
        # model.bu_dec_l3.projection_layer.cbr[0].set_mask(masks[106])
        # model.bu_dec_l3.merge_layer[2].cbr[0].set_mask(masks[107])
        # model.bu_dec_l3.merge_layer[3].set_mask(masks[108])
        # model.bu_dec_l4.stages[0].set_mask(masks[109])
        # model.bu_dec_l4.stages[1].set_mask(masks[110])
        # model.bu_dec_l4.stages[2].set_mask(masks[111])
        # model.bu_dec_l4.stages[3].set_mask(masks[112])
        # model.bu_dec_l4.stages[4].set_mask(masks[113])
        # model.bu_dec_l4.projection_layer.cbr[0].set_mask(masks[114])
        # model.bu_dec_l4.merge_layer[2].cbr[0].set_mask(masks[115])
        # model.bu_dec_l4.merge_layer[3].set_mask(masks[116])
        # model.merge_enc_dec_l2.wt_layer[1].set_mask(masks[117])
        # model.merge_enc_dec_l2.expansion_layer.cbr[0].set_mask(masks[118])
        # model.merge_enc_dec_l3.wt_layer[1].set_mask(masks[119])
        # model.merge_enc_dec_l3.expansion_layer.cbr[0].set_mask(masks[120])
        # model.merge_enc_dec_l4.wt_layer[1].set_mask(masks[121])
        # model.merge_enc_dec_l4.expansion_layer.cbr[0].set_mask(masks[122])


def espnetv2_seg(args):
    classes = args.classes
    scale=args.s
    weights = args.weights
    dataset=args.dataset
    model = ESPNetv2Segmentation(args, classes=classes, dataset=dataset)
    if weights:
        import os
        if os.path.isfile(weights):
            num_gpus = torch.cuda.device_count()
            device = 'cuda' if num_gpus >= 1 else 'cpu'
            pretrained_dict = torch.load(weights, map_location=torch.device(device))
        else:
            print_error_message('Weight file does not exist at {}. Please check. Exiting!!'.format(weights))
            exit()
        print_info_message('Loading pretrained basenet model weights')
        basenet_dict = model.base_net.state_dict()
        model_dict = model.state_dict()
        overlap_dict = {k: v for k, v in pretrained_dict.items() if k in basenet_dict}
        if len(overlap_dict) == 0:
            print_error_message('No overlaping weights between model file and pretrained weight file. Please check')
            exit()
        print_info_message('{:.2f} % of weights copied from basenet to segnet'.format(len(overlap_dict) * 1.0/len(model_dict) * 100))
        basenet_dict.update(overlap_dict)
        model.base_net.load_state_dict(basenet_dict)
        print_info_message('Pretrained basenet model loaded!!')
    else:
        print_warning_message('Training from scratch!!')
    return model

if __name__ == "__main__":

    from utilities.utils import compute_flops, model_parameters
    import torch
    import argparse

    parser = argparse.ArgumentParser(description='Testing')
    args = parser.parse_args()

    args.classes = 21
    args.s = 2.0
    args.weights='../classification/model_zoo/espnet/espnetv2_s_2.0_imagenet_224x224.pth'
    args.dataset='pascal'

    input = torch.Tensor(1, 3, 384, 384)
    model = espnetv2_seg(args)
    from utilities.utils import compute_flops, model_parameters
    print_info_message(compute_flops(model, input=input))
    print_info_message(model_parameters(model))
    out = model(input)
    print_info_message(out.size())
