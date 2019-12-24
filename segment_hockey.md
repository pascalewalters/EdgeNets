# Segment hockey lines

Everything for Newton

## Original frames as input

### Training

2 stages:

```
CUDA_VISIBLE_DEVICES=0 python train_segmentation.py --model espnetv2 --s 2.0 --dataset hockey --data-path ./vision_datasets/hockey/ --batch-size 40 --crop-size 256 256 --lr 0.009 --scheduler hybrid --clr-max 61 --epochs 100
CUDA_VISIBLE_DEVICES=0 python train_segmentation.py --model espnetv2 --s 2.0 --dataset hockey --data-path ./vision_datasets/hockey/ --batch-size 16 --crop-size 384 384 --lr 0.005 --scheduler hybrid --clr-max 61 --epochs 100 --freeze-bn --finetune ./results_segmentation/model_espnetv2_hockey/s_2.0_sch_hybrid_loss_ce_res_256_sc_0.5_2.0/20191128-202617/espnetv2_2.0_256_best.pth
```

### Evaluating

This script calculates the mIOU of the segmented line image

```
CUDA_VISIBLE_DEVICES=0 python eval_segmentation.py --model espnetv2 --s 2.0 --dataset hockey --data-path ./vision_datasets/hockey/ --split val --im-size 384 384 --weights-test ./results_segmentation/model_espnetv2_hockey/s_2.0_sch_hybrid_loss_ce_res_384_sc_0.5_2.0/20191129-092351/espnetv2_2.0_384_best.pth
```

### Testing

Do inference

```
CUDA_VISIBLE_DEVICES=0 python test_segmentation.py --model espnetv2 --s 2.0 --dataset hockey --data-path ./vision_datasets/hockey/ --split val --im-size 384 384 --weights-test ./results_segmentation/model_espnetv2_hockey/s_2.0_sch_hybrid_loss_ce_res_384_sc_0.5_2.0/20191129-092351/espnetv2_2.0_384_best.pth
```


## Segmented rink as input

### Training

2 stages:

```
CUDA_VISIBLE_DEVICES=0 python train_segmentation.py --model espnetv2 --s 2.0 --dataset hockey_rink_seg --data-path ./vision_datasets/hockey/ --batch-size 40 --crop-size 256 256 --lr 0.009 --scheduler hybrid --clr-max 61 --epochs 100
CUDA_VISIBLE_DEVICES=0 python train_segmentation.py --model espnetv2 --s 2.0 --dataset hockey_rink_seg --data-path ./vision_datasets/hockey/ --batch-size 16 --crop-size 384 384 --lr 0.005 --scheduler hybrid --clr-max 61 --epochs 100 --freeze-bn --finetune ./results_segmentation/model_espnetv2_hockey_rink_seg/s_2.0_sch_hybrid_loss_ce_res_256_sc_0.5_2.0/20191129-141955/espnetv2_2.0_256_best.pth
```

### Evaluating

This script calculates the mIOU of the segmented line image

```
CUDA_VISIBLE_DEVICES=0 python eval_segmentation.py --model espnetv2 --s 2.0 --dataset hockey_rink_seg --data-path ./vision_datasets/hockey/ --split val --im-size 384 384 --weights-test ./results_segmentation/model_espnetv2_hockey_rink_seg/s_2.0_sch_hybrid_loss_ce_res_384_sc_0.5_2.0/20191129-154203/espnetv2_2.0_384_best.pth
```

### Testing

Do inference

```
CUDA_VISIBLE_DEVICES=0 python test_segmentation.py --model espnetv2 --s 2.0 --dataset hockey_rink_seg --data-path ./vision_datasets/hockey/ --split val --im-size 384 384 --weights-test ./results_segmentation/model_espnetv2_hockey_rink_seg/s_2.0_sch_hybrid_loss_ce_res_384_sc_0.5_2.0/20191129-154203/espnetv2_2.0_384_best.pth
```
