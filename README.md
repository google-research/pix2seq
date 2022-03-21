# Pix2Seq - A general framework for turning RGB pixels into semantically meaningful sequences

![pix2seq](pix2seq.png)


## Objects365 object detection pretrained checkpoints

Backbone       | Total params (M) | Image size | Google cloud storage location
-------------: | ---------------: | ---------: | -----------:
ResNet-50      | 36.6             | 640x640    | [gs://pix2seq/obj365_pretrain/resnet_640x640_b256_s400k](https://console.cloud.google.com/storage/browser/pix2seq/obj365_pretrain/resnet_640x640_b256_s400k)
ResNet-50 (C4) | 84.7             | 640x640    | [gs://pix2seq/obj365_pretrain/resnetc_640x640_b256_s400k](https://console.cloud.google.com/storage/browser/pix2seq/obj365_pretrain/resnetc_640x640_b256_s400k)
ViT-L          | 115.2            | 640x640    | [gs://pix2seq/obj365_pretrain/vit_b_640x640_b256_s400k](https://console.cloud.google.com/storage/browser/pix2seq/obj365_pretrain/vit_b_640x640_b256_s400k)
ViT-B          | 341.2            | 640x640    | [gs://pix2seq/obj365_pretrain/vit_l_640x640_b256_s400k](https://console.cloud.google.com/storage/browser/pix2seq/obj365_pretrain/vit_l_640x640_b256_s400k)


## COCO object detection fine-tuned checkpoints

Backbone       | Total params (M) | Image size | COCO AP   | Google cloud storage location
-------------: | ---------------: | ---------: | --------: | -----------:
ResNet-50      | 36.6             | 640x640    | 39.1      | [gs://pix2seq/coco_det_finetune/resnet_640x640](https://console.cloud.google.com/storage/browser/pix2seq/coco_det_finetune/resnet_640x640)
ResNet-50      | 36.6             | 1024x1024  | 41.7      | [gs://pix2seq/coco_det_finetune/resnet_1024x1024](https://console.cloud.google.com/storage/browser/pix2seq/coco_det_finetune/resnet_1024x1024)
ResNet-50      | 36.6             | 1333x1333  | 42.6      | [gs://pix2seq/coco_det_finetune/resnet_1333x1333](https://console.cloud.google.com/storage/browser/pix2seq/coco_det_finetune/resnet_1333x1333)
ResNet-50 (C4) | 84.7             | 640x640    | 44.7      | [gs://pix2seq/coco_det_finetune/resnetc_640x640](https://console.cloud.google.com/storage/browser/pix2seq/coco_det_finetune/resnetc_640x640)
ResNet-50 (C4) | 84.7             | 1024x1024  | 46.9      | [gs://pix2seq/coco_det_finetune/resnetc_1024x1024](https://console.cloud.google.com/storage/browser/pix2seq/coco_det_finetune/resnetc_1024x1024)
ResNet-50 (C4) | 84.7             | 1333x1333  | 47.3      | [gs://pix2seq/coco_det_finetune/resnetc_1333x1333](https://console.cloud.google.com/storage/browser/pix2seq/coco_det_finetune/resnetc_1333x1333)
ViT-B          | 115.2            | 640x640    | 44.2      | [gs://pix2seq/coco_det_finetune/vit_b_640x640](https://console.cloud.google.com/storage/browser/pix2seq/coco_det_finetune/vit_b_640x640)
ViT-B          | 115.2            | 1024x1024  | 46.5      | [gs://pix2seq/coco_det_finetune/vit_b_1024x1024](https://console.cloud.google.com/storage/browser/pix2seq/coco_det_finetune/vit_b_1024x1024)
ViT-B          | 115.2            | 1333x1333  | 47.1      | [gs://pix2seq/coco_det_finetune/vit_b_1333x1333](https://console.cloud.google.com/storage/browser/pix2seq/coco_det_finetune/vit_b_1333x1333)
ViT-L          | 341.2            | 640x640    | 47.6      | [gs://pix2seq/coco_det_finetune/vit_l_640x640](https://console.cloud.google.com/storage/browser/pix2seq/coco_det_finetune/vit_l_640x640)
ViT-L          | 341.2            | 1024x1024  | 49.2      | [gs://pix2seq/coco_det_finetune/vit_l_1024x1024](https://console.cloud.google.com/storage/browser/pix2seq/coco_det_finetune/vit_l_1024x1024)
ViT-L          | 341.2            | 1333x1333  | 50.0      | [gs://pix2seq/coco_det_finetune/vit_l_1333x1333](https://console.cloud.google.com/storage/browser/pix2seq/coco_det_finetune/vit_l_1333x1333)


## Cite

[Pix2seq paper](https://arxiv.org/abs/2109.10852):

```
@article{chen2021pix2seq,
  title={Pix2seq: A language modeling framework for object detection},
  author={Chen, Ting and Saxena, Saurabh and Li, Lala and Fleet, David J and Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2109.10852},
  year={2021}
}
```


## Disclaimer
This is not an officially supported Google product.
