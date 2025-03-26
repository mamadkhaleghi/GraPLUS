# SimOPA Binary Classifier

This directory contains the binary classifier model used by GraPLUS to evaluate object placement plausibility. The model is part of the SimOPA framework developed for the Object Placement Assessment dataset.

## Model Details

- **File**: `best-acc.pth` (not included in the repository)
- **Source**: [Object Placement Assessment Dataset (OPA)](https://github.com/bcmi/Object-Placement-Assessment-Dataset-OPA)
- **Purpose**: Distinguishes between reasonable and unreasonable object placements

## Obtaining the Model

The pre-trained binary classifier must be downloaded separately:

1. Download from [BCMI Cloud](https://cloud.bcmi.sjtu.edu.cn/sharing/XPEgkSHdQ) or [Baidu Disk](https://pan.baidu.com/s/1skFRfLyczzXUpp-6tMHArA) (code: 0qty)
2. Place the downloaded file in this directory as `best-acc.pth`

## Dependencies

This classifier requires the Faster R-CNN model pre-trained on Visual Genome:

1. Download from [Google Drive](https://drive.google.com/file/d/18n_3V1rywgeADZ3oONO0DsuuS9eMW6sN/view)
2. Place the file in `faster-rcnn/models/faster_rcnn_res101_vg.pth`

## Citation
If you use this model in your research, please cite the original OPA dataset paper:
```bibtex
@article{liu2022opa,
  title={OPA: Object Placement Assessment Dataset},
  author={Liu, Liu and Liu, Zhenchen and Zhang, Bo and Li, Jiangtong and Niu, Li and Liu, Qingyang and Zhang, Liqing},
  journal={arXiv preprint arXiv:2107.01889},
  year={2022}
}
```
