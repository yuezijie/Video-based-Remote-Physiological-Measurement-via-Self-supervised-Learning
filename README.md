This is the code base for paper [Video-based Remote Physiological Measurement via Self-supervised Learning](https://arxiv.org/abs/2210.15401)

# Install and compile the prerequisites
- Python 3.8
- PyTorch >= 1.8
- NVIDIA GPU + CUDA
- Python packages: numpy,opencv-python,scipy
# Pretrained model
Please download the [pretrained model](https://drive.google.com/file/d/1AZ5YpD7sjp_mLlBgK0tgYTC21NCF9pPe/view?usp=sharing), and put it under weights/

# Main experiment

1. Modify the data path in trainlist.txt and testlist.txt to your own data path.
2. Run [python main_rppg.py].

(Some codes are still being updated)


# Citation
```
@article{yue2022video,
  title={Video-based Remote Physiological Measurement via Self-supervised Learning},
  author={Yue, Zijie and Shi, Miaojing and Ding, Shuai},
  journal={arXiv preprint arXiv:2210.15401},
  year={2022}
}
```
