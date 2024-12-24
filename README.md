This is the code base for paper [Facial video-based Remote Physiological Measurement via Self-supervised Learning](https://arxiv.org/abs/2210.15401) (TPAMI, 2023)

Abstract:

Facial video-based remote physiological measurement aims to estimate remote photoplethysmography (rPPG) signals from human facial videos and then measure multiple vital signs (e.g. heart rate, respiration frequency) from rPPG signals. Recent approaches achieve it by training deep neural networks, which normally require abundant facial videos and synchronously recorded photoplethysmography (PPG) signals for supervision. However, the collection of these annotated corpora is not easy in practice. In this paper, we introduce a novel frequency-inspired self-supervised framework that learns to estimate rPPG signals from facial videos without the need of ground truth PPG signals. Given a video sample, we first augment it into multiple positive/negative samples which contain similar/dissimilar signal frequencies to the original one. Specifically, positive samples are generated using spatial augmentation. Negative samples are generated via a learnable frequency augmentation module, which performs non-linear signal frequency transformation on the input without excessively changing its visual appearance. Next, we introduce a local rPPG expert aggregation module to estimate rPPG signals from augmented samples. It encodes complementary pulsation information from different face regions and aggregate them into one rPPG prediction. Finally, we propose a series of frequency-inspired losses, i.e. frequency contrastive loss, frequency ratio consistency loss, and cross-video frequency agreement loss, for the optimization of estimated rPPG signals from multiple augmented video samples and across temporally neighboring video samples. We conduct rPPG-based heart rate, heart rate variability and respiration frequency estimation on four standard benchmarks. The experimental results demonstrate that our method improves the state of the art by a large margin.

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
@article{yue2023facial,
  title={Facial video-based remote physiological measurement via self-supervised learning},
  author={Yue, Zijie and Shi, Miaojing and Ding, Shuai},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
```
