# [MICCAI 2025] Sparsely Labeled fMRI Data Denoising with Meta-Learning-Based Semi-Supervised Domain Adaptation
The official implementation of *Keun-Soo Heo, Ji-Wung Han, Soyeon Bak, Minjoo Lim, Bogyeong Kang, Sang-Jun Park, Weili Lin, Han Zhang, Dinggang Shen, and Tae-Eui Kam*, “Sparsely Labeled fMRI Data Denoising with Meta-Learning-Based Semi-Supervised Domain Adaptation,” International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), Daejeon, Korea, Sep. 23-27, 2025.

- Contact : [Keun-Soo Heo](https://keunsooheo.github.io) ([gjrmstn1440@korea.ac.kr](mailto:gjrmstn1440@korea.ac.kr))

## Overview
<p align="center"><img src="https://keunsooheo.github.io/images/MICCAI2025_1.png" width="80%" /></p>


## Getting Started

### Installation 
Anaconda envs Installation
```
pip install -r requirement.txt
```

### Data Preparation
Four datasets were trained.
- [Baby Connectome Project](https://babyconnectomeproject.org/) (BCP)
- [Human Connectome Project](http://www.humanconnectome.org/) (HCP)
- [Whitehall II imaging study](http://www.psych.ox.ac.uk/research/neurobiology-of-ageing/research-projects-1/whitehall-oxford) (WhII-MB6, WhII-STD)  

Dataset Human Connectome Project (HCP) (Smith et al., 2013) and Whitehall II imaging study (WHII-MB6 and WHII-STD) available in [FSL FIX](https://www.fmrib.ox.ac.uk/datasets/FIX-training) (Salimi-Khorshidi et al., 2014).

Adjust the directory in the function "get_dataset_dir" in "util.py".

### Train & Test
- List the datasets used for training and testing. The last dataset serves as the target dataset, while the others are used as source datasets.
- Please specify the label percentage of the target dataset.

```
python main.py --dataset <datasets> --label_percentage <label_percentage>
```

*Examples*

HCP, MB6 (source) $\rightarrow$ STD (target)

100% labels of target domain 
```
python main.py --dataset "hcp mb6 std" --label_percentage 100
```
10% labels of target domain 
```
python main.py --dataset "hcp mb6 std" --label_percentage 10
```
It can be specified in 10% increments as 10, 20, …, 100.

30% labels of target domain 
```
python main.py --dataset "hcp mb6 std" --label_percentage 30
```
> We randomly assigned these values in data/metadata.csv. If a custom setting is required, manual assignment is required.
---


## Citation
If you utilize this code for your study, please cite our paper. Bibtex format is attached below:
```
@inproceedings{heo2025,
  title={Sparsely Labeled fMRI Data Denoising with Meta-Learning-Based Semi-Supervised Domain Adaptation},
  author={Keun-Soo Heo and Ji-Wung Han and Soyeon Bak and Minjoo Lim and Bogyeong Kang and Sang-Jun Park and Weili Lin and Han Zhang and Dinggang Shen and Tae-Eui Kam},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2025}
}

```

---
### Acknowledgements
This work was supported by grants from 
the Institute of Information & Communications Technology Planning & Evaluation (IITP)—specifically, the Artificial Intelligence Graduate School Program at Korea University (No. RS-2019-II190079), the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2023-00212498, RS-2024-00415812), the MSIT under the ITRC (Information Technology Research Center) support program (IITP-2025-RS-2024-00436857) supervised by the IITP, and IITP grant funded by MSIT (No. RS-2024-00457882, Artificial Intelligence Research Hub Project).
