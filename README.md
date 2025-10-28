# FrameSkipSAM: Memory-Guided Frame Skipping for Real-Time SAM 2 Video Segmentation

This repository contains the implementation of **Memory-Guided Frame Skipping (MGFS)** for SAM 2, a novel approach to accelerate video object segmentation while maintaining high accuracy. Our work extends Meta's SAM 2 (Segment Anything Model 2) with intelligent frame-skipping strategies that significantly improve inference speed.

## Overview

**Segment Anything Model 2 (SAM 2)** is Meta AI's foundation model for promptable visual segmentation in images and videos. While SAM 2 achieves state-of-the-art accuracy, its computational demands limit real-time applications. This project introduces **Memory-Guided Frame Skipping**, which intelligently skips frames with minimal scene changes, achieving **up to 4.3× speedup** with minimal accuracy loss.

### Key Features

- **Memory-Guided Frame Skipping (MGFS)**: Intelligent frame-skipping based on temporal changes
- **Multiple Strategies**: Naive and Mask-Aware implementations with optional optical flow
- **Comprehensive Evaluation**: Tested on DAVIS 2017 dataset with J&F metrics
- **Production-Ready**: Drop-in replacement for SAM 2's video predictor
- **Configurable Thresholds**: Tunable skip thresholds to balance speed vs. accuracy

### Performance Highlights

| Method                | Threshold | FPS   | J&F Mean | Speedup   |
| --------------------- | --------- | ----- | -------- | --------- |
| Baseline              | -         | 4.1   | 0.419    | 1.0×      |
| **MGFS (Naive)**      | 0.05      | 5.23  | 0.415    | **1.28×** |
| **MGFS (Naive)**      | 0.15      | 17.81 | 0.380    | **4.34×** |
| **MGFS (Mask-Aware)** | 0.05      | 4.23  | 0.417    | **1.03×** |
| **MGFS (Mask-Aware)** | 0.10      | 6.28  | 0.388    | **1.53×** |

_Results on DAVIS 2017 validation set. See our [paper](Memory_Guided_Frame_Skipping_for_Real_Time_SAM_2_Video_Segmentation.pdf) for full details._

## Research Paper

📄 **[Memory-Guided Frame Skipping for Real-Time SAM 2 Video Segmentation](Memory_Guided_Frame_Skipping_for_Real_Time_SAM_2_Video_Segmentation.pdf)**

Our paper presents a comprehensive analysis of frame-skipping strategies for SAM 2, including:

- Theoretical framework for memory-guided segmentation
- Comparison of naive vs. mask-aware skipping approaches
- Evaluation with and without optical flow
- Speed-accuracy tradeoff analysis across multiple thresholds

## Evaluation Results

Comprehensive evaluation results and prediction masks are available at:

- **Evaluation Repository**: [https://github.com/bchou9/davis2017eval](https://github.com/bchou9/davis2017eval)
- Contains DAVIS 2017 predictions for all MGFS variants
- Includes baseline comparisons and detailed per-sequence metrics

---

## Original SAM 2 Information

**[AI at Meta, FAIR](https://ai.meta.com/research/)**

[Nikhila Ravi](https://nikhilaravi.com/), [Valentin Gabeur](https://gabeur.github.io/), [Yuan-Ting Hu](https://scholar.google.com/citations?user=E8DVVYQAAAAJ&hl=en), [Ronghang Hu](https://ronghanghu.com/), [Chaitanya Ryali](https://scholar.google.com/citations?user=4LWx24UAAAAJ&hl=en), [Tengyu Ma](https://scholar.google.com/citations?user=VeTSl0wAAAAJ&hl=en), [Haitham Khedr](https://hkhedr.com/), [Roman Rädle](https://scholar.google.de/citations?user=Tpt57v0AAAAJ&hl=en), [Chloe Rolland](https://scholar.google.com/citations?hl=fr&user=n-SnMhoAAAAJ), [Laura Gustafson](https://scholar.google.com/citations?user=c8IpF9gAAAAJ&hl=en), [Eric Mintun](https://ericmintun.github.io/), [Junting Pan](https://junting.github.io/), [Kalyan Vasudev Alwala](https://scholar.google.co.in/citations?user=m34oaWEAAAAJ&hl=en), [Nicolas Carion](https://www.nicolascarion.com/), [Chao-Yuan Wu](https://chaoyuan.org/), [Ross Girshick](https://www.rossgirshick.info/), [Piotr Dollár](https://pdollar.github.io/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/)

[[`SAM 2 Paper`](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)] [[`Project`](https://ai.meta.com/sam2)] [[`Demo`](https://sam2.metademolab.com/)] [[`Dataset`](https://ai.meta.com/datasets/segment-anything-video)] [[`Blog`](https://ai.meta.com/blog/segment-anything-2)]

![SAM 2 architecture](assets/model_diagram.png?raw=true)

![SA-V dataset](assets/sa_v_dataset.jpg?raw=true)

---

## Getting Started with Frame Skipping

### Quick Start: Memory-Guided Frame Skipping

To use MGFS with SAM 2 for video segmentation:

```python
import torch
from sam2.build_sam import build_sam2_video_predictor

# Build predictor with frame skipping
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# Configure skip threshold (default: 0.05)
# Lower = more conservative (fewer skips), Higher = more aggressive (more skips)
predictor.skip_mad_threshold = 0.10  # 10% change threshold

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video_path="<your_video_dir>")

    # Add initial prompt (e.g., first frame mask)
    predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=<your_mask>)

    # Propagate with automatic frame skipping
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        # Process results - skipped frames reuse previous predictions
        ...
```

### Running MGFS on DAVIS 2017

We provide scripts to reproduce our DAVIS 2017 evaluation:

```bash
# Run MGFS inference on all DAVIS sequences
python run_mgfs_davis.py

# Evaluate predictions against ground truth
python eval_davis_jf.py \
    --gt_root ./datasets/DAVIS/DAVIS2017/DAVIS \
    --pred1 ./predictions/DAVIS2017_baseline \
    --pred2 ./predictions/DAVIS2017
```

Or use the Jupyter notebook for interactive exploration:

```bash
jupyter notebook run_mgfs_davis.ipynb
```

### Frame Skipping Strategies

This repository implements multiple frame-skipping approaches:

1. **Naive Frame Skipping**: Skips frames based on whole-frame pixel difference

   - Fast and simple
   - Works well for static camera scenarios
   - Configure via `skip_mad_threshold`

2. **Mask-Aware Frame Skipping**: Only analyzes regions of interest

   - Focuses on object regions
   - Better for dynamic backgrounds
   - Slightly slower but more accurate

3. **Optical Flow Enhancement**: Uses dense optical flow for mask warping
   - Can improve accuracy in some scenarios
   - Higher computational cost
   - See `sam2/utils/optical_flow.py`

### Threshold Selection Guide

Choose your `skip_mad_threshold` based on your speed vs. accuracy requirements:

- **0.05** (Conservative): ~1.3× speedup, minimal accuracy loss (~0.4% J&F drop)
- **0.07** (Balanced): ~1.4× speedup, small accuracy loss (~0.7% J&F drop)
- **0.10** (Moderate): ~2.3× speedup, moderate accuracy loss (~1.8% J&F drop)
- **0.15** (Aggressive): ~4.3× speedup, larger accuracy loss (~3.9% J&F drop)

---

## Latest Updates (Original SAM 2)

**12/11/2024 -- full model compilation for a major VOS speedup and a new `SAM2VideoPredictor` to better handle multi-object tracking**

- We now support `torch.compile` of the entire SAM 2 model on videos, which can be turned on by setting `vos_optimized=True` in `build_sam2_video_predictor`, leading to a major speedup for VOS inference.
- We update the implementation of `SAM2VideoPredictor` to support independent per-object inference, allowing us to relax the assumption of prompting for multi-object tracking and adding new objects after tracking starts.
- See [`RELEASE_NOTES.md`](RELEASE_NOTES.md) for full details.

**09/30/2024 -- SAM 2.1 Developer Suite (new checkpoints, training code, web demo) is released**

- A new suite of improved model checkpoints (denoted as **SAM 2.1**) are released. See [Model Description](#model-description) for details.
  - To use the new SAM 2.1 checkpoints, you need the latest model code from this repo. If you have installed an earlier version of this repo, please first uninstall the previous version via `pip uninstall SAM-2`, pull the latest code from this repo (with `git pull`), and then reinstall the repo following [Installation](#installation) below.
- The training (and fine-tuning) code has been released. See [`training/README.md`](training/README.md) on how to get started.
- The frontend + backend code for the SAM 2 web demo has been released. See [`demo/README.md`](demo/README.md) for details.

---

## Installation

SAM 2 with Frame Skipping needs to be installed first before use. The code requires `python>=3.10`, as well as `torch>=2.5.1` and `torchvision>=0.20.1`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies.

### Install FrameSkipSAM

```bash
git clone https://github.com/bchou9/FrameSkipSAM.git && cd FrameSkipSAM

pip install -e .
```

### Additional Dependencies for Frame Skipping

The MGFS implementation requires OpenCV for change detection and optical flow:

```bash
pip install opencv-python numpy
```

If you are installing on Windows, it's strongly recommended to use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu.

To use the SAM 2 predictor and run the example notebooks, `jupyter` and `matplotlib` are required and can be installed by:

```bash
pip install -e ".[notebooks]"
```

### For DAVIS Evaluation

To reproduce our DAVIS 2017 evaluation results:

```bash
# Install evaluation dependencies
pip install tqdm imageio

# Download DAVIS 2017 dataset (follow instructions at https://davischallenge.org/)
# Place in ./datasets/DAVIS/DAVIS2017/DAVIS/
```

Note:
1. It's recommended to create a new Python environment via [Anaconda](https://www.anaconda.com/) for this installation and install PyTorch 2.5.1 (or higher) via `pip` following https://pytorch.org/. If you have a PyTorch version lower than 2.5.1 in your current environment, the installation command above will try to upgrade it to the latest PyTorch version using `pip`.
2. The step above requires compiling a custom CUDA kernel with the `nvcc` compiler. If it isn't already available on your machine, please install the [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) with a version that matches your PyTorch CUDA version.
3. If you see a message like `Failed to build the SAM 2 CUDA extension` during installation, you can ignore it and still use SAM 2 (some post-processing functionality may be limited, but it doesn't affect the results in most cases).

Please see [`INSTALL.md`](./INSTALL.md) for FAQs on potential issues and solutions.

## Getting Started

### Download Checkpoints

First, we need to download a model checkpoint. All the model checkpoints can be downloaded by running:

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

or individually from:

- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

(note that these are the improved checkpoints denoted as SAM 2.1; see [Model Description](#model-description) for details.)

Then SAM 2 can be used in a few lines as follows for image and video prediction.

### Video prediction with Frame Skipping

For promptable segmentation and tracking in videos **with intelligent frame skipping**, we provide an enhanced video predictor. The frame-skipping logic is built directly into the `propagate_in_video` method:

```python
import torch
from sam2.build_sam import build_sam2_video_predictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# Configure frame skipping threshold
predictor.skip_mad_threshold = 0.05  # Skip if <5% pixel change

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<your_video>)

    # add new prompts and instantly get the output on the same frame
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>)

    # propagate the prompts to get masklets throughout the video
    # Frame skipping happens automatically based on temporal changes
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        # Frames with minimal changes reuse previous predictions (much faster!)
        # You'll see console output: "Skipping frame X due to low MAD (Y)"
        ...
```

**How it works:** During propagation, each frame is compared to the previous frame using Mean Absolute Difference (MAD). If MAD is below `skip_mad_threshold`, the previous frame's masks are reused, skipping expensive inference. This dramatically speeds up videos with static scenes or slow camera motion.

Please refer to the examples in [video_predictor_example.ipynb](./notebooks/video_predictor_example.ipynb) for details on how to add click or box prompts, make refinements, and track multiple objects in videos.

### Image prediction

SAM 2 has all the capabilities of [SAM](https://github.com/facebookresearch/segment-anything) on static images, and we provide image prediction APIs that closely resemble SAM for image use cases. The `SAM2ImagePredictor` class has an easy interface for image prompting.

```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)
```

Please refer to the examples in [image_predictor_example.ipynb](./notebooks/image_predictor_example.ipynb) (also in Colab [here](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb)) for static image use cases.

SAM 2 also supports automatic mask generation on images just like SAM. Please see [automatic_mask_generator_example.ipynb](./notebooks/automatic_mask_generator_example.ipynb) (also in Colab [here](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb)) for automatic mask generation in images.

## Load from 🤗 Hugging Face

Alternatively, models can also be loaded from [Hugging Face](https://huggingface.co/models?search=facebook/sam2) (requires `pip install huggingface_hub`).

For image prediction:

```python
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)
```

For video prediction:

```python
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

# Enable frame skipping (MGFS extension)
predictor.skip_mad_threshold = 0.05  # Adjust threshold as needed

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<your_video>)

    # add new prompts and instantly get the output on the same frame
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>)

    # propagate the prompts to get masklets throughout the video
    # Frame skipping automatically applied
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...
```

---

---

## Implementation Details

### Core MGFS Components

The Memory-Guided Frame Skipping implementation consists of several key components:

1. **`sam2/sam2_video_predictor.py`**: Enhanced video predictor with frame-skipping logic

   - `skip_mad_threshold`: Configurable threshold parameter (default: 0.05)
   - `_mean_abs_diff()`: Computes Mean Absolute Difference between consecutive frames
   - Modified `propagate_in_video()`: Implements the frame-skipping decision logic

2. **`sam2/utils/change_detection.py`**: Temporal change detection utilities

   - Frame comparison and threshold-based skip decisions
   - Handles both PyTorch tensors and NumPy arrays

3. **`sam2/utils/optical_flow.py`**: Optical flow-based mask warping (optional)

   - Dense optical flow computation using OpenCV
   - Forward mask warping for improved accuracy

4. **`run_mgfs_davis.py`**: DAVIS 2017 evaluation script

   - Batch processing of video sequences
   - Automatic mask generation and saving

5. **`eval_davis_jf.py`**: J&F metrics evaluation
   - Region similarity (J) and contour accuracy (F) computation
   - Comparative evaluation between baseline and MGFS

### How Frame Skipping Works

During video propagation, the algorithm:

1. **Computes temporal change**: Calculates Mean Absolute Difference (MAD) between current and previous frames
2. **Makes skip decision**: If MAD < `skip_mad_threshold`, the frame is skipped
3. **Reuses predictions**: Skipped frames use cached mask outputs from the previous frame
4. **Maintains memory**: Only non-skipped frames update the memory bank

This approach achieves significant speedups by avoiding expensive transformer inference on frames with minimal scene changes, while maintaining the temporal consistency benefits of SAM 2's streaming memory architecture.

### Repository Structure

```
FrameSkipSAM/
├── sam2/                           # Core SAM 2 model with MGFS extensions
│   ├── sam2_video_predictor.py    # Enhanced video predictor with frame skipping
│   ├── utils/
│   │   ├── change_detection.py    # Temporal change detection
│   │   └── optical_flow.py        # Optical flow utilities
│   └── ...
├── run_mgfs_davis.py              # DAVIS evaluation script
├── run_mgfs_davis.ipynb           # Interactive notebook for DAVIS
├── eval_davis_jf.py               # J&F metrics evaluation
├── results.ipynb                  # Results visualization
├── convert.ipynb                  # Prediction format conversion
├── predictions/                    # Generated predictions
│   └── DAVIS2017/                 # DAVIS 2017 predictions
└── Memory_Guided_Frame_Skipping_for_Real_Time_SAM_2_Video_Segmentation.pdf
```

---

## Model Description

### SAM 2.1 checkpoints

The table below shows the improved SAM 2.1 checkpoints released on September 29, 2024.
| **Model** | **Size (M)** | **Speed (FPS)** | **SA-V test (J&F)** | **MOSE val (J&F)** | **LVOS v2 (J&F)** |
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: | :---------------: |
| sam2.1_hiera_tiny <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_t.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)) | 38.9 | 91.2 | 76.5 | 71.8 | 77.3 |
| sam2.1_hiera_small <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_s.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)) | 46 | 84.8 | 76.6 | 73.5 | 78.3 |
| sam2.1_hiera_base_plus <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_b+.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)) | 80.8 | 64.1 | 78.2 | 73.7 | 78.2 |
| sam2.1_hiera_large <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_l.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)) | 224.4 | 39.5 | 79.5 | 74.6 | 80.6 |

### SAM 2 checkpoints

The previous SAM 2 checkpoints released on July 29, 2024 can be found as follows:

|                                                                                  **Model**                                                                                   | **Size (M)** | **Speed (FPS)** | **SA-V test (J&F)** | **MOSE val (J&F)** | **LVOS v2 (J&F)** |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------: | :-------------: | :-----------------: | :----------------: | :---------------: |
|      sam2_hiera_tiny <br /> ([config](sam2/configs/sam2/sam2_hiera_t.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt))       |     38.9     |      91.5       |        75.0         |        70.9        |       75.3        |
|     sam2_hiera_small <br /> ([config](sam2/configs/sam2/sam2_hiera_s.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt))      |      46      |      85.6       |        74.9         |        71.5        |       76.4        |
| sam2_hiera_base_plus <br /> ([config](sam2/configs/sam2/sam2_hiera_b+.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt)) |     80.8     |      64.8       |        74.7         |        72.8        |       75.8        |
|     sam2_hiera_large <br /> ([config](sam2/configs/sam2/sam2_hiera_l.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt))      |    224.4     |      39.7       |        76.0         |        74.6        |       79.8        |

Speed measured on an A100 with `torch 2.5.1, cuda 12.4`. See `benchmark.py` for an example on benchmarking (compiling all the model components). Compiling only the image encoder can be more flexible and also provide (a smaller) speed-up (set `compile_image_encoder: True` in the config).

## Segment Anything Video Dataset

See [sav_dataset/README.md](sav_dataset/README.md) for details.

## Training SAM 2

You can train or fine-tune SAM 2 on custom datasets of images, videos, or both. Please check the training [README](training/README.md) on how to get started.

## Web demo for SAM 2

We have released the frontend + backend code for the SAM 2 web demo (a locally deployable version similar to https://sam2.metademolab.com/demo). Please see the web demo [README](demo/README.md) for details.

## License

The SAM 2 model checkpoints, SAM 2 demo code (front-end and back-end), and SAM 2 training code are licensed under [Apache 2.0](./LICENSE), however the [Inter Font](https://github.com/rsms/inter?tab=OFL-1.1-1-ov-file) and [Noto Color Emoji](https://github.com/googlefonts/noto-emoji) used in the SAM 2 demo code are made available under the [SIL Open Font License, version 1.1](https://openfontlicense.org/open-font-license-official-text/).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

### FrameSkipSAM (MGFS) Contributors

The Memory-Guided Frame Skipping extension was created by Henry Chou (Head Developer and Team Lead), Raymond Kang, Wei Shao, Yiqiao Lin. For questions or contributions related to the frame-skipping implementation, please open an issue or submit a pull request.

### Original SAM 2 Contributors

The SAM 2 project was made possible with the help of many contributors (alphabetical):

Karen Bergan, Daniel Bolya, Alex Bosenberg, Kai Brown, Vispi Cassod, Christopher Chedeau, Ida Cheng, Luc Dahlin, Shoubhik Debnath, Rene Martinez Doehner, Grant Gardner, Sahir Gomez, Rishi Godugu, Baishan Guo, Caleb Ho, Andrew Huang, Somya Jain, Bob Kamma, Amanda Kallet, Jake Kinney, Alexander Kirillov, Shiva Koduvayur, Devansh Kukreja, Robert Kuo, Aohan Lin, Parth Malani, Jitendra Malik, Mallika Malhotra, Miguel Martin, Alexander Miller, Sasha Mitts, William Ngan, George Orlin, Joelle Pineau, Kate Saenko, Rodrick Shepard, Azita Shokrpour, David Soofian, Jonathan Torres, Jenny Truong, Sagar Vaze, Meng Wang, Claudette Ward, Pengchuan Zhang.

Third-party code: we use a GPU-based connected component algorithm adapted from [`cc_torch`](https://github.com/zsef123/Connected_components_PyTorch) (with its license in [`LICENSE_cctorch`](./LICENSE_cctorch)) as an optional post-processing step for the mask predictions.

## Citations

### Citing FrameSkipSAM (Memory-Guided Frame Skipping)

If you use the Memory-Guided Frame Skipping implementation in your research, please cite our work:

```bibtex
@article{frameskipsam2025,
  title={Memory-Guided Frame Skipping for Real-Time SAM 2 Video Segmentation},
  author={Henry Chou, Raymond Kang, Wei Shao, Yiqiao Lin},
  year={2025},
  note={Available at: https://github.com/bchou9/FrameSkipSAM}
}
```

### Citing SAM 2

If you use SAM 2 or the SA-V dataset in your research, please use the following BibTeX entry.

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```
