# 3DReasonKnee

A Dataset for Advancing Grounded Reasoning in Medical Vision Language Models with Baseline Models for ReasonKnee-Bench

## Overview

3DReasonKnee is the first 3D grounded reasoning dataset for medical images, specifically designed for multimodal vision-language models analyzing 3D knee MRI data. Our framework processes volumetric MRI data, extracts visual embeddings, and performs complex reasoning tasks related to knee conditions and diagnoses.

The dataset can be accessed at: [https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee](https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee)

## Dataset Details

The 3DReasonKnee dataset provides 494,000 high-quality quintuples derived from 7,970 3D knee MRI volumes. Each quintuple includes:

1. A 3D MRI volume
2. A diagnostic question targeting a specific anatomical region
3. A 3D bounding box precisely localizing the relevant anatomical structures
4. Clinician-generated diagnostic reasoning steps that explicitly detail the 3D reasoning process
5. Structured severity assessments for the relevant anatomical region

The creation and validation of 3DReasonKnee involved over 450 hours of expert clinician time for manually segmenting MRIs and generating reasoning chains, ensuring superior quality and clinical relevance.

## ReasonKnee-Bench

We establish ReasonKnee-Bench to evaluate:
- Localization accuracy
- Diagnostic accuracy

This benchmark provides novel insight into Vision Language Models' ability to perform grounding and severity assessment across diverse anatomical regions and diagnostic inquiries.

## Repository Structure

```
.
├── code/
│   ├── sft/
│   │   ├── data_preprocess.py           # Data preprocessing utilities
│   │   ├── sft_qwen25vl3b_inference.py  # Inference code for Qwen2.5-VL-3B
│   │   └── sft_qwen25vl3b.py            # Fine-tuning code for Qwen2.5-VL-3B
│   └── zero_shot/
│       ├── o1.py                        # Zero-shot evaluation with O1 model
│       ├── qwen2vl-3b.py                # Zero-shot evaluation with Qwen2-VL-3B
│       └── qwen2vl.py                   # Zero-shot evaluation with Qwen2-VL
├── data/                                # Please download dataset from huggingface and save to here
│   └── question_files/
│       ├── questions_nobbox_guided.json # Questions with guided reasoning without bboxes
│       └── questions_nobbox_zeroshot.json # Questions for zero-shot evaluation without bboxes
└── README.md                            # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/3DReasonKnee.git
cd 3DReasonKnee

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preprocessing: preprocess images to embeddings for efficient fine-tune

```bash
python code/sft/data_preprocess.py  
```

### Fine-tuning Models: training code for finetune qwen25vl-3b using 3DReasonKnee

```bash
python code/sft/sft_qwen25vl3b.py 
```

### Inference: inference code for fine-tune models 

```bash
python code/sft/sft_qwen25vl3b_inference.py  
```

### Zero-shot Evaluation

```bash
# Using Qwen2-VL-3B
python code/zero_shot/qwen2vl-3b.py --data_path /path/to/test/data --output_path /path/to/results

# Using O1 model
python code/zero_shot/o1.py --data_path /path/to/test/data --output_path /path/to/results
```

## Citation

If you use our dataset or code in your research, please cite our paper:

```bibtex
@article{3DReasonKnee2025,
  title={3DReasonKnee: A Dataset for Advancing Grounded Reasoning in Medical Vision Language Models},
  author={Sraavya Sambara, Sung Eun Kim, Xiaoman Zhang, Luyang Luo, Shreya Johri, Mohammed Baharoon, Du Hyun Ro, Pranav Rajpurkar},
  journal={arXiv preprint},
  year={2025}
}
```

## Contact

For questions or inquiries, please open an issue on this repository or contact us.