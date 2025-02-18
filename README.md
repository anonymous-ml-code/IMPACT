# IMPACT

Official Repo of ***IMPACT: Integrating Modern Phology-Pharmacology And Chinese Traditional Medicine for Enhanced Herb Prescription***

(👉Under construction! The RoBERTa fine-tuning code is uploaded. However, the full proceses of the training codes are not perfectly ready for formal release under review stage. I will gradually update it! Please stay tuned.)


<!--<img src="./images/mainfig.png" width="70%">-->
<div align="center">
  <img src=".\images\mainfig.png">
</div>
<p align="center">
 Figure1: Overview of our M2PT approach. Here, visual prompts are embedded into each layer of the Visual Encoder, and textual prompts are embedded into each layer of the LLM. These prompts facilitate the extraction and alignment of features across modalities (e.g., vision, language). The cross-modality interaction between visual and textual features is enhanced through layered integration, ultimately improving the model's capability in zero-shot instruction learning tasks.
</p>

## Install

1.  Clone this repository and navigate to LLaVA folder

```bash
git@github.com:William-wAng618/M2PT.git
cd M2PT
```

2.  Install Package

```Shell
conda create -n M2PT python=3.10 -y
conda activate M2PT
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3.  Start training.

There are several parameter need to be notice in **==\scripts\TCM_prescription.py**



