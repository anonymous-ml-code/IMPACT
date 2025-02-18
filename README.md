# IMPACT

Official Repo of ***IMPACT: Integrating Modern Phology-Pharmacology And Chinese Traditional Medicine for Enhanced Herb Prescription***

(👉Under construction! The RoBERTa fine-tuning code is uploaded. However, the full proceses of the training codes are not perfectly ready for formal release under review stage. I will gradually update it! Please stay tuned.)


<!--<img src="./images/mainfig.png" width="70%">-->
<div align="center">
  <img src=".\images\framework7.png">
</div>
<p align="center">
 Figure1: The role of MM pathology and MM pharmacology serve as a scientific bridge to understand the TCM symptom and TCM prescription. To connect the modern disease diagnosis and the TCM prescription, our method can help practitioners without TCM backgrounds and boost TCM modernization.
</p>

## Install

1.  Clone this repository

```bash
git@github.com:anonymous-ml-code/IMPACT.git
cd IMPACT
```

2.  Install Package

```Shell
conda create -n IMPACT python=3.10 -y
conda activate IMPACT
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3.  Start training.
```Shell
\scripts\TCM_prescription.py
```


