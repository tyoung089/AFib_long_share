# Deep Learning With Longitudinal Electrocardiograms Predicts Atrial Fibrillation Better Than a Single Electrocardiogram

This is the repository for the paper "Deep Learning With Longitudinal Electrocardiograms Predicts Atrial
Fibrillation Better Than a Single Electrocardiogram" (JACC: Advances, 2026, https://doi.org/10.1016/j.jacadv.2026.103468).
It holds the code and trained weights of the paper's main model, the fine-tuned longitudinal model with age
and sex. No patient data are included.

## Method

Each 10-second 12-lead ECG is embedded by a fine-tuned ECG-FM encoder. A Transformer reads the patient's ECG
history, up to the 20 most recent ECGs, together with the time of each ECG before the index ECG. Age and sex
are then added. The output is the risk of atrial fibrillation within 1 year after the index ECG.

## Installation

```bash
conda create -n afib_long python=3.9 -y
conda activate afib_long
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu   # or a GPU build from pytorch.org
pip install -r requirements.txt
```

`requirements.txt` installs fairseq-signals (the ECG-FM code), which needs git and a C++ compiler.

## Weights

```bash
curl -L --create-dirs -o weights/encoder_ecgfm_finetuned.pt https://github.com/tyoung089/AFib_long_share/releases/download/v1.0.0/encoder_ecgfm_finetuned.pt
curl -L --create-dirs -o weights/longitudinal_model.pt https://github.com/tyoung089/AFib_long_share/releases/download/v1.0.0/longitudinal_model.pt
```

## Usage

```bash
python scripts/predict.py --demo                                     # synthetic ECGs, checks the installation
python scripts/predict.py --table my_ecgs.csv --out predictions.csv  # your own ECGs
```

The columns of the input table are described at the top of `afib_long/data.py`. The predicted probabilities
are not recalibrated, and the high-risk threshold is not clinically validated.

## Training

```bash
# 1. Fine-tune ECG-FM with the single-ECG model
python scripts/1_finetune_encoder.py --table ecgs.csv --ecgfm mimic_iv_ecg_physionet_pretrained.pt --out-dir runs/encoder
# 2. Embed every ECG with the fine-tuned encoder
python scripts/2_extract_embeddings.py --table ecgs.csv --encoder runs/encoder/encoder.pt --out runs/embeddings.npz
# 3. Train the longitudinal model with age and sex
python scripts/3_train_longitudinal.py --table ecgs.csv --embeddings runs/embeddings.npz --out-dir runs/longitudinal
# 4. Bootstrap evaluation on the test set
python scripts/4_evaluate.py --predictions runs/longitudinal/test_predictions.csv --val-predictions runs/longitudinal/val_predictions.csv
```

Stage 1 needs the original ECG-FM checkpoint, `mimic_iv_ecg_physionet_pretrained.pt` from https://huggingface.co/wanglab/ecg-fm.

## Citation

```bibtex
@article{kwak2026longitudinal,
  author  = {Kwak, Sunwoo and Liu, Fengbei and Nizam, Nusrat B. and Okin, Peter M. and Sabuncu, Mert R.},
  title   = {Deep Learning With Longitudinal Electrocardiograms Predicts Atrial Fibrillation Better Than a Single Electrocardiogram},
  journal = {JACC: Advances},
  year    = {2026},
  volume  = {5},
  number  = {12},
  pages   = {103468},
  doi     = {10.1016/j.jacadv.2026.103468}
}
```

Please also cite ECG-FM. McKeen K, Masood S, Toma A, Rubin B, Wang B. ECG-FM: an open electrocardiogram
foundation model. JAMIA Open. 2025;8:ooaf122. doi:10.1093/jamiaopen/ooaf122
