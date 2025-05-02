# CLIP Jailbreak via Universal Adversarial Image

This project implements a **universal adversarial attack** on OpenAI’s CLIP model to flip its alignment from safe (benign) prompts to harmful (malicious) ones using a single, human-imperceptible image perturbation. The work is inspired by the research paper:

> **"Visual Adversarial Examples Jailbreak Aligned Large Language Models"**  
> [https://arxiv.org/abs/2404.05813](https://arxiv.org/abs/2404.05813)

The goal is to show that even well-aligned vision-language models like CLIP can be fooled by carefully crafted inputs without modifying model weights or prompts.


---

## 📦 Requirements

Make sure you have Python 3.7+ and install the following packages:

```bash
pip install torch torchvision transformers pillow tqdm matplotlib


## 🚀 How to Run

### 🧪 Step 1: Generate Adversarial Image

Run the following command to generate a visually imperceptible adversarial image using PGD:

```bash
python3 attack2.py \
  --image data/cat.jpg \
  --benign_prompts benign.txt \
  --target_prompts malicious.txt \
  --eps 0.10 \
  --alpha 0.02 \
  --steps 100 \
  --out outputs/adv_cat_universal.png

### 📊 Step 2: Evaluate the Attack

Run the following command to compute similarity scores and visualize the model's shifted alignment:

```bash
python3 evaluate.py \
  --image outputs/adv_cat_universal.png \
  --benign_prompts benign.txt \
  --target_prompts malicious.txt \
  --show_plot
