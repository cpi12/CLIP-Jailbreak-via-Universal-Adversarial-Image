#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

def load_model(device):
    # Load ResNet-18 pretrained on ImageNet
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights).to(device).eval()
    preprocess = weights.transforms()
    return model, preprocess

def predict(model, img_tensor):
    # img_tensor: [1,3,H,W] already preprocessed
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        top1_prob, top1_idx = probs.max(1)
    return int(top1_idx), float(top1_prob)

def fgsm_attack(model, img, label, eps):
    # img: [1,3,H,W] normalized in [0,1]
    img_adv = img.clone().detach().requires_grad_(True)
    logits = model(img_adv)
    loss = F.cross_entropy(logits, label)
    loss.backward()
    # FGSM step: ascend loss
    img_adv = img_adv + eps * img_adv.grad.sign()
    # clip to valid range [0,1]
    img_adv = torch.clamp(img_adv, 0, 1).detach()
    return img_adv

def main():
    p = argparse.ArgumentParser(
        description="Simple FGSM attack on ResNet-18 ImageNet classifier"
    )
    p.add_argument("--image", required=True, help="Path to input image (JPEG/PNG)")
    p.add_argument("--eps", type=float, default=0.03,
                   help="FGSM perturbation magnitude (default: 0.03)")
    p.add_argument("--out", default="adv.png",
                   help="Filename to save adversarial image (default: adv.png)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = load_model(device)

    # Load & preprocess
    img_pil = Image.open(args.image).convert("RGB")
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    # Clean prediction
    clean_idx, clean_prob = predict(model, img_tensor)
    print(f"[CLEAN]   Predicted class idx: {clean_idx},  Prob: {clean_prob:.4f}")

    # FGSM attack
    label = torch.tensor([clean_idx], device=device)
    img_adv = fgsm_attack(model, img_tensor, label, eps=args.eps)

    # Adversarial prediction
    adv_idx, adv_prob = predict(model, img_adv)
    print(f"[ADV   ]  Predicted class idx: {adv_idx},  Prob: {adv_prob:.4f}")

    # Save adversarial image (denormalize & to PIL)
    # ResNet transforms normalize to ImageNet mean/std then tensor: we need to un-normalize.
    # But here we used weights.transforms() which already maps to [0,1], so img_adv is in [0,1].
    to_pil = transforms.ToPILImage()
    adv_pil = to_pil(img_adv.squeeze(0).cpu())
    adv_pil.save(args.out)
    print(f"[+] Adversarial image saved to: {args.out}")

if __name__ == "__main__":
    main()
