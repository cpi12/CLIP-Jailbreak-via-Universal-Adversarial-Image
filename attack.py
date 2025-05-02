#!/usr/bin/env python3
"""
clip_simple_pgd.py

A minimal, self-contained script that:
1) Loads CLIP (openai/clip-vit-base-patch32).
2) Reads one image and two text prompts (original & target).
3) Crafts an L∞-bounded PGD adversarial image to *flip* CLIP’s preference
   from the original to the target prompt via a 2-way CE loss.
4) Prints before/after prompt probabilities and saves the adversarial image.
"""

import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer

# reproducibility
torch.manual_seed(0)

# CLIP’s preprocessing statistics
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def pgd_attack( model, orig, orig_emb, target_emb, eps, alpha, steps, norm_fn, device ):
    """
    PGD to *flip* CLIP’s preference by optimizing CE over [sim(orig), sim(target)].
    """
    adv = orig.clone().detach().requires_grad_(True)
    for _ in range(steps):
        # 1) embed adv
        x_norm = norm_fn(adv.squeeze(0)).unsqueeze(0)            # [1,3,224,224]
        img_emb = model.get_image_features(pixel_values=x_norm)  # [1,D]
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        # 2) build 2-way logits & CE loss
        sim_orig   = (img_emb * orig_emb).sum(dim=-1)   # [1]
        sim_target = (img_emb * target_emb).sum(dim=-1) # [1]
        logits     = torch.cat([sim_orig, sim_target], dim=0).unsqueeze(0)  # [1,2]
        labels     = torch.tensor([1], device=device)  # index 1 = target
        loss       = F.cross_entropy(logits, labels)

        # 3) step
        loss.backward()
        with torch.no_grad():
            adv.data += alpha * adv.grad.sign()
            adv.data  = torch.max(torch.min(adv, orig + eps), orig - eps)
            adv.data.clamp_(0,1)
        adv.grad.zero_()

    return adv.detach()

def main():
    parser = argparse.ArgumentParser(
        description="PGD attack on CLIP zero-shot prompts with CE loss"
    )
    parser.add_argument("--image",        required=True,
                        help="Path to input image")
    parser.add_argument("--orig_prompt",  required=True,
                        help="Original (benign) text prompt")
    parser.add_argument("--target_prompt",required=True,
                        help="Malicious target text prompt")
    parser.add_argument("--eps",    type=float, default=0.03,
                        help="L∞ perturbation bound")
    parser.add_argument("--alpha",  type=float, default=0.005,
                        help="PGD step size")
    parser.add_argument("--steps",  type=int,   default=20,
                        help="Number of PGD iterations")
    parser.add_argument("--out",    default="adv.png",
                        help="Filename to save adversarial image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load CLIP
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")\
                      .to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # 2) Preprocess definitions
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    normalize = transforms.Normalize(_CLIP_MEAN, _CLIP_STD)
    to_pil    = transforms.ToPILImage()

    # 3) Load & preprocess image
    pil  = Image.open(args.image).convert("RGB")
    orig = preprocess(pil).unsqueeze(0).to(device)

    # 4) Embed prompts
    texts = [args.orig_prompt, args.target_prompt]
    txt_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        txt_embs = model.get_text_features(**txt_inputs)
        txt_embs = txt_embs / txt_embs.norm(dim=-1, keepdim=True)
    orig_emb, target_emb = txt_embs[0:1], txt_embs[1:2]

    # 5) Before-attack probabilities
    with torch.no_grad():
        x_norm = normalize(orig.squeeze(0)).unsqueeze(0)
        img_emb = model.get_image_features(pixel_values=x_norm)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sims = torch.cat([
            (img_emb*orig_emb).sum(dim=-1),
            (img_emb*target_emb).sum(dim=-1)
        ], dim=0)
        probs0 = F.softmax(sims, dim=0).cpu().tolist()
    print("Before attack:")
    print(f"  {args.orig_prompt}:   {probs0[0]:.4f}")
    print(f"  {args.target_prompt}: {probs0[1]:.4f}")

    # 6) Run PGD attack with CE loss
    adv = pgd_attack(
        model, orig, orig_emb, target_emb,
        eps=args.eps, alpha=args.alpha,
        steps=args.steps, norm_fn=normalize,
        device=device
    )

    # 7) After-attack probabilities
    with torch.no_grad():
        x_norm = normalize(adv.squeeze(0)).unsqueeze(0)
        img_emb = model.get_image_features(pixel_values=x_norm)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sims2 = torch.cat([
            (img_emb*orig_emb).sum(dim=-1),
            (img_emb*target_emb).sum(dim=-1)
        ], dim=0)
        probs1 = F.softmax(sims2, dim=0).cpu().tolist()
    print("\nAfter attack:")
    print(f"  {args.orig_prompt}:   {probs1[0]:.4f}")
    print(f"  {args.target_prompt}: {probs1[1]:.4f}")

    # 8) Save adversarial image
    to_pil(adv.squeeze(0).cpu()).save(args.out)
    print(f"\nAdversarial image saved to: {args.out}")

if __name__ == "__main__":
    main()
