import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer
from tqdm import tqdm


_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def universal_pgd_attack(model, orig, pos_embs, neg_embs, eps, alpha, steps, norm_fn, device):
    adv = orig.clone().detach().requires_grad_(True)

    for _ in tqdm(range(steps)):
        x_norm = norm_fn(adv.squeeze(0)).unsqueeze(0)
        img_emb = model.get_image_features(pixel_values=x_norm)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)  # [1, D]

        # Cosine similarities: [B, N]
        sim_pos = (img_emb @ pos_embs.T).squeeze(0)
        sim_neg = (img_emb @ neg_embs.T).squeeze(0)

        # Maximize log-softmax over target prompts, minimize for benign
        loss_pos = -F.log_softmax(sim_pos, dim=0).mean()
        loss_neg = F.log_softmax(sim_neg, dim=0).mean()
        loss = loss_pos + loss_neg

        loss.backward()
        with torch.no_grad():
            adv.data += alpha * adv.grad.sign()
            adv.data = torch.max(torch.min(adv, orig + eps), orig - eps)
            adv.data.clamp_(0, 1)
        adv.grad.zero_()

    return adv.detach()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to initial benign image")
    parser.add_argument("--benign_prompts", required=True, help="Text file with 5–10 benign prompts")
    parser.add_argument("--target_prompts", required=True, help="Text file with 5–10 malicious prompts")
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--out", default="adv_universal.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    normalize = transforms.Normalize(_CLIP_MEAN, _CLIP_STD)
    to_pil = transforms.ToPILImage()

    # Image
    pil = Image.open(args.image).convert("RGB")
    orig = preprocess(pil).unsqueeze(0).to(device)

    # Prompts
    with open(args.benign_prompts) as f:
        benign_prompts = [line.strip() for line in f if line.strip()]
    with open(args.target_prompts) as f:
        target_prompts = [line.strip() for line in f if line.strip()]

    all_prompts = benign_prompts + target_prompts
    txt_inputs = tokenizer(all_prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        txt_embs = model.get_text_features(**txt_inputs)
        txt_embs = txt_embs / txt_embs.norm(dim=-1, keepdim=True)

    neg_embs = txt_embs[:len(benign_prompts)]  # benign
    pos_embs = txt_embs[len(benign_prompts):]  # harmful

    # PGD attack
    adv = universal_pgd_attack(
        model, orig, pos_embs, neg_embs,
        eps=args.eps, alpha=args.alpha,
        steps=args.steps, norm_fn=normalize,
        device=device
    )

    # Save result
    to_pil(adv.squeeze(0).cpu()).save(args.out)
    print(f"✅ Adversarial image saved to: {args.out}")

if __name__ == "__main__":
    main()
