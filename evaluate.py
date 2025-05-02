import argparse
import torch
from torchvision import transforms
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer
import matplotlib.pyplot as plt

# CLIP normalization stats
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def load_prompts(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to adversarial image (e.g., adv_cat_universal.png)")
    parser.add_argument("--benign_prompts", required=True, help="Text file with benign prompts")
    parser.add_argument("--target_prompts", required=True, help="Text file with target/malicious prompts")
    parser.add_argument("--show_plot", action="store_true", help="Show matplotlib bar chart")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    normalize = transforms.Normalize(_CLIP_MEAN, _CLIP_STD)

    # Load image
    pil = Image.open(args.image).convert("RGB")
    img = preprocess(pil).unsqueeze(0).to(device)
    img_norm = normalize(img.squeeze(0)).unsqueeze(0)

    # Load prompts
    benign_prompts = load_prompts(args.benign_prompts)
    malicious_prompts = load_prompts(args.target_prompts)
    all_prompts = benign_prompts + malicious_prompts

    # Text embeddings
    txt_inputs = tokenizer(all_prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embs = model.get_text_features(**txt_inputs)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        img_emb = model.get_image_features(pixel_values=img_norm)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        sims = (img_emb @ text_embs.T).squeeze(0).tolist()

    # Print scores
    print("\n🔹 Similarity to benign prompts:")
    for i, s in enumerate(sims[:len(benign_prompts)]):
        print(f"  {benign_prompts[i]}: {s:.4f}")

    print("\n🔸 Similarity to malicious prompts:")
    for i, s in enumerate(sims[len(benign_prompts):]):
        print(f"  {malicious_prompts[i]}: {s:.4f}")

    # Plot
    if args.show_plot:
        plt.figure(figsize=(10, 4))
        labels = [f"Benign {i}" for i in range(len(benign_prompts))] + \
                 [f"Malicious {i}" for i in range(len(malicious_prompts))]
        plt.bar(labels, sims, color=['blue'] * len(benign_prompts) + ['red'] * len(malicious_prompts))
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Cosine Similarity")
        plt.title("CLIP Prompt Similarity to Adversarial Image")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
