import torch
from tqdm import tqdm
import random
from minigpt_utils import prompt_wrapper, generator
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import seaborn as sns

def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device)
    return (images - mean[None, :, None, None]) / std[None, :, None, None]

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device)
    return images * std[None, :, None, None] + mean[None, :, None, None]

class Attacker:
    def __init__(self, args, model, targets, device='cuda:0', is_rtp=False):
        self.args = args
        self.model = model
        self.device = device
        self.is_rtp = is_rtp
        self.targets = targets
        self.loss_buffer = []

        self.model.eval()
        self.model.requires_grad_(False)

    def attack_unconstrained(self, text_prompt, img, batch_size=8, num_iter=2000, alpha=1/255):
        print('>>> batch_size:', batch_size)
        my_generator = generator.Generator(model=self.model)

        adv_noise = torch.rand_like(img).to(self.device)
        adv_noise.requires_grad_(True)

        for t in tqdm(range(num_iter + 1)):
            batch_targets = random.sample(self.targets, batch_size)
            text_prompts = [text_prompt] * batch_size

            x_adv = normalize(adv_noise)
            if x_adv.dim() == 3:
                x_adv = x_adv.unsqueeze(0)

            x_adv_batch = x_adv.expand(batch_size, -1, -1, -1).contiguous()
            img_prompt_batch = [[x_adv_batch[i]] for i in range(batch_size)]

            prompt = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, img_prompts=img_prompt_batch)
            prompt.update_context_embs()

            target_loss = self.attack_loss(prompt, batch_targets)
            target_loss.backward()

            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(0, 1)
            adv_noise.grad.zero_()
            self.model.zero_grad()

            self.loss_buffer.append(target_loss.item())
            print("target_loss: %f" % target_loss.item())

            if t % 20 == 0:
                self.plot_loss()

            if t % 100 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv_vis = normalize(adv_noise)
                if x_adv_vis.dim() == 3:
                    x_adv_vis = x_adv_vis.unsqueeze(0)

                x_adv_vis_batch = x_adv_vis.expand(batch_size, -1, -1, -1).contiguous()
                prompt_vis = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, img_prompts=[[x_adv_vis_batch[i]] for i in range(batch_size)])
                prompt_vis.update_context_embs()
                with torch.no_grad():
                    response, _ = my_generator.generate(prompt_vis)
                print('>>>', response)

                adv_img_prompt = denormalize(x_adv_vis).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))

        return adv_img_prompt

    def attack_constrained(self, text_prompt, img, batch_size=8, num_iter=2000, alpha=1/255, epsilon=128/255):
        print('>>> batch_size:', batch_size)
        my_generator = generator.Generator(model=self.model)

        adv_noise = torch.rand_like(img).to(self.device) * 2 * epsilon - epsilon
        x = denormalize(img).clone().to(self.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
        adv_noise.requires_grad_(True)

        for t in tqdm(range(num_iter + 1)):
            batch_targets = random.sample(self.targets, batch_size)
            text_prompts = [text_prompt] * batch_size

            x_adv = x + adv_noise
            x_adv = normalize(x_adv)
            if x_adv.dim() == 3:
                x_adv = x_adv.unsqueeze(0)

            x_adv_batch = x_adv.expand(batch_size, -1, -1, -1).contiguous()
            img_prompt_batch = [[x_adv_batch[i]] for i in range(batch_size)]

            prompt = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, img_prompts=img_prompt_batch)
            prompt.update_context_embs()

            target_loss = self.attack_loss(prompt, batch_targets)
            target_loss.backward()

            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-epsilon, epsilon)
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
            adv_noise.grad.zero_()
            self.model.zero_grad()

            self.loss_buffer.append(target_loss.item())
            print("target_loss: %f" % target_loss.item())

            if t % 20 == 0:
                self.plot_loss()

            if t % 100 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv_vis = x + adv_noise
                x_adv_vis = normalize(x_adv_vis)
                if x_adv_vis.dim() == 3:
                    x_adv_vis = x_adv_vis.unsqueeze(0)

                x_adv_vis_batch = x_adv_vis.expand(batch_size, -1, -1, -1).contiguous()
                prompt_vis = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, img_prompts=[[x_adv_vis_batch[i]] for i in range(batch_size)])
                prompt_vis.update_context_embs()
                with torch.no_grad():
                    response, _ = my_generator.generate(prompt_vis)
                print('>>>', response)

                adv_img_prompt = denormalize(x_adv_vis).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))

        return adv_img_prompt

    def plot_loss(self):
        sns.set_theme()
        x_ticks = list(range(len(self.loss_buffer)))
        plt.plot(x_ticks, self.loss_buffer, label='Target Loss')
        plt.title('Loss Plot')
        plt.xlabel('Iters')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.savefig('%s/loss_curve.png' % (self.args.save_dir))
        plt.clf()
        torch.save(self.loss_buffer, '%s/loss' % (self.args.save_dir))

    def attack_loss(self, prompts, targets):
        context_embs = prompts.context_embs
        if len(context_embs) == 1:
            context_embs = context_embs * len(targets)
        assert len(context_embs) == len(targets)

        batch_size = len(targets)
        self.model.llama_tokenizer.padding_side = "right"

        to_regress_tokens = self.model.llama_tokenizer(
            targets, return_tensors="pt", padding="longest", truncation=True,
            max_length=self.model.max_txt_len, add_special_tokens=False).to(self.device)
        to_regress_embs = self.model.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

        bos = torch.ones([1, 1], dtype=to_regress_tokens.input_ids.dtype, device=to_regress_tokens.input_ids.device) * self.model.llama_tokenizer.bos_token_id
        bos_embs = self.model.llama_model.model.embed_tokens(bos)
        pad = torch.ones([1, 1], dtype=to_regress_tokens.input_ids.dtype, device=to_regress_tokens.input_ids.device) * self.model.llama_tokenizer.pad_token_id
        pad_embs = self.model.llama_model.model.embed_tokens(pad)

        T = to_regress_tokens.input_ids.masked_fill(to_regress_tokens.input_ids == self.model.llama_tokenizer.pad_token_id, -100)
        pos_padding = torch.argmin(T, dim=1)

        input_embs, targets_mask, target_tokens_length, context_tokens_length, seq_tokens_length = [], [], [], [], []
        for i in range(batch_size):
            pos = int(pos_padding[i])
            target_length = pos if T[i][pos] == -100 else T.shape[1]
            targets_mask.append(T[i:i+1, :target_length])
            input_embs.append(to_regress_embs[i:i+1, :target_length])

            context_length = context_embs[i].shape[1]
            seq_length = target_length + context_length
            target_tokens_length.append(target_length)
            context_tokens_length.append(context_length)
            seq_tokens_length.append(seq_length)

        max_length = max(seq_tokens_length)
        attention_mask = []

        for i in range(batch_size):
            context_mask = torch.ones([1, context_tokens_length[i] + 1], dtype=torch.long).to(self.device).fill_(-100)
            num_to_pad = max_length - seq_tokens_length[i]
            padding_mask = torch.ones([1, num_to_pad], dtype=torch.long).to(self.device).fill_(-100)
            targets_mask[i] = torch.cat([context_mask, targets_mask[i], padding_mask], dim=1)
            input_embs[i] = torch.cat([bos_embs, context_embs[i], input_embs[i], pad_embs.repeat(1, num_to_pad, 1)], dim=1)
            attention_mask.append(torch.LongTensor([[1] * (1 + seq_tokens_length[i]) + [0] * num_to_pad]))

        targets = torch.cat(targets_mask, dim=0).to(self.device)
        inputs_embs = torch.cat(input_embs, dim=0).to(self.device)
        attention_mask = torch.cat(attention_mask, dim=0).to(self.device)

        outputs = self.model.llama_model(
            inputs_embeds=inputs_embs,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )

        return outputs.loss
