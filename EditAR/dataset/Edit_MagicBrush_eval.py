import io
import os

from datasets import load_from_disk
import numpy as np
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset


class MagicBrush_Eval_Dataset(Dataset):
    def __init__(self,
                 args,
                 dataset_path,
                 llm_tokenizer,
                 mode='train',
                 ):

        self.args = args
        self.dataset_root = dataset_path
        self.dataset_path = load_from_disk(dataset_path)
        if hasattr(self.dataset_path, "keys"):
            split = "dev" if "dev" in self.dataset_path else next(iter(self.dataset_path.keys()))
            self.dataset_path = self.dataset_path[split]
        self.llm_tokenizer = llm_tokenizer

    def __len__(self,):
        return len(self.dataset_path)

    def _decode_image(self, image, mode='RGB'):
        if isinstance(image, PIL.Image.Image):
            return image.convert(mode)
        if isinstance(image, dict) and image.get('bytes') is not None:
            return Image.open(io.BytesIO(image['bytes'])).convert(mode)
        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(image)).convert(mode)
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert(mode)
        if isinstance(image, str):
            image_path = image
            if not os.path.isabs(image_path):
                image_path = os.path.join(self.dataset_root, image_path)
            return Image.open(image_path).convert(mode)
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _whiten_transparency(self, img: PIL.Image) -> PIL.Image:
        if img.mode == "RGB":
            return img

        vals_rgba = np.array(img.convert("RGBA"))
        if not (vals_rgba[:, :, 3] < 255).any():
            return img.convert("RGB")

        alpha = vals_rgba[:, :, 3] / 255.0
        vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[
            :, :, np.newaxis
        ] * vals_rgba[:, :, :3]
        return PIL.Image.fromarray(vals_rgb.astype("uint8"), "RGB")

    def _center_crop_resize(self, img: PIL.Image, target_image_size=512, resample=Image.BILINEAR):
        s = min(img.size)
        scale = target_image_size / s
        new_size = (round(scale * img.size[0]), round(scale * img.size[1]))
        img = img.resize(new_size, resample)

        x0 = (img.width - target_image_size) // 2
        y0 = (img.height - target_image_size) // 2
        return img.crop((x0, y0, x0 + target_image_size, y0 + target_image_size))

    def _vqgan_input_from(self, img: PIL.Image, target_image_size=512) -> torch.Tensor:
        img = self._center_crop_resize(img, target_image_size, Image.LANCZOS)

        np_img = np.array(img) / 255.0
        np_img = np_img * 2 - 1
        tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float()

        return tensor_img

    def _save_root(self):
        return getattr(self.args, "output_dir", None) or self.args.gpt_ckpt[:-3]

    def _mask_from(self, mask: PIL.Image, target_image_size=512):
        mask = self._center_crop_resize(mask, target_image_size, Image.NEAREST)
        mask = np.array(mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 127).astype(np.uint8)
        mask = mask[:, :, np.newaxis].repeat(3, axis=2)
        return torch.from_numpy(mask).to(torch.uint8)

    def __getitem__(self, index):
        data = self.dataset_path[index]

        input_img = self._decode_image(data['source_img'])
        edited_img = self._decode_image(data['target_img'])
        _input_img = input_img
        edit_txt = data['instruction']

        edit_text_tokens_and_mask = self.llm_tokenizer(
            edit_txt,
            max_length=120,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        edit_txt_token = edit_text_tokens_and_mask['input_ids']
        edit_txt_attn_mask = edit_text_tokens_and_mask['attention_mask']

        input_ids = edit_txt_token[0]
        input_ids_attn_mask = edit_txt_attn_mask[0]

        input_img = self._whiten_transparency(input_img)
        input_img = self._vqgan_input_from(input_img)
        edited_img = self._whiten_transparency(edited_img)
        edited_img = self._vqgan_input_from(edited_img)

        save_root = self._save_root()
        os.makedirs(f"{save_root}/magicbrush/input", exist_ok=True)
        os.makedirs(f"{save_root}/magicbrush/edit", exist_ok=True)
        os.makedirs(f"{save_root}/magicbrush/text", exist_ok=True)

        save_input_img = (input_img.permute(1,2,0)+1)/2 * 255.
        save_input_img = Image.fromarray(np.array(save_input_img).astype(np.uint8))
        save_input_img.save(f"{save_root}/magicbrush/input/magicbrush_{index:08d}_input.png")
        save_edited_img = (edited_img.permute(1,2,0)+1)/2 * 255.
        save_edited_img = Image.fromarray(np.array(save_edited_img).astype(np.uint8))
        save_edited_img.save(f"{save_root}/magicbrush/edit/magicbrush_{index:08d}_edit.png")
        with open(f'{save_root}/magicbrush/text/magicbrush_{index:08d}_txt.txt', "w") as file:
            file.write(f"Edited Text:\n{edit_txt}")

        result = {
                'index': index,
                'dataset': 'magicbrush',
                'mode': 1,
                'input_ids': input_ids,
                'input_ids_attn_mask': input_ids_attn_mask,
                'input_img': input_img,
                'edited_img': edited_img,
                '_input_img': np.array(_input_img),
                '_edit_txt': edit_txt,
                }

        if 'mask_img' in data and data['mask_img'] is not None:
            mask_img = self._decode_image(data['mask_img'], mode='L')
            result['_mask'] = self._mask_from(mask_img)

        return result
