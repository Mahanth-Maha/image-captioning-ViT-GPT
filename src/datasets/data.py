import torch
from torchvision import transforms
from torch.utils.data import Dataset
import json, os
from PIL import Image

# from training.utils import preprocess_captions
import training.utils as ut

image_stats = {
    'imagenet': {
        'mean': [0.485, 0.456, 0.406],
        'std':  [0.229, 0.224, 0.225],
    }, 
    'coco': {
        'mean': [0.471, 0.448, 0.408], 
        'std':  [0.234, 0.239, 0.242],
    }, 
    'general': {
        'mean': [0.5, 0.5, 0.5],
        'std':  [0.5, 0.5, 0.5],
    }
}


class COCODataset(Dataset):
    def __init__(self, root, ann_file, tokenizer, max_seq_len=128, transform=None, default_resize = 224, default_stats = 'imagenet'):
        self.root = root
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.default_resize = default_resize

        self.transform = transform or transforms.Compose([
            transforms.Resize((default_resize, default_resize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=image_stats[default_stats]['mean'], 
                std=image_stats[default_stats]['std'], 
                ),
        ])

        with open(ann_file, 'r') as f:
            data = json.load(f)

        self.images = {img['id']: img for img in data['images']}
        self.annotations = data['annotations']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_info = self.images[ann['image_id']]
        img_path = os.path.join(self.root, img_info['file_name'])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Warning] Failed to open {img_path}: {e}")
            image = Image.new("RGB", (self.default_resize, self.default_resize), (0, 0, 0))

        image = self.transform(image)
        
        caption = ut.preprocess_captions(ann['caption'])
        token_ids = self.tokenizer.encode(caption)

        sos = self.tokenizer.special_tokens[ut.TOKEN_SOS]
        eos = self.tokenizer.special_tokens[ut.TOKEN_EOS]
        pad = self.tokenizer.special_tokens[ut.TOKEN_PAD]

        token_ids = [sos] + token_ids + [eos]

        if len(token_ids) < self.max_seq_len:
            token_ids += [pad] * (self.max_seq_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_seq_len]

        token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        return image, token_ids
