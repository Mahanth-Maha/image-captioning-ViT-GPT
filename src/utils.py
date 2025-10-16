from dotenv import load_dotenv
load_dotenv()

import os
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import constants  as cnst
from datasets.data import COCODataset
from datasets.tokenizer import myTokenizer_faster


root_coco = os.getenv('DATA_DIR')
train_coco = os.getenv('TRAIN_DIR')
val_coco = os.getenv('VAL_DIR')
test_coco = os.getenv('TEST_DIR')
tokenizer_models_dir = os.getenv('TOK_DIR')

train_folder = os.path.join(root_coco, f"{train_coco}")
val_folder = os.path.join(root_coco, f"{val_coco}")
test_folder = os.path.join(root_coco, f"{test_coco}")
train_ann_file = os.path.join(root_coco, "annotations", f"captions_{train_coco}.json")
val_ann_file = os.path.join(root_coco, "annotations", f"captions_{val_coco}.json")
test_ann_file = os.path.join(root_coco, "annotations", f"captions_{test_coco}.json")

coco_dataset = {
    'train' : {'folder': train_folder, 'annotations':train_ann_file},
    'val' : {'folder': val_folder, 'annotations':val_ann_file},
    'test' : {'folder': test_folder, 'annotations':test_ann_file},
}

def get_tokenizer(vocab_size , model = 'bpe'):
    # models : 'bpe', 'books_bpe', 'caps_bpe'
    tokenizer = myTokenizer_faster(
        token_model_file=f'{tokenizer_models_dir}/{model}{vocab_size}.model',
        special_tokens={
            cnst.TOKEN_UNK: vocab_size - 5,
            cnst.TOKEN_SOS: vocab_size - 4,
            cnst.TOKEN_EOS: vocab_size - 3,
            cnst.TOKEN_PAD: vocab_size - 2,
            cnst.TOKEN_END: vocab_size - 1
        },
        protected_words=[]
    )
    
    tokenizer.load()
    
    return tokenizer


def get_coco_dataloader(tokenizer, batch_size = 8, max_seq_len = 128, resize_to = 224, datatype='train',num_workers=4):
    dataset = COCODataset(
        # root= train_folder if datatype == 'train' else val_folder,
        # ann_file=train_ann_file if datatype == 'train' else val_ann_file,
        root = coco_dataset[datatype]['folder'],
        ann_file = coco_dataset[datatype]['annotations'],
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        transform=None,
        default_resize = resize_to, 
        default_stats = 'coco',
    )

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=datatype == 'train', 
        num_workers=num_workers,
    )

    return loader


def get_coco_dataloader_ddp(
    tokenizer,
    batch_size=8,
    max_seq_len=128,
    resize_to=224,
    datatype='train',
    num_workers=4,
    is_distributed=None
):
    dataset = COCODataset(
        # root=train_folder if datatype == 'train' else val_folder,
        # ann_file=train_ann_file if datatype == 'train' else val_ann_file,
        root = coco_dataset[datatype]['folder'],
        ann_file = coco_dataset[datatype]['annotations'],
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        transform=None,
        default_resize=resize_to,
        default_stats='coco',
    )

    if is_distributed is None:
        is_distributed = dist.is_available() and dist.is_initialized()

    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=(datatype == 'train'))
        shuffle = False
    else:
        sampler = None
        shuffle = (datatype == 'train')

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        # drop_last=False,
        drop_last=True,
        persistent_workers=True,
    )

    return loader, sampler