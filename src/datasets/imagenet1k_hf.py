# HuggingFace datasets wrapper for ImageNet-1K
# Drop-in replacement for imagenet1k.py that loads from HF hub

import warnings
import torch
from datasets import load_dataset
from logging import getLogger
from PIL import Image

# Suppress Pillow EXIF warnings (common in ImageNet)
warnings.filterwarnings('ignore', message='Corrupt EXIF data', category=UserWarning)
Image.MAX_IMAGE_PIXELS = None

logger = getLogger()


class HFImageNet(torch.utils.data.Dataset):
    """Wraps HuggingFace ILSVRC/imagenet-1k as a map-style dataset."""

    def __init__(self, split='train', transform=None):
        self.transform = transform
        self.dataset = load_dataset('ILSVRC/imagenet-1k', split=split, num_proc=16)
        logger.info(f'Loaded HF ImageNet split={split}, len={len(self.dataset)}')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        img = sample['image'].convert('RGB')
        label = sample['label']
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def make_imagenet1k(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    training=True,
    drop_last=True,
    # ignored params kept for interface compat
    root_path=None,
    image_folder=None,
    copy_data=False,
    subset_file=None,
):
    split = 'train' if training else 'validation'
    dataset = HFImageNet(split=split, transform=transform)
    logger.info('ImageNet HF dataset created')

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)

    logger.info('ImageNet HF data loader created')
    return dataset, data_loader, dist_sampler
