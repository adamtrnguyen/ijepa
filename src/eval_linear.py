# Linear probe evaluation for I-JEPA
# Freezes the target encoder, average-pools patch embeddings,
# and trains a single nn.Linear(embed_dim, num_classes) on ImageNet labels.

import argparse
import logging
import os
import sys
import yaml

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.datasets.imagenet1k_hf import make_imagenet1k
from src.helper import init_model
from src.utils.logging import AverageMeter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_target_encoder(checkpoint_path, encoder, device):
    """Load target_encoder weights from checkpoint, stripping module. prefix."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint.get('epoch', '?')

    state_dict = checkpoint['target_encoder']
    # Strip DDP module. prefix
    cleaned = {}
    for k, v in state_dict.items():
        key = k.replace('module.', '', 1) if k.startswith('module.') else k
        cleaned[key] = v

    msg = encoder.load_state_dict(cleaned)
    logger.info(f'Loaded target encoder from epoch {epoch}: {msg}')
    del checkpoint
    return encoder


def make_eval_transforms(crop_size, training):
    """Simple transforms for linear probe (no heavy augmentation)."""
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if training:
        return transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(crop_size * 256 / 224)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ])


def accuracy(output, target, topk=(1, 5)):
    """Computes top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def main(args):
    # -- Parse config
    model_name = args['meta']['model_name']
    use_bfloat16 = args['meta'].get('use_bfloat16', True)
    use_fused_attn = args['meta'].get('use_fused_attention', False)
    checkpoint_path = args['meta']['checkpoint']

    batch_size = args['eval']['batch_size']
    num_workers = args['eval']['num_workers']
    num_epochs = args['eval']['epochs']
    lr = args['eval']['lr']
    momentum = args['eval']['momentum']
    weight_decay = args['eval']['weight_decay']

    crop_size = args['data']['crop_size']
    patch_size = args['mask']['patch_size']
    log_folder = args['logging']['folder']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    os.makedirs(log_folder, exist_ok=True)

    # -- Build encoder and load target encoder weights
    encoder, _ = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        model_name=model_name,
        use_fused_attn=use_fused_attn,
    )
    encoder = load_target_encoder(checkpoint_path, encoder, device)
    encoder.eval()
    encoder.requires_grad_(False)
    embed_dim = encoder.embed_dim
    logger.info(f'Encoder frozen: embed_dim={embed_dim}')

    # -- Linear head
    linear_head = nn.Linear(embed_dim, 1000).to(device)
    nn.init.trunc_normal_(linear_head.weight, std=0.01)
    nn.init.zeros_(linear_head.bias)

    # -- Optimizer and scheduler
    optimizer = torch.optim.SGD(
        linear_head.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # -- Data loaders
    train_transform = make_eval_transforms(crop_size, training=True)
    val_transform = make_eval_transforms(crop_size, training=False)

    _, train_loader, train_sampler = make_imagenet1k(
        transform=train_transform,
        batch_size=batch_size,
        num_workers=num_workers,
        training=True,
        drop_last=True,
    )
    _, val_loader, _ = make_imagenet1k(
        transform=val_transform,
        batch_size=batch_size,
        num_workers=num_workers,
        training=False,
        drop_last=False,
    )

    # -- Training loop
    best_acc1 = 0.0
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        linear_head.train()

        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f'Train {epoch+1}/{num_epochs}', dynamic_ncols=True)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Extract features
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    features = encoder(images)  # [B, N, D]
                features = features.mean(dim=1)  # [B, D]
                features = features.float()

            # Linear head forward + backward
            logits = linear_head(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc1, = accuracy(logits, labels, topk=(1,))
            loss_meter.update(loss.item(), images.size(0))
            acc1_meter.update(acc1, images.size(0))

            pbar.set_postfix(loss=f'{loss_meter.avg:.3f}', acc1=f'{acc1_meter.avg:.1f}%')

        scheduler.step()

        # -- Validation
        linear_head.eval()
        val_loss_meter = AverageMeter()
        val_acc1_meter = AverageMeter()
        val_acc5_meter = AverageMeter()

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Val {epoch+1}/{num_epochs}', dynamic_ncols=True):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    features = encoder(images)
                features = features.mean(dim=1).float()

                logits = linear_head(features)
                loss = F.cross_entropy(logits, labels)

                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
                val_loss_meter.update(loss.item(), images.size(0))
                val_acc1_meter.update(acc1, images.size(0))
                val_acc5_meter.update(acc5, images.size(0))

        is_best = val_acc1_meter.avg > best_acc1
        best_acc1 = max(val_acc1_meter.avg, best_acc1)

        logger.info(
            f'Epoch {epoch+1}/{num_epochs} — '
            f'Train loss: {loss_meter.avg:.3f}, Train acc1: {acc1_meter.avg:.1f}% | '
            f'Val loss: {val_loss_meter.avg:.3f}, Val acc1: {val_acc1_meter.avg:.1f}%, '
            f'Val acc5: {val_acc5_meter.avg:.1f}% | '
            f'Best acc1: {best_acc1:.1f}%{"*" if is_best else ""}'
        )

        # Save checkpoint
        save_dict = {
            'epoch': epoch + 1,
            'linear_head': linear_head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc1': best_acc1,
        }
        torch.save(save_dict, os.path.join(log_folder, 'linear-latest.pth.tar'))
        if is_best:
            torch.save(save_dict, os.path.join(log_folder, 'linear-best.pth.tar'))

    logger.info(f'Final best top-1 accuracy: {best_acc1:.1f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, required=True, help='path to eval config yaml')
    cli_args = parser.parse_args()

    with open(cli_args.fname, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)
