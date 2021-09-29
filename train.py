import os.path

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from model import Unet

from torch.utils.data import DataLoader
from dataset import SARDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# CONSTANTS AND HYPER-PARAMETERS
NUM_CLASSES = 2
IN_CHANNELS = 1
OUT_CHANNELS = 1

LEARNING_RATE = 1e-4
DEVICE = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
CUDA = "cuda"
BATCH_SIZE = 32
NUM_EPOCHS = 5
NUM_WORKERS = 4

IMAGE_HEIGHT = 160  # 1000 original
IMAGE_WIDTH = 160 # 1000 original
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/sentinel-1_csar-vv/VVD_1000/train"
VAL_IMG_DIR = "data/sentinel-1_csar-vv/VVD_1000/test"
MASK_DIR = "data/masks/mask_tiles_1000"

LOGS = "unet-run.log"
CHECKPOINT = "unet-checkpoint.tar"
RESULT_DIR = "saved_images"


def save_checkpoint(state, filename=CHECKPOINT):
    torch.save(state, filename)
    print("Checkpoint saved!")


def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint["model"])
    print("Checkpoint loaded!")


def get_loaders(
    train_dir,
    val_dir,
    maskdir,
    batch_size,
    train_transforms,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_dataset = SARDataset(
        image_dir=train_dir, mask_dir=maskdir, transform=train_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_dataset = SARDataset(image_dir=val_dir, mask_dir=maskdir, transform=val_transform,)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def calculate_accuracy_and_dice(loader, model, device=CUDA):
    """


    :param loader:
    :param model:
    :param device:
    :return:
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for input_data, target_data in loader:
            input_data = input_data.to(device)
            target_data = target_data.to(device).unsqueeze(1)
            prediction = model(input_data)
            prediction = torch.sigmoid(prediction)
            prediction = (prediction > (1/NUM_CLASSES)).float()
            num_correct += (prediction == target_data).sum()
            num_pixels += torch.numel(prediction)
            dice_score += (2 * (prediction * target_data).sum()) / ((prediction + target_data).sum() + 1e-8)

    accuracy = num_correct / num_pixels
    dice = dice_score / len(loader)

    log_msg = f"Accuracy: {accuracy}, Dice score: {dice}\n"
    with open(LOGS, "a") as log_file:
        log_file.write(log_msg)

    print(log_msg)

    model.train()


def save_predictions(loader, model, folder=RESULT_DIR, device=CUDA):
    model.eval()

    for idx, (input_data, target_data) in enumerate(loader):
        input_data = input_data.to(device=device)
        target_data = target_data.to(device=device).unsqueeze(1)
        with torch.no_grad():
            predictions = model(input_data)
            predictions = torch.sigmoid(predictions)
            predictions = (predictions > (1/NUM_CLASSES)).float()

        target_file_name = f"{idx}.png"
        target_file_path = os.path.join(folder, target_file_name)
        torchvision.utils.save_image(target_data, target_file_path)

        prediction_file_name = f"prediction_{idx}.png"
        prediction_file_path = os.path.join(folder, prediction_file_name)
        torchvision.utils.save_image(predictions, prediction_file_path)

    model.train()


def train_function(loader, model, optimizer, loss_function, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_function(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())


def main():

    # transformations for albumentation
    resizer = A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
    rotater = A.Rotate(limit=35, p=1.0)
    horizontal_flip = A.HorizontalFlip(p=0.5)
    vertical_flip = A.VerticalFlip(p=0.1)
    normalizer = A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0,)
    to_tensor = ToTensorV2()

    train_transforms = A.Compose(
        [
            resizer,
            rotater,
            horizontal_flip,
            vertical_flip,
            normalizer,
            to_tensor,
        ],
    )

    val_transforms = A.Compose(
        [
            resizer,
            normalizer,
            to_tensor,
        ],
    )

    model = Unet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        VAL_IMG_DIR,
        MASK_DIR,
        BATCH_SIZE,
        train_transforms,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(CHECKPOINT), model)

    calculate_accuracy_and_dice(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_function(train_loader, model, optimizer, loss_function, scaler)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        calculate_accuracy_and_dice(val_loader, model, device=DEVICE)
        save_predictions(
            val_loader, model, folder=RESULT_DIR, device=DEVICE
        )


if __name__ == "__main__":
    main()
