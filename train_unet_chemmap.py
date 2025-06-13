from pathlib import Path

import torch

from src.callbacks.early_stopping import EarlyStopping
from src.callbacks.model_checkpoint import ModelCheckpoint
from src.dataloaders.preprocessed_image_dataloader import (
    get_preprocessed_image_dataloader,
)
from src.losses.total_loss import TotalLoss, get_total_loss
from src.models.compound_model import CompoundModel
from src.models.conv3don2d import Conv3Don2D
from src.models.unet import UNet
from src.optimizers.optimizers import Adam
from src.pipeline.pipeline import Pipeline
from src.schedulers.learning_rate_schedulers import reduce_lr_on_plateau
from src.transforms.transforms import get_random_flip_transform

if __name__ == "__main__":
    batch_size = 1
    max_epochs = 250
    burn_in_epochs = 30
    reduce_lr_patience = 10
    early_stop_patience = 30
    init_lr = 1e-3
    lr_multiplier = 0.1
    min_lr = 1e-7
    smoothness_lambd = 2e1
    l2_lambd = 1e-3
    out_of_bounds_lambd = 1e-3
    mse_lambd = 1
    csv_path = Path("") # Path to the CSV file containing metadata
    reference_values = ["Fat_chem"]
    metric_mode = "min"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_sets = [0]
    cv_sets = [1, 2, 3, 4, 5]
    base_path = Path("") # Base path to the dataset, should contain 'images', 'quant_biases', 'quant_scales', and 'masks' directories

    input_channels = 124 // 2
    images_path = base_path / "images"
    quant_biases_path = base_path / "quant_biases"
    quant_scales_path = base_path / "quant_scales"
    masks_path = base_path / "masks"
    modality = "reflectance"
    out_channels = 1

    conv3d_filters = 1
    conv3d_kernel_size = (7, 2, 2)
    unet_input_channels = (
        input_channels - (conv3d_kernel_size[0] - 1)
    ) * conv3d_filters

    parent_save_dir = Path(
        f"./results/unet/{modality}/conv3d_{"_".join(map(str, conv3d_kernel_size))}/{"_".join(reference_values)}"
    )

    for val_split in cv_sets:
        train_sets = [i for i in cv_sets if i != val_split]
        val_sets = [val_split]
        split_save_dir = parent_save_dir / f"{val_split}"
        split_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Reference values: {reference_values}")

        compression_model = Conv3Don2D(
            in_channels=input_channels,
            num_filters=conv3d_filters,
            kernel_size=conv3d_kernel_size,
            normalization=None,
        )
        unet = UNet(
            in_channels=unet_input_channels,
            out_channels=out_channels,
            pad=False,
            bilinear=True,
            normalization=None,
        )

        model = CompoundModel(compression_model, unet)

        optimizer = Adam(model, lr=init_lr)
        scheduler = reduce_lr_on_plateau(
            optimizer,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            lr_multiplier=lr_multiplier,
        )
        early_stopping = EarlyStopping(patience=early_stop_patience, mode=metric_mode)
        model_checkpoint = ModelCheckpoint(
            model, optimizer, mode=metric_mode, save_dir=split_save_dir
        )

        eval_transform = None
        train_transform = get_random_flip_transform()

        train_dataloader = get_preprocessed_image_dataloader(
            reference_values=reference_values,
            images_path=images_path,
            quant_biases_path=quant_biases_path,
            quant_scales_path=quant_scales_path,
            masks_path=masks_path,
            csv_path=csv_path,
            split=train_sets,
            transform=train_transform,
            batch_size=batch_size,
            shuffle=True,
        )

        val_dataloader = get_preprocessed_image_dataloader(
            reference_values=reference_values,
            images_path=images_path,
            quant_biases_path=quant_biases_path,
            quant_scales_path=quant_scales_path,
            masks_path=masks_path,
            csv_path=csv_path,
            split=val_sets,
            transform=eval_transform,
            batch_size=batch_size,
            shuffle=False,
        )

        test_dataloader = get_preprocessed_image_dataloader(
            reference_values=reference_values,
            images_path=images_path,
            quant_biases_path=quant_biases_path,
            quant_scales_path=quant_scales_path,
            masks_path=masks_path,
            csv_path=csv_path,
            split=test_sets,
            transform=eval_transform,
            batch_size=batch_size,
            shuffle=False,
        )

        train_eval_dataloader = get_preprocessed_image_dataloader(
            reference_values=reference_values,
            images_path=images_path,
            quant_biases_path=quant_biases_path,
            quant_scales_path=quant_scales_path,
            masks_path=masks_path,
            csv_path=csv_path,
            split=train_sets,
            transform=eval_transform,
            batch_size=batch_size,
            shuffle=False,
        )

        avg_batch_size = train_dataloader.dataset.__len__() / torch.ceil(
            torch.tensor(train_dataloader.dataset.__len__() / batch_size),
        )

        loss_fn = TotalLoss(
            device=device,
            average_batch_size=avg_batch_size,
            out_of_bounds_lambd=out_of_bounds_lambd,
            smoothness_lambd=smoothness_lambd,
            l2_lambd=l2_lambd,
            mse_lambd=mse_lambd,
            return_individual_losses=False,
        )

        metric_fn = get_total_loss(
            device=device,
            average_batch_size=None,
            out_of_bounds_lambd=out_of_bounds_lambd,
            smoothness_lambd=smoothness_lambd,
            l2_lambd=l2_lambd,
            mse_lambd=mse_lambd,
            return_individual_losses=True,
        )

        metric_decision_idx = -1

        pipeline = Pipeline(
            model=model,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            test_loader=test_dataloader,
            train_eval_loader=train_eval_dataloader,
            loss_fn=loss_fn,
            metric_fn=metric_fn,
            metric_decision_idx=metric_decision_idx,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            model_checkpoint=model_checkpoint,
            num_burn_in_epochs=burn_in_epochs,
            save_dir=split_save_dir,
            device=device,
        )

        pipeline.train(max_epochs=max_epochs)
