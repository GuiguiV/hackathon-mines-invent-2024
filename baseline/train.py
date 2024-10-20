from pathlib import Path
from sklearn.metrics import jaccard_score
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.transforms import Resize, InterpolationMode


from baseline.ndvi import remove_low_ndvi_images,calculate_ndvi_means

def print_iou_per_class(
    targets: torch.Tensor,
    preds: torch.Tensor,
    nb_classes: int,
) -> None:
    """
    Compute IoU between predictions and targets, for each class.

    Args:
        targets (torch.Tensor): Ground truth of shape (B, H, W).
        preds (torch.Tensor): Model predictions of shape (B, nb_classes, H, W).
        nb_classes (int): Number of classes in the segmentation task.
    """

    # Compute IoU for each class
    # Note: I use this for loop to iterate also on classes not in the demo batch

    iou_per_class = []
    for class_id in range(nb_classes):
        iou = jaccard_score(
            targets == class_id,
            preds == class_id,
            average="binary",
            zero_division=0,
        )
        iou_per_class.append(iou)

    for class_id, iou in enumerate(iou_per_class):
        print(
            "class {} - IoU: {:.4f} - targets: {} - preds: {}".format(
                class_id, iou, (targets == class_id).sum(), (preds == class_id).sum()
            )
        )


def print_mean_iou(targets: torch.Tensor, preds: torch.Tensor) -> None:
    """
    Compute mean IoU between predictions and targets.

    Args:
        targets (torch.Tensor): Ground truth of shape (B, H, W).
        preds (torch.Tensor): Model predictions of shape (B, nb_classes, H, W).
    """

    mean_iou = jaccard_score(targets, preds, average="macro")
    print(f"meanIOU (over existing classes in targets): {mean_iou:.4f}")


def train_model(
    model,
    dl,
    num_epochs: int = 1,
    filter_ndvi: bool = False,
    learning_rate: float = 1e-3,
):
    """
    Modified training pipeline with some data augmentation.
    """
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    print(device)
    resize_up = Resize((256,256),InterpolationMode.NEAREST_EXACT)
    resize_down = Resize((128,128),InterpolationMode.NEAREST_EXACT)


    
    # Initialize the model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Move the model to the appropriate device (GPU if available)
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        

        for i, (inputs_with_t, targets) in enumerate(dl):
            
            if filter_ndvi:
                for ii in range(len(inputs_with_t)):
                    calculate_ndvi_means(inputs_with_t,bid=ii)
                    inputs_with_t = remove_low_ndvi_images(inputs_with_t,bid=ii)
                    
                    
            inputs = torch.median(inputs_with_t["S2"],dim=1)[0]

            # Move data to device
            inputs = inputs.to(device)  # Satellite data
            targets = targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()
            

            for n_rot_90 in range(1,4):
                #data augmentation by 90 deg rotation
                augm_batch = torch.rot90(inputs,k=n_rot_90,dims=(-2,-1))
                augm_batch_targets = torch.rot90(targets,k=n_rot_90,dims=(-2,-1))
                inputs = torch.cat((inputs,augm_batch))
                targets = torch.cat((targets,augm_batch_targets))
                

            outputs = resize_down(model(resize_up(inputs)))
            targets = targets.flatten()
            outputs = outputs.permute(0,2,3,1).flatten(end_dim=-2)

            

            # Loss computation
            loss = criterion(outputs,targets )

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Print the loss for this epoch
        epoch_loss = running_loss/(i+1)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training complete.")
    return model