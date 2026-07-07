"""
Model construction for the EEG binary classification pipeline.

Wraps braindecode's CBraMod (https://braindecode.org/dev/generated/braindecode.models.CBraMod.html),
loading pretrained weights from the Hugging Face Hub and configuring the
classification head for the target number of channels / classes.
"""

import torch
from braindecode.models import CBraMod

PRETRAINED_REPO_ID = "braindecode/cbramod-pretrained"


def build_model(
    n_chans: int,
    n_outputs: int,
    n_times: int = 200,
    pretrained_repo_id: str = PRETRAINED_REPO_ID,
    freeze_backbone: bool = True,
):
    """
    Instantiate CBraMod from pretrained weights, configured for this task.

    Parameters
    ----------
    n_chans : int
        Number of EEG channels in the input.
    n_outputs : int
        Number of target classes.
    n_times : int
        Number of time samples per input window (n_segments * patch_size).
        Must be set so the classification head is a concrete nn.Linear
        instead of a LazyLinear — pretrained weights can't be loaded into
        an uninitialized LazyLinear.
    pretrained_repo_id : str
        Hugging Face Hub repo to load pretrained weights from.
    freeze_backbone : bool
        If True, freeze everything except the classification head.
        Recommended for small datasets to avoid overfitting.

    Returns
    -------
    model : CBraMod
    device : torch.device
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CBraMod.from_pretrained(
        pretrained_repo_id,
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
    ).to(device)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("final_layer"):
                param.requires_grad = False
        print("Backbone frozen — only classification head will be trained")
    else:
        print("Full fine-tuning — backbone + classification head")

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_trainable:,} trainable / {n_total:,} total")

    return model, device
