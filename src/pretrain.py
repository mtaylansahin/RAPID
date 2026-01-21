"""Encoder pretraining for RAPID - trains RGCN + entity embeddings on link prediction."""

from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from src.data.dataset import PPIDataModule
from src.losses import get_loss_function
from src.metrics import MetricsComputer
from src.models.rapid import RAPIDModel
from src.models.classifier import SymmetricPairProjection, EdgeClassifier


def pretrain_encoder(
    model: RAPIDModel,
    data_module: PPIDataModule,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    output_path: Optional[Path] = None,
    patience: int = 5,
    focal_gamma: float = 2.0,
) -> Path:
    """
    Pretrain encoder on link prediction task.

    With the edge-centric architecture, this pretrains:
    - Entity embeddings
    - RGCN structural encoder
    - A simple classification head (not used in final model)

    The pretrained encoder provides good static entity representations
    for the decoder to use in cross-attention.

    Args:
        model: RAPIDModel to pretrain
        data_module: Data module with training data
        device: Torch device
        epochs: Number of training epochs
        lr: Learning rate
        output_path: Path to save checkpoint
        patience: Early stopping patience
        focal_gamma: Focal loss gamma parameter

    Returns:
        Path to saved checkpoint
    """
    print(f"\n{'=' * 60}")
    print("Encoder Pretraining")
    print(f"{'=' * 60}")
    print(f"  Entities: {data_module.num_entities}")
    print(f"  Relations: {data_module.num_rels}")
    print(f"  Train timesteps: {len(data_module.train_times)}")
    print(f"{'=' * 60}\n")

    model = model.to(device)

    # Create a simple classifier head for pretraining
    # (won't be used in the final model - decoder has its own)
    pair_proj = SymmetricPairProjection(model.hidden_dim).to(device)
    pretrain_classifier = EdgeClassifier(
        input_dim=pair_proj.output_dim,
        hidden_dim=128,
        dropout=0.2,
    ).to(device)

    # Combine parameters for optimization
    all_params = (
        list(model.parameters())
        + list(pair_proj.parameters())
        + list(pretrain_classifier.parameters())
    )
    optimizer = torch.optim.Adam(all_params, lr=lr, weight_decay=1e-5)
    criterion = get_loss_function(loss_type="focal", gamma=focal_gamma)

    if output_path is None:
        output_path = Path("./models/encoder.pth")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_auprc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        pair_proj.train()
        pretrain_classifier.train()
        metrics = MetricsComputer()

        dataloader = data_module.get_train_dataloader()
        pbar = tqdm(dataloader, desc=f"Pretrain Epoch {epoch:03d}")

        for batch in pbar:
            entity1 = batch["entity1"].to(device)
            entity2 = batch["entity2"].to(device)
            labels = batch["labels"].to(device)

            # Get entity embeddings
            entity1_embed, entity2_embed = model(
                entity1_ids=entity1,
                entity2_ids=entity2,
                graph_dict=data_module.graph_dict,
            )

            # Apply symmetric pair projection + classifier
            pair_features = pair_proj(entity1_embed, entity2_embed)
            logits = pretrain_classifier(pair_features)

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            metrics.update(logits, labels, loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Compute epoch metrics
        epoch_metrics = metrics.compute()
        print(f"  Epoch {epoch}: {epoch_metrics.short_str()}")

        # Save best (only save the encoder, not pretrain classifier)
        if epoch_metrics.auprc > best_auprc:
            best_auprc = epoch_metrics.auprc
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "auprc": best_auprc,
                    "config": {
                        "num_entities": data_module.num_entities,
                        "num_rels": data_module.num_rels,
                        "hidden_dim": model.hidden_dim,
                    },
                },
                output_path,
            )
            print(f"    -> Saved (AUPRC: {best_auprc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print("\nEncoder pretraining complete!")
    print(f"  Best AUPRC: {best_auprc:.4f}")
    print(f"  Saved to: {output_path}")

    return output_path
