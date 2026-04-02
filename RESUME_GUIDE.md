# Checkpoint Resuming Guide for TOTNet

## Overview
This guide explains how to resume training from a checkpoint if the training process is interrupted.

## How to Resume Training

### Option 1: Resume from Latest Checkpoint (Recommended)
To resume from the most recent checkpoint automatically:

```bash
python main.py \
    --saved_fn your_model_name \
    --model_choice tracknet \
    --resume
```

The `--resume` flag will automatically find and load the latest checkpoint from the `checkpoints/your_model_name/` directory.

### Option 2: Resume from Specific Checkpoint
To resume from a specific epoch:

```bash
python main.py \
    --saved_fn your_model_name \
    --model_choice tracknet \
    --resume \
    --resume_from epoch_15
```

This will load the checkpoint from `checkpoints/your_model_name/your_model_name_epoch_15.pth`.

### Option 3: Resume from Best Checkpoint
To resume from the best validation loss checkpoint:

```bash
python main.py \
    --saved_fn your_model_name \
    --model_choice tracknet \
    --resume \
    --resume_from best
```

This will load `checkpoints/your_model_name/your_model_name_best.pth`.

## What Gets Restored

When you resume from a checkpoint, the following are automatically restored:

1. **Model Weights** - The neural network weights
2. **Optimizer State** - Momentum, moving averages, etc.
3. **Learning Rate Scheduler State** - Current learning rate and scheduler state
4. **Training Epoch** - Training continues from the exact epoch where it stopped
5. **Best Validation Loss** - For early stopping and best model tracking
6. **Early Stopping Count** - If using early stopping, this is restored

## Example Training Session

### Initial Training
```bash
python main.py \
    --saved_fn my_model \
    --model_choice tracknet \
    --num_epochs 100 \
    --batch_size 8
```

If this stops at epoch 45, your checkpoint saves the complete training state.

### Resume Training
```bash
python main.py \
    --saved_fn my_model \
    --model_choice tracknet \
    --num_epochs 100 \
    --batch_size 8 \
    --resume
```

Training will continue from epoch 46 with:
- All model weights from epoch 45
- Optimizer state from epoch 45
- Learning rate scheduler at the correct step
- Best validation loss value preserved
- Early stopping counter (if applicable)

## Checkpoint Files

After training, your checkpoint directory will contain:

```
checkpoints/my_model/
├── my_model_best.pth          # Best model (lowest validation loss)
├── my_model_epoch_1.pth        # Periodic checkpoints
├── my_model_epoch_10.pth
├── my_model_epoch_20.pth
└── ...
```

The frequency of periodic checkpoints is controlled by `--checkpoint_freq` (default: 1 = save every epoch).

## Important Notes

1. **Batch Size Consistency**: Use the same batch size when resuming for consistent behavior
2. **Data Order**: With distributed training, ensure the same data order by using `--seed`
3. **GPU Setup**: The script will automatically detect and use appropriate GPUs
4. **Learning Rate**: The learning rate scheduler will continue from where it left off
5. **Validation Set**: Validation set is preserved for consistent metric tracking

## Troubleshooting

### Issue: "Checkpoint file not found"
- Verify the checkpoint directory exists: `ls checkpoints/your_model_name/`
- Check that the checkpoint file name is correct
- Ensure you're using the same `--saved_fn` as during initial training

### Issue: "Size mismatch" errors
- Ensure the model configuration (e.g., `--model_choice`, `--num_channels`) matches the original training
- Check that the checkpoint was saved from a compatible model architecture

### Issue: Training continues from epoch 1 instead of resuming
- Make sure to include the `--resume` flag
- Check that the checkpoint directory has checkpoint files

## Distributed Training Resume

For distributed training with multiple GPUs:

```bash
torchrun --nproc_per_node=4 main.py \
    --saved_fn my_model \
    --model_choice tracknet \
    --num_epochs 100 \
    --resume
```

The resume mechanism works seamlessly with distributed training.

## Monitoring Checkpoints

To see all available checkpoints:
```bash
ls -lh checkpoints/your_model_name/
```

To check checkpoint metadata (epoch, best_val_loss, etc.):
```bash
python -c "import torch; ckpt = torch.load('checkpoints/your_model/your_model_best.pth'); print(f\"Epoch: {ckpt['epoch']}, Best Loss: {ckpt['best_val_loss']:.4f}\")"
```
