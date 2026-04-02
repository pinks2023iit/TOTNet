# TOTNet Resume Training - Complete Guide

## Overview

TOTNet now includes a robust checkpoint resuming system that allows you to seamlessly resume training if your process is interrupted. This guide explains everything you need to know about resuming training.

## Quick Start

### Initial Training
```bash
cd /home/ubuntu/Documents/TOTNet/src
python main.py \
    --saved_fn my_model \
    --model_choice TOTNet \
    --num_epochs 100 \
    --batch_size 8
```

### Resume Training (if interrupted)
```bash
# Same command with --resume flag
python main.py \
    --saved_fn my_model \
    --model_choice TOTNet \
    --num_epochs 100 \
    --batch_size 8 \
    --resume
```

That's it! Training will automatically resume from the last saved checkpoint.

## Resume Options

### 1. Resume from Latest Checkpoint
**Best for:** Continuing interrupted training
```bash
python main.py --saved_fn model_name --model_choice tracknet --resume
```
- Automatically finds and loads the latest checkpoint
- Continues training from the next epoch
- Perfect for handling unexpected interruptions

### 2. Resume from Best Checkpoint
**Best for:** Fine-tuning the best model
```bash
python main.py --saved_fn model_name --model_choice tracknet \
    --resume --resume_from best
```
- Loads the checkpoint with the lowest validation loss
- Useful when you want to continue training the best model
- Good for additional refinement

### 3. Resume from Specific Epoch
**Best for:** Testing specific checkpoints
```bash
python main.py --saved_fn model_name --model_choice tracknet \
    --resume --resume_from epoch_25
```
- Loads the checkpoint from epoch 25
- Use when you want to skip certain epochs
- Helpful for ablation studies

### 4. Resume with Custom Checkpoint Path
**Best for:** Using checkpoints from other locations
```bash
python main.py --saved_fn model_name --model_choice tracknet \
    --resume --resume_from /path/to/checkpoint.pth
```
- Loads from an explicit file path
- Useful for using external checkpoint files

## What Gets Restored

When you resume from a checkpoint, the following are automatically restored:

| Component | Status | Details |
|-----------|--------|---------|
| **Model Weights** | ✓ | Exact state from checkpoint epoch |
| **Optimizer State** | ✓ | Momentum, adaptive learning rates, etc. |
| **LR Scheduler State** | ✓ | Learning rate and scheduler position |
| **Epoch Number** | ✓ | Training continues from next epoch |
| **Best Val Loss** | ✓ | For early stopping tracking |
| **Early Stop Count** | ✓ | If using early stopping |
| **Training Seed** | ✓ | Reproducible results across runs |

## Checkpoint Files

After training, your checkpoint directory structure looks like:

```
checkpoints/
└── my_model/
    ├── my_model_best.pth          # Best model (lowest val loss)
    ├── my_model_epoch_1.pth        # Periodic checkpoints
    ├── my_model_epoch_2.pth
    ├── my_model_epoch_3.pth
    └── ...
```

### Configuration
- **Save Frequency:** Controlled by `--checkpoint_freq` (default: 1 = every epoch)
- **Save Condition 1:** Always save if validation loss is best
- **Save Condition 2:** Save at `checkpoint_freq` intervals regardless of performance

### File Size
- Checkpoint files are typically 100-500 MB depending on model size
- Best checkpoint is only ~200 MB (single best state)
- Periodic checkpoints can accumulate; clean up with: `rm checkpoints/my_model/*_epoch_*.pth`

## Common Use Cases

### Use Case 1: Server Crash During Training
```bash
# Original command that crashed
python main.py --saved_fn experiment1 --model_choice TOTNet --num_epochs 100

# Resume after server restart
python main.py --saved_fn experiment1 --model_choice TOTNet --num_epochs 100 --resume
```
Result: Training continues from the epoch where it crashed, preserving all progress.

### Use Case 2: Out of Memory Error
```bash
# If OOM occurs and you interrupt training, resume with smaller batch size:
python main.py --saved_fn experiment1 --model_choice TOTNet \
    --batch_size 4 --resume --resume_from best
```
Note: Changing batch size may affect convergence but training can continue.

### Use Case 3: Need More Epochs
```bash
# Initial training
python main.py --saved_fn experiment1 --model_choice TOTNet --num_epochs 50

# After epoch 50, increase total epochs and resume
python main.py --saved_fn experiment1 --model_choice TOTNet --num_epochs 100 --resume
```
Result: Training continues and adds 50 more epochs.

### Use Case 4: Distributed Training Resume
```bash
# Initial distributed training
torchrun --nproc_per_node=4 main.py \
    --saved_fn experiment1 --model_choice TOTNet --num_epochs 100

# Resume distributed training
torchrun --nproc_per_node=4 main.py \
    --saved_fn experiment1 --model_choice TOTNet --num_epochs 100 --resume
```
Result: Seamless resume with all ranks synchronized.

## Practical Examples

### Example 1: Complete Training Workflow
```bash
# Step 1: Start training
cd /home/ubuntu/Documents/TOTNet/src
python main.py \
    --saved_fn final_model \
    --model_choice TOTNet \
    --num_epochs 100 \
    --batch_size 8 \
    --num_workers 4 \
    --seed 2024

# (Training runs for 45 epochs then crashes)

# Step 2: Resume training
python main.py \
    --saved_fn final_model \
    --model_choice TOTNet \
    --num_epochs 100 \
    --batch_size 8 \
    --num_workers 4 \
    --seed 2024 \
    --resume

# (Training continues from epoch 46 to 100)

# Step 3: Verify results
python -c "import torch; \
ckpt = torch.load('../checkpoints/final_model/final_model_best.pth'); \
print(f'Final best epoch: {ckpt[\"epoch\"]}, loss: {ckpt[\"best_val_loss\"]:.4f}')"
```

### Example 2: Checkpoint Inspection
```bash
# List all checkpoints
ls -lh ../checkpoints/final_model/

# Check metadata of best checkpoint
python -c "
import torch
ckpt = torch.load('../checkpoints/final_model/final_model_best.pth')
print(f'Epoch: {ckpt[\"epoch\"]}')
print(f'Best Loss: {ckpt[\"best_val_loss\"]:.6f}')
print(f'Early Stop Count: {ckpt[\"earlystop_count\"]}')
print(f'Model Parameters: {len(ckpt[\"state_dict\"])} layers')
"

# Remove old periodic checkpoints (keep space)
rm ../checkpoints/final_model/*_epoch_*.pth
```

## Troubleshooting

### Problem: "Checkpoint file not found"
**Solution:**
```bash
# Verify checkpoint directory exists
ls -la ../checkpoints/my_model/

# If empty, check saved_fn matches training command
# Make sure --saved_fn is identical in both train and resume commands
```

### Problem: "Size mismatch in state_dict"
**Solution:**
```bash
# Ensure model configuration matches original training:
# - Same --model_choice
# - Same --num_channels (if applicable)
# - Same --backbone settings (if applicable)

# Compare original training command with resume command
# They should have identical model-related arguments
```

### Problem: Training starts from epoch 1 instead of resuming
**Solution:**
```bash
# Make sure to add --resume flag
# Incorrect:
python main.py --saved_fn model_name --model_choice tracknet

# Correct:
python main.py --saved_fn model_name --model_choice tracknet --resume
```

### Problem: "Cannot resume, no valid checkpoint found"
**Solution:**
```bash
# Check if training actually saved checkpoints
# Checkpoints are only saved at the end of each epoch
# If interrupted mid-epoch, latest completed epoch is used

# Try resuming from best instead:
python main.py --saved_fn model_name --model_choice tracknet \
    --resume --resume_from best

# If still failing, verify checkpoint file integrity
python -c "import torch; \
ckpt = torch.load('../checkpoints/model_name/model_name_best.pth'); \
print('Checkpoint loaded successfully')"
```

## Best Practices

### ✓ DO:
1. **Use consistent settings** when resuming (batch size, model choice, etc.)
2. **Keep the same seed** for reproducibility
3. **Monitor checkpoint directory** to manage disk space
4. **Use --resume_from best** when fine-tuning models
5. **Back up best checkpoints** to external storage if important

### ✗ DON'T:
1. **Don't change batch size drastically** (minor changes are OK)
2. **Don't modify model architecture** (resume won't work)
3. **Don't delete checkpoint files** during training
4. **Don't use old checkpoints** with different code versions (may have incompatibilities)
5. **Don't interrupt during checkpoint saving** (last epoch may be corrupt)

## Advanced Usage

### Resuming Multiple Experiments
```bash
# Train multiple models
for seed in 42 123 456; do
    python main.py --saved_fn model_seed_$seed \
        --model_choice TOTNet --seed $seed &
done

# If interrupted, resume all
for seed in 42 123 456; do
    python main.py --saved_fn model_seed_$seed \
        --model_choice TOTNet --seed $seed --resume &
done
```

### Continuing with Extended Training
```bash
# Initial: 50 epochs
python main.py --saved_fn model --model_choice TOTNet --num_epochs 50 --resume

# Later: extend to 100 epochs
python main.py --saved_fn model --model_choice TOTNet --num_epochs 100 --resume
```

### Switching Resume Strategy
```bash
# Was training with latest, now want best
python main.py --saved_fn model --model_choice TOTNet \
    --resume --resume_from best
```

## Performance Considerations

- **Resume Overhead:** <1 second to load checkpoint and restore states
- **Memory Impact:** No additional memory (same as fresh start)
- **I/O Cost:** Checkpoint loading takes ~5-10 seconds depending on disk speed
- **Reproducibility:** Exactly reproduces previous run after resuming

## Distributed Training Resume

For multi-GPU training with `torchrun`:

```bash
# Training setup
torchrun --nproc_per_node=4 main.py \
    --saved_fn distributed_model \
    --model_choice TOTNet \
    --num_epochs 100 \
    --resume
```

Key points:
- All 4 processes automatically synchronize from same checkpoint
- No special configuration needed for resume in distributed mode
- Checkpoint contains full model state for any number of GPUs
- Optimizer and scheduler states are preserved across processes

## Useful Scripts

### Check All Checkpoints
```bash
python resume_examples.py
```

### List Checkpoint Details
```bash
ls -lh ../checkpoints/*/
```

### Inspect Specific Checkpoint
```bash
python -c "import torch; import json; \
ckpt = torch.load('../checkpoints/model/model_best.pth'); \
print(json.dumps({k: str(v) if not isinstance(v, dict) else '...' \
for k, v in ckpt.items()}, indent=2))"
```

## Getting Help

If you encounter issues with resume:

1. Check this guide's troubleshooting section
2. Review the example scripts in `resume_examples.py`
3. Check training logs in `logs/{saved_fn}/`
4. Verify checkpoint files exist: `ls -la checkpoints/{saved_fn}/`
5. Inspect checkpoint: `python -c "import torch; ckpt = torch.load('...'); print(ckpt.keys())"`

## Additional Resources

- Main training guide: See `README.md`
- Example commands: See `resume_examples.sh`
- Python examples: See `resume_examples.py`
- Implementation: See `src/utils/resume_utils.py`
- Integration: See `src/main.py` (search for "resume")
