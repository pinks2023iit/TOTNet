# Automatic Resume Feature

## Overview

The TOTNet training script now includes **automatic checkpoint detection**. When you run the training script, it will automatically:

1. **Check** if logs and checkpoints exist for your model
2. **Report** what it found
3. **Resume** training from the latest checkpoint if any exist
4. **Start fresh** from epoch 1 if no checkpoints are found

This means you can simply run the same command repeatedly, and the training will automatically resume from where it left off if interrupted.

---

## How It Works

### Automatic Detection Flow

```
┌─────────────────────────────────────────┐
│   Run: python main.py --saved_fn model  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│   Check for existing logs & checkpoints │
└────────────┬────────────────────────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
   FOUND       NOT FOUND
      │             │
      ▼             ▼
  Resume      Start Fresh
  Training    Training
```

### Detection Report

When you run the script, you'll see a report like this:

**If checkpoints exist:**
```
======================================================================
CHECKPOINT DETECTION REPORT
======================================================================
✓ Existing logs found in: logs/my_model
✓ Found 45 checkpoint(s)
✓ Latest epoch: 45
✓ Will resume from: my_model_epoch_45.pth
→ Auto-enabling resume mode due to existing checkpoints...
======================================================================
```

**If no checkpoints exist:**
```
======================================================================
CHECKPOINT DETECTION REPORT
======================================================================
✗ No existing logs found
✗ No checkpoints found
→ Starting training from scratch
======================================================================
```

---

## Usage Examples

### Example 1: Simple Training with Auto-Resume

```bash
cd /home/ubuntu/Documents/TOTNet/src

# First run - starts from scratch
python main.py \
    --saved_fn my_model \
    --model_choice tracknet \
    --num_epochs 100 \
    --batch_size 8

# If interrupted at epoch 45 and you run the same command again...
# The script will automatically resume from epoch 45!

python main.py \
    --saved_fn my_model \
    --model_choice tracknet \
    --num_epochs 100 \
    --batch_size 8
```

### Example 2: Distributed Training with Auto-Resume

```bash
# First run
torchrun --nproc_per_node=4 main.py \
    --saved_fn my_model \
    --model_choice tracknet \
    --num_epochs 100 \
    --batch_size 8

# If interrupted, simply run the same command again
# It will auto-resume from the latest checkpoint
torchrun --nproc_per_node=4 main.py \
    --saved_fn my_model \
    --model_choice tracknet \
    --num_epochs 100 \
    --batch_size 8
```

### Example 3: Override Auto-Resume

If you want to start fresh despite having existing checkpoints, use:

```bash
python main.py \
    --saved_fn my_model \
    --model_choice tracknet \
    --num_epochs 100 \
    --resume_from epoch_0  # Forces start from scratch
```

Or simply delete the checkpoint directory:

```bash
rm -rf checkpoints/my_model/
python main.py --saved_fn my_model --model_choice tracknet --num_epochs 100
```

---

## What Gets Automatically Resumed

When the script detects existing checkpoints and resumes:

1. ✓ **Model Weights** - All trained weights
2. ✓ **Optimizer State** - Momentum, adaptive learning rates, etc.
3. ✓ **Learning Rate Scheduler** - Learning rate schedule position
4. ✓ **Training Epoch** - Continues from the exact epoch
5. ✓ **Best Validation Loss** - For tracking improvements
6. ✓ **Early Stopping Count** - If using early stopping

---

## Configuration

### Default Behavior

By default, the auto-resume feature is **always active**. The script will:

- Check logs and checkpoints directories
- Print detection report
- Automatically resume if checkpoints exist
- Start fresh if no checkpoints exist

### Manual Control (Optional)

You can also manually control resume behavior with command-line flags:

```bash
# Explicit resume flag (though not needed due to auto-detection)
python main.py --saved_fn my_model --resume

# Resume from specific checkpoint
python main.py --saved_fn my_model --resume --resume_from epoch_20

# Resume from best checkpoint
python main.py --saved_fn my_model --resume --resume_from best
```

---

## Important Notes

1. **Same Configuration**: Use the same model configuration (`--model_choice`, `--num_channels`, etc.) when resuming

2. **Same Batch Size**: Keep the batch size consistent for reproducible results

3. **Seed**: Use the same `--seed` for reproducibility

4. **Distributed Training**: Auto-resume works seamlessly with distributed training

5. **Directory Consistency**: Don't move checkpoints or logs directories while training

---

## Troubleshooting

### Issue: "Still starting from epoch 1 instead of resuming"

Check that:
1. The logs directory exists and has content
2. The checkpoints directory exists and has .pth files
3. Run `ls -la checkpoints/my_model/` to verify

### Issue: "Size mismatch when loading checkpoint"

This occurs when:
- Model architecture changed (e.g., different `--num_channels`)
- Checkpoint is from a different model
- Solution: Use a different `--saved_fn` to create new checkpoints

### Issue: "Want to start fresh but checkpoints exist"

Delete existing checkpoints:
```bash
rm -rf checkpoints/my_model/
rm -rf logs/my_model/
```

Or use a different model name:
```bash
python main.py --saved_fn my_model_v2 --model_choice tracknet
```

---

## Viewing Checkpoint Information

### List all checkpoints for a model:

```bash
ls -lh checkpoints/my_model/
```

Example output:
```
-rw-r--r-- 1 user user 256M Apr 2 10:45 my_model_best.pth
-rw-r--r-- 1 user user 256M Apr 2 10:30 my_model_epoch_45.pth
-rw-r--r-- 1 user user 256M Apr 2 10:15 my_model_epoch_44.pth
-rw-r--r-- 1 user user 256M Apr 2 10:00 my_model_epoch_43.pth
```

### Check specific checkpoint metadata:

```bash
python3 << 'EOF'
import torch
ckpt = torch.load('checkpoints/my_model/my_model_best.pth')
print(f"Epoch: {ckpt['epoch']}")
print(f"Best Val Loss: {ckpt['best_val_loss']:.6f}")
print(f"Early Stop Count: {ckpt['earlystop_count']}")
EOF
```

---

## Summary

The **automatic resume feature** makes training more robust:

- ✓ **No manual intervention needed** - Just run the same command
- ✓ **Safe** - Checks for existing checkpoints before overwriting
- ✓ **Informative** - Shows what it detected and what it's doing
- ✓ **Flexible** - Can still manually control resume behavior if needed
- ✓ **Distributed-friendly** - Works with multi-GPU training

**Simply run your training command, and it will automatically resume if interrupted!**
