# Enhanced Resume Mechanism with Checkpoint Validation

## Overview

The TOTNet training system now includes an **enhanced resume mechanism** that:

1. **Automatically detects** existing logs and checkpoints
2. **Validates checkpoint sequence** to detect interrupted training
3. **Prevents resuming from stale checkpoints** when training is abruptly terminated
4. **Provides detailed warnings** if potential issues are detected

## How It Works

### 1. Automatic Checkpoint Detection

When you run the training script, it automatically checks for:
- Existing logs in `logs/{saved_fn}/`
- Existing checkpoints in `checkpoints/{saved_fn}/`

**Output Example:**
```
======================================================================
CHECKPOINT DETECTION REPORT
======================================================================
✓ Existing logs found in: ../logs/TOTNet_final
✓ Found 31 checkpoint(s)
✓ Latest epoch: 30
✓ Will resume from: TOTNet_final_epoch_30.pth
======================================================================

→ Auto-enabling resume mode due to existing checkpoints...
```

### 2. Checkpoint Sequence Validation

The system validates checkpoints by analyzing **modification timestamps**:

- Reads modification time of each checkpoint file
- Detects time gaps between consecutive epochs
- Flags gaps larger than 60 minutes (configurable)
- Warns if training appears to have been interrupted

**Detection Logic:**
```
Epoch 1:  [timestamp: 10:00] ✓
Epoch 2:  [timestamp: 10:05] ✓ (5 min gap - normal)
Epoch 3:  [timestamp: 10:10] ✓ (5 min gap - normal)
...
Epoch 15: [timestamp: 11:15] ✓ (5 min gap - normal)
Epoch 16: [timestamp: 18:45] ⚠ (7.5 hour gap - INTERRUPTED!)
                          ↑
              Resume from epoch 15 instead
```

### 3. Safe Resume Decision

Based on validation:

| Scenario | Action |
|----------|--------|
| Continuous checkpoints, all recent | Resume from latest checkpoint |
| Large gap detected between epochs | Resume from last epoch before gap |
| No checkpoints found | Start training from epoch 1 |
| Corrupted/invalid checkpoints | Start training from epoch 1 |

## Understanding the Output

### Example 1: Normal Resume (No Issues)

```
Using latest checkpoint: ../checkpoints/TOTNet_final/TOTNet_final_epoch_30.pth

============================================================
Loading checkpoint from: ../checkpoints/TOTNet_final/TOTNet_final_epoch_30.pth
============================================================
✓ Model state loaded successfully
✓ Optimizer state loaded successfully
✓ Learning rate scheduler state loaded successfully

✓ Checkpoint Summary:
  - Last saved epoch: 30
  - Best validation loss: 7.911729
  - Early stopping count: 0
============================================================

Continuing training from epoch 31
```

### Example 2: Interrupted Training Detected

```
Using latest checkpoint: ../checkpoints/TOTNet_final/TOTNet_final_epoch_15.pth

⚠ WARNING: Large time gap detected between epoch 15 and 30: 7.5 hours.
Training may have been interrupted.
→ Resuming from epoch 15 instead of latest checkpoint.

============================================================
Loading checkpoint from: ../checkpoints/TOTNet_final/TOTNet_final_epoch_15.pth
============================================================
✓ Model state loaded successfully
✓ Optimizer state loaded successfully
✓ Learning rate scheduler state loaded successfully

✓ Checkpoint Summary:
  - Last saved epoch: 15
  - Best validation loss: 8.234512
  - Early stopping count: 0
============================================================

Continuing training from epoch 16
```

## Configuration

### Time Gap Threshold

The default time gap threshold is **60 minutes**. To modify:

Edit `src/main.py` line ~140:
```python
validation_result = validate_checkpoint_sequence(
    configs.checkpoints_dir, 
    configs.saved_fn,
    max_time_gap_minutes=60  # Change this value
)
```

### Usage Examples

#### 1. Auto-Resume (Recommended)
```bash
python main.py \
    --saved_fn my_model \
    --model_choice tracknet \
    --num_epochs 100
```

#### 2. Force Resume from Latest
```bash
python main.py \
    --saved_fn my_model \
    --model_choice tracknet \
    --num_epochs 100 \
    --resume
```

#### 3. Resume from Specific Epoch
```bash
python main.py \
    --saved_fn my_model \
    --model_choice tracknet \
    --num_epochs 100 \
    --resume \
    --resume_from epoch_15
```

#### 4. Resume from Best Checkpoint
```bash
python main.py \
    --saved_fn my_model \
    --model_choice tracknet \
    --num_epochs 100 \
    --resume \
    --resume_from best
```

#### 5. Start Fresh (Ignore Existing Checkpoints)
```bash
python main.py \
    --saved_fn my_model_v2 \
    --model_choice tracknet \
    --num_epochs 100
```
*(Use different `--saved_fn` to avoid auto-resume)*

## Checkpoint State

When resuming, the following are **fully restored**:

```
Checkpoint Contains:
├── Model Weights (state_dict)
├── Optimizer State
│   ├── Momentum buffers (for SGD/Adam)
│   ├── Running averages
│   └── Gradient history
├── LR Scheduler State
│   ├── Current learning rate
│   ├── Scheduler step count
│   └── Last epoch info
├── Training Metadata
│   ├── Current epoch
│   ├── Best validation loss
│   ├── Early stopping count
│   └── Training configuration
└── Validation Tracking
    └── Best model info
```

## What Gets Restored

### ✓ Fully Restored
- Model weights and biases
- Optimizer momentum and state
- Learning rate and scheduler
- Training epoch counter
- Best validation loss
- Early stopping counter

### ✗ NOT Restored
- Dataloader state (starts from beginning of epoch)
- Random seed (use `--seed` for reproducibility)
- Batch sampler order (use `--seed` for reproducibility)

## Best Practices

### 1. **Use Consistent Settings**
```bash
# Initial training
python main.py --saved_fn my_model --batch_size 8 --seed 2024

# Resume with same settings
python main.py --saved_fn my_model --batch_size 8 --seed 2024
```

### 2. **Monitor Checkpoint Timestamps**
```bash
# Check checkpoint modification times
ls -lh checkpoints/my_model/

# Example output:
# -rw-r--r-- 1 user group 123M Apr 02 10:05 my_model_epoch_1.pth
# -rw-r--r-- 1 user group 123M Apr 02 10:10 my_model_epoch_2.pth
# -rw-r--r-- 1 user group 123M Apr 02 10:15 my_model_epoch_3.pth
# ↑ Normal progression (5 minutes per epoch)
```

### 3. **Clean Up Before New Training**
If starting completely fresh:
```bash
# Remove old checkpoints
rm -rf checkpoints/my_model/
rm -rf logs/my_model/

# Use different name for new training
python main.py --saved_fn my_model_v2 ...
```

### 4. **Check Logs for Issues**
```bash
# View last few lines of training log
tail -20 logs/my_model/*.log

# Check for errors
grep -i "error\|warning" logs/my_model/*.log
```

## Troubleshooting

### Issue: "Training may have been interrupted" Warning

**Cause:** Large time gap detected between consecutive checkpoints

**Solution:**
1. Check if training was actually interrupted
2. If yes, use the suggested epoch to resume
3. If no (e.g., system clock adjusted), adjust `max_time_gap_minutes`

### Issue: Resuming from Wrong Epoch

**Cause:** Checkpoint numbering issue or manual checkpoint deletion

**Solution:**
```bash
# List all available checkpoints
ls -lh checkpoints/my_model/

# Resume from specific safe epoch
python main.py --saved_fn my_model --resume --resume_from epoch_10
```

### Issue: "Size Mismatch" Error When Resuming

**Cause:** Model architecture changed since original training

**Solution:**
```bash
# Use exact same model config
python main.py \
    --saved_fn my_model \
    --model_choice tracknet \
    --num_channels 64 \
    --num_frames 9
```

## Advanced: Programmatic Access

```python
from utils.checkpoint_validator import validate_checkpoint_sequence, find_safe_checkpoint

# Validate checkpoint sequence
result = validate_checkpoint_sequence(
    checkpoints_dir='../checkpoints/my_model',
    saved_fn='my_model',
    max_time_gap_minutes=60
)

if result['is_valid']:
    print(f"Safe to resume from epoch {result['valid_epoch']}")
    if result['gap_detected']:
        print(f"Warning: {result['warning']}")
else:
    print("No valid checkpoints found")

# Get safe checkpoint directly
checkpoint_path, epoch, is_safe = find_safe_checkpoint(
    '../checkpoints/my_model',
    'my_model',
    max_time_gap_minutes=60
)

if is_safe:
    print(f"Resume from {checkpoint_path} (epoch {epoch})")
```

## Summary

The enhanced resume mechanism provides:

1. **Automatic detection** of existing training sessions
2. **Smart validation** to prevent resuming from stale checkpoints
3. **Clear warnings** when potential issues are detected
4. **Flexible options** for manual checkpoint selection
5. **Complete state restoration** including optimizer and scheduler

This ensures that your training can be safely interrupted and resumed without data corruption or loss of training progress.
