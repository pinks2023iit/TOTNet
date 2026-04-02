# Quick Reference: Auto-Resume Training

## One-Line Summary
**Run the same training command repeatedly - it automatically resumes from the latest checkpoint!**

---

## Quick Start

### First Training Run
```bash
cd src
python main.py \
    --saved_fn my_model \
    --model_choice tracknet \
    --num_epochs 100
```

### Training Gets Interrupted at Epoch 45?
Simply run the **exact same command** again:
```bash
cd src
python main.py \
    --saved_fn my_model \
    --model_choice tracknet \
    --num_epochs 100
```

**Result**: Automatically resumes from epoch 46! ✓

---

## What You'll See

### First Run Output
```
Number of GPUs: 2
GPU 0: NVIDIA A100
GPU 1: NVIDIA A100

======================================================================
CHECKPOINT DETECTION REPORT
======================================================================
✗ No existing logs found
✗ No checkpoints found
→ Starting training from scratch
======================================================================

[Epoch 1/100]  Loss: 0.4521 ...
[Epoch 2/100]  Loss: 0.3891 ...
```

### Resume Run Output (After Interruption)
```
Number of GPUs: 2
GPU 0: NVIDIA A100
GPU 1: NVIDIA A100

======================================================================
CHECKPOINT DETECTION REPORT
======================================================================
✓ Existing logs found in: logs/my_model
✓ Found 45 checkpoint(s)
✓ Latest epoch: 45
✓ Will resume from: my_model_epoch_45.pth
→ Auto-enabling resume mode due to existing checkpoints...

============================================================
Loading checkpoint from: checkpoints/my_model/my_model_epoch_45.pth
============================================================
✓ Model state loaded successfully
✓ Optimizer state loaded successfully
✓ Learning rate scheduler state loaded successfully

✓ Checkpoint Summary:
  - Last saved epoch: 45
  - Best validation loss: 0.234567
  - Early stopping count: 0
============================================================

[Epoch 46/100] Loss: 0.3124 ...
[Epoch 47/100] Loss: 0.2891 ...
```

---

## Common Scenarios

### Scenario 1: Normal Training
```bash
python main.py --saved_fn model_v1 --model_choice tracknet --num_epochs 100
# Runs epochs 1-100
```

### Scenario 2: Training Interrupted
```bash
# Stops at epoch 45...
# Run same command:
python main.py --saved_fn model_v1 --model_choice tracknet --num_epochs 100
# Resumes at epoch 46 ✓
```

### Scenario 3: Start Fresh (Force Restart)
```bash
# Delete checkpoints first
rm -rf checkpoints/model_v1/
rm -rf logs/model_v1/

# Now run - starts fresh
python main.py --saved_fn model_v1 --model_choice tracknet --num_epochs 100
```

### Scenario 4: Multiple Experiments
```bash
# Each model name is independent
python main.py --saved_fn model_v1 --model_choice tracknet --num_epochs 100
python main.py --saved_fn model_v2 --model_choice tracknetv2 --num_epochs 100
python main.py --saved_fn model_v3 --model_choice wasb --num_epochs 100

# Each has its own checkpoint directory
```

---

## What Gets Restored

- ✓ Model weights
- ✓ Optimizer momentum/state
- ✓ Learning rate schedule position
- ✓ Training epoch number
- ✓ Best validation loss tracking
- ✓ Early stopping counter

---

## Commands Cheat Sheet

```bash
# View available checkpoints
ls -lh checkpoints/my_model/

# Check checkpoint info
python3 -c "import torch; ck=torch.load('checkpoints/my_model/my_model_best.pth'); print(f'Epoch: {ck[\"epoch\"]}')"

# List logs
ls -lh logs/my_model/

# Delete all checkpoints for model
rm -rf checkpoints/my_model/

# Delete all logs for model
rm -rf logs/my_model/

# Run with all defaults
python main.py --saved_fn model --model_choice tracknet
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Still starts from epoch 1 | Verify `checkpoints/my_model/` has .pth files |
| "Size mismatch" error | Use different `--saved_fn` or delete old checkpoints |
| Want to train multiple variants | Use different `--saved_fn` for each |
| Checkpoints from old model | Change `--saved_fn` or delete checkpoint directory |
| Slow checkpoint loading | Checkpoints are large; this is normal |

---

## Key Points

1. **Automatic** - No manual `--resume` flag needed (though it still works)
2. **Safe** - Only resumes if checkpoints are found
3. **Smart** - Finds latest checkpoint automatically
4. **Flexible** - Can start fresh by deleting checkpoints
5. **Distributed** - Works with multi-GPU training

---

## For Distributed Training

```bash
# Auto-resume works here too!
torchrun --nproc_per_node=4 main.py \
    --saved_fn my_model \
    --model_choice tracknet \
    --num_epochs 100
```

---

## Remember

**Just run the same command again if interrupted - it will automatically resume!** 🚀
