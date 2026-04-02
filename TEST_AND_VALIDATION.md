# TOTNet Resume Training - Testing & Validation Guide

## Quick Validation Checklist

Before deploying resume functionality to production, verify:

- [ ] Resume configuration arguments are recognized
- [ ] Checkpoint files are created during training
- [ ] Resume functionality correctly identifies checkpoints
- [ ] Model weights are correctly restored from checkpoint
- [ ] Optimizer state is correctly restored
- [ ] Learning rate scheduler state is correctly restored
- [ ] Training continues from the correct epoch
- [ ] Distributed training resume works correctly
- [ ] Error handling works for missing checkpoints

## Test 1: Configuration Arguments

### Objective
Verify that `--resume` and `--resume_from` arguments are properly parsed.

### Steps
```bash
cd /home/ubuntu/Documents/TOTNet/src

# Test 1a: Check help for resume arguments
python main.py --help | grep -A 2 resume

# Expected output:
# --resume              Resume training from the latest checkpoint in the checkpoint directory
# --resume_from RESUME_FROM
#                       Resume from a specific checkpoint. Can be "best" for best checkpoint, "epoch_X" for specific epoch, or a checkpoint filename
```

### Validation
- [ ] Both `--resume` and `--resume_from` appear in help
- [ ] Arguments are properly described
- [ ] Help text is clear and accurate

---

## Test 2: Checkpoint File Creation

### Objective
Verify that checkpoints are created during training.

### Steps
```bash
cd /home/ubuntu/Documents/TOTNet/src

# Create a small test to verify checkpoint creation
mkdir -p ../test_checkpoints

# Run a very short training (1-2 epochs) to generate checkpoints
python main.py \
    --saved_fn test_model \
    --model_choice tracknet \
    --num_epochs 2 \
    --batch_size 4 \
    --num_workers 2 \
    --num_samples 10 \
    --seed 2024 \
    2>&1 | head -100

# Verify checkpoints were created
ls -la ../checkpoints/test_model/
```

### Validation
- [ ] Training completes without errors
- [ ] Checkpoint directory is created
- [ ] At least one checkpoint file is created
- [ ] Checkpoint file size is > 50 MB (should contain model)
- [ ] Checkpoint file contains expected data

---

## Test 3: Checkpoint Content Verification

### Objective
Verify checkpoint files contain all required information.

### Steps
```bash
cd /home/ubuntu/Documents/TOTNet/src

# Inspect checkpoint content
python << 'EOF'
import torch
import os

checkpoint_dir = '../checkpoints/test_model'
if os.path.exists(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth'):
            filepath = os.path.join(checkpoint_dir, file)
            print(f"\nInspecting: {file}")
            print("-" * 60)
            
            try:
                ckpt = torch.load(filepath, map_location='cpu')
                print(f"Checkpoint Keys: {list(ckpt.keys())}")
                
                # Verify required fields
                required_keys = ['state_dict', 'optimizer', 'epoch', 'best_val_loss', 'earlystop_count']
                for key in required_keys:
                    if key in ckpt:
                        print(f"✓ {key}: Present")
                    else:
                        print(f"✗ {key}: MISSING!")
                
                # Show details
                print(f"\nDetails:")
                print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
                print(f"  Best Val Loss: {ckpt.get('best_val_loss', 'N/A')}")
                print(f"  Early Stop Count: {ckpt.get('earlystop_count', 'N/A')}")
                print(f"  Model Layers: {len(ckpt.get('state_dict', {}))}")
                
            except Exception as e:
                print(f"✗ Error loading checkpoint: {e}")
else:
    print("Checkpoint directory not found!")
EOF
```

### Validation
- [ ] All required keys are present in checkpoint
- [ ] `state_dict` contains model weights
- [ ] `optimizer` contains optimizer state
- [ ] `epoch` is a valid integer
- [ ] `best_val_loss` is a float or infinity
- [ ] `earlystop_count` is an integer

---

## Test 4: Resume Functionality - Latest Checkpoint

### Objective
Verify resuming from the latest checkpoint works correctly.

### Steps
```bash
cd /home/ubuntu/Documents/TOTNet/src

# Step 1: Initial training (3 epochs)
python main.py \
    --saved_fn test_resume_latest \
    --model_choice tracknet \
    --num_epochs 3 \
    --batch_size 4 \
    --num_workers 2 \
    --num_samples 10 \
    --seed 2024 2>&1 | tail -50

# Step 2: List checkpoints from first run
echo ""
echo "Checkpoints after first run:"
ls -la ../checkpoints/test_resume_latest/

# Step 3: Resume training (should continue from epoch 4)
python main.py \
    --saved_fn test_resume_latest \
    --model_choice tracknet \
    --num_epochs 5 \
    --batch_size 4 \
    --num_workers 2 \
    --num_samples 10 \
    --seed 2024 \
    --resume 2>&1 | head -100

# Step 4: Verify training continued (check for "epoch 4" or "epoch 5")
```

### Validation
- [ ] First run completes successfully
- [ ] Checkpoints are created for each epoch
- [ ] Resume command is recognized (no error about unknown argument)
- [ ] Training output shows resuming from checkpoint
- [ ] Training continues from epoch 4 (not epoch 1)
- [ ] Model state is loaded (check logs)
- [ ] Training progresses through epochs 4-5

---

## Test 5: Resume Functionality - Best Checkpoint

### Objective
Verify resuming from the best checkpoint specifically.

### Steps
```bash
cd /home/ubuntu/Documents/TOTNet/src

# Assuming test_resume_latest has a best checkpoint

# Resume from best checkpoint
python main.py \
    --saved_fn test_resume_latest \
    --model_choice tracknet \
    --num_epochs 6 \
    --batch_size 4 \
    --num_workers 2 \
    --num_samples 10 \
    --seed 2024 \
    --resume \
    --resume_from best 2>&1 | head -100

# Verify checkpoint metadata
python << 'EOF'
import torch

checkpoint_path = '../checkpoints/test_resume_latest/test_resume_latest_best.pth'
try:
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    print(f"Best Checkpoint Metadata:")
    print(f"  Epoch: {ckpt['epoch']}")
    print(f"  Best Val Loss: {ckpt['best_val_loss']}")
    print(f"  Training will resume from epoch: {ckpt['epoch'] + 1}")
except FileNotFoundError:
    print(f"Best checkpoint not found: {checkpoint_path}")
EOF
```

### Validation
- [ ] Best checkpoint is found
- [ ] Resume from best works without error
- [ ] Training continues from best epoch + 1
- [ ] Correct checkpoint is loaded (verify in logs)

---

## Test 6: Resume Functionality - Specific Epoch

### Objective
Verify resuming from a specific epoch works.

### Steps
```bash
cd /home/ubuntu/Documents/TOTNet/src

# Resume from specific epoch (e.g., epoch 2)
python main.py \
    --saved_fn test_resume_latest \
    --model_choice tracknet \
    --num_epochs 6 \
    --batch_size 4 \
    --num_workers 2 \
    --num_samples 10 \
    --seed 2024 \
    --resume \
    --resume_from epoch_2 2>&1 | head -100

# Verify correct checkpoint was loaded
```

### Validation
- [ ] Resume from specific epoch works
- [ ] Correct checkpoint file is identified
- [ ] Training continues from epoch 3 (not epoch 2)
- [ ] No errors about missing checkpoint

---

## Test 7: Error Handling - Missing Checkpoint

### Objective
Verify graceful error handling when checkpoint doesn't exist.

### Steps
```bash
cd /home/ubuntu/Documents/TOTNet/src

# Try to resume non-existent model
python main.py \
    --saved_fn nonexistent_model \
    --model_choice tracknet \
    --num_epochs 3 \
    --batch_size 4 \
    --num_workers 2 \
    --num_samples 10 \
    --seed 2024 \
    --resume 2>&1 | head -50
```

### Validation
- [ ] No crash or exception
- [ ] Warning message displayed
- [ ] Training starts from epoch 1
- [ ] Training proceeds normally

---

## Test 8: Error Handling - Invalid Checkpoint

### Objective
Verify handling of corrupted checkpoint files.

### Steps
```bash
cd /home/ubuntu/Documents/TOTNet/src

# Create a corrupted checkpoint
echo "corrupted data" > ../checkpoints/test_resume_latest/corrupted.pth

# Try to resume from corrupted checkpoint
python main.py \
    --saved_fn test_resume_latest \
    --model_choice tracknet \
    --num_epochs 3 \
    --batch_size 4 \
    --num_workers 2 \
    --num_samples 10 \
    --seed 2024 \
    --resume \
    --resume_from corrupted 2>&1 | head -50

# Clean up
rm ../checkpoints/test_resume_latest/corrupted.pth
```

### Validation
- [ ] Error is caught and displayed
- [ ] No crash
- [ ] Training continues or starts fresh (depends on implementation)

---

## Test 9: Model Weights Restoration

### Objective
Verify model weights are correctly restored.

### Steps
```bash
cd /home/ubuntu/Documents/TOTNet/src

# Create a simple test script to verify weights
python << 'EOF'
import torch
import sys
sys.path.insert(0, '.')

from model import Model_Loader
from config.config import parse_configs

# Mock configs for testing
class MockConfig:
    model_choice = 'tracknet'
    device = torch.device('cpu')
    num_channels = 64
    # Add other required attributes as needed

# Load model
try:
    config = MockConfig()
    model = Model_Loader(config).load_model()
    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Get initial weights
    initial_weights = {name: param.clone() for name, param in model.named_parameters()}
    print(f"✓ Initial weights captured ({len(initial_weights)} parameters)")
    
except Exception as e:
    print(f"✗ Error: {e}")
EOF
```

### Validation
- [ ] Model loads successfully
- [ ] Weights can be captured
- [ ] Weights can be restored from checkpoint

---

## Test 10: Distributed Training Resume

### Objective
Verify resume works with distributed training.

### Steps
```bash
cd /home/ubuntu/Documents/TOTNet/src

# Note: Requires multiple GPUs to test fully
# Single GPU simulation:

# Start distributed training
torchrun --nproc_per_node=1 main.py \
    --saved_fn test_distributed_resume \
    --model_choice tracknet \
    --num_epochs 2 \
    --batch_size 4 \
    --num_workers 2 \
    --num_samples 10 \
    --seed 2024 2>&1 | tail -50

# Resume distributed training
torchrun --nproc_per_node=1 main.py \
    --saved_fn test_distributed_resume \
    --model_choice tracknet \
    --num_epochs 4 \
    --batch_size 4 \
    --num_workers 2 \
    --num_samples 10 \
    --seed 2024 \
    --resume 2>&1 | tail -50
```

### Validation
- [ ] Distributed training starts successfully
- [ ] Distributed resume works without error
- [ ] All processes load same checkpoint
- [ ] Training continues correctly

---

## Test 11: Checkpoint Metadata Verification

### Objective
Verify that epoch, loss, and counters are correctly restored.

### Steps
```bash
cd /home/ubuntu/Documents/TOTNet/src

# Run training for a few epochs
python main.py \
    --saved_fn test_metadata \
    --model_choice tracknet \
    --num_epochs 3 \
    --batch_size 4 \
    --num_workers 2 \
    --num_samples 10 \
    --seed 2024 2>&1 | grep -i "epoch\|best_val_loss"

# Check checkpoint metadata
python << 'EOF'
import torch

for epoch in [1, 2, 3]:
    checkpoint_path = f'../checkpoints/test_metadata/test_metadata_epoch_{epoch}.pth'
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        print(f"Epoch {epoch} checkpoint:")
        print(f"  - Saved epoch: {ckpt['epoch']}")
        print(f"  - Best val loss: {ckpt['best_val_loss']:.6f}")
        print(f"  - Early stop count: {ckpt['earlystop_count']}")
    except FileNotFoundError:
        print(f"Checkpoint for epoch {epoch} not found")
    except Exception as e:
        print(f"Error loading epoch {epoch}: {e}")
EOF
```

### Validation
- [ ] Checkpoint epoch matches saved epoch
- [ ] Best validation loss is monotonically decreasing (or starts at inf)
- [ ] Early stop count increases correctly

---

## Test 12: Checkpoint Storage and Cleanup

### Objective
Verify checkpoint management and cleanup.

### Steps
```bash
cd /home/ubuntu/Documents/TOTNet

# Check total checkpoint storage
du -sh checkpoints/*/

# Count checkpoints per model
for model_dir in checkpoints/*/; do
    count=$(ls "$model_dir"*.pth 2>/dev/null | wc -l)
    size=$(du -sh "$model_dir" | cut -f1)
    echo "$(basename $model_dir): $count checkpoints, $size total"
done

# Optional: Clean up old periodic checkpoints
# This removes all epoch_* checkpoints but keeps best.pth
for checkpoint_dir in checkpoints/*/; do
    echo "Cleaning up old checkpoints in $checkpoint_dir"
    find "$checkpoint_dir" -name "*_epoch_*.pth" -type f -delete
done
```

### Validation
- [ ] Checkpoint storage is reasonable (< 10 GB for typical models)
- [ ] Best checkpoint is preserved after cleanup
- [ ] Can safely remove old periodic checkpoints

---

## Test Results Template

```
Date: ____________________
Tester: __________________

╔════════════════════════════════════════════════════════════╗
║           Resume Functionality Test Results                ║
╚════════════════════════════════════════════════════════════╝

Test 1: Configuration Arguments
  Status: [ ] PASS  [ ] FAIL  [ ] SKIP
  Notes: _________________________________________________

Test 2: Checkpoint File Creation
  Status: [ ] PASS  [ ] FAIL  [ ] SKIP
  Notes: _________________________________________________

Test 3: Checkpoint Content Verification
  Status: [ ] PASS  [ ] FAIL  [ ] SKIP
  Notes: _________________________________________________

Test 4: Resume - Latest Checkpoint
  Status: [ ] PASS  [ ] FAIL  [ ] SKIP
  Notes: _________________________________________________

Test 5: Resume - Best Checkpoint
  Status: [ ] PASS  [ ] FAIL  [ ] SKIP
  Notes: _________________________________________________

Test 6: Resume - Specific Epoch
  Status: [ ] PASS  [ ] FAIL  [ ] SKIP
  Notes: _________________________________________________

Test 7: Error Handling - Missing Checkpoint
  Status: [ ] PASS  [ ] FAIL  [ ] SKIP
  Notes: _________________________________________________

Test 8: Error Handling - Invalid Checkpoint
  Status: [ ] PASS  [ ] FAIL  [ ] SKIP
  Notes: _________________________________________________

Test 9: Model Weights Restoration
  Status: [ ] PASS  [ ] FAIL  [ ] SKIP
  Notes: _________________________________________________

Test 10: Distributed Training Resume
  Status: [ ] PASS  [ ] FAIL  [ ] SKIP
  Notes: _________________________________________________

Test 11: Checkpoint Metadata Verification
  Status: [ ] PASS  [ ] FAIL  [ ] SKIP
  Notes: _________________________________________________

Test 12: Checkpoint Storage and Cleanup
  Status: [ ] PASS  [ ] FAIL  [ ] SKIP
  Notes: _________________________________________________

═════════════════════════════════════════════════════════════

Overall Status: [ ] ALL PASS  [ ] SOME FAIL  [ ] ISSUES FOUND

Issues Found:
_________________________________________________________________

Recommendations:
_________________________________________________________________

```

---

## Automated Test Script

```bash
#!/bin/bash
# Automated resume functionality test

cd /home/ubuntu/Documents/TOTNet/src

echo "Running Resume Functionality Tests..."
echo "======================================"

# Test 1: Help argument
echo -n "Test 1: Help includes --resume... "
if python main.py --help | grep -q "\-\-resume"; then
    echo "✓ PASS"
else
    echo "✗ FAIL"
fi

# Test 2: Create checkpoints
echo -n "Test 2: Checkpoints are created... "
python main.py \
    --saved_fn test_auto \
    --model_choice tracknet \
    --num_epochs 1 \
    --batch_size 4 \
    --num_workers 2 \
    --num_samples 5 2>/dev/null

if [ -f ../checkpoints/test_auto/test_auto_epoch_1.pth ]; then
    echo "✓ PASS"
else
    echo "✗ FAIL"
fi

# Test 3: Resume function
echo -n "Test 3: Resume is recognized... "
if python main.py --saved_fn test_auto --model_choice tracknet --num_epochs 2 --resume --batch_size 4 2>&1 | grep -q "checkpoint\|resume\|Resuming"; then
    echo "✓ PASS"
else
    echo "✗ FAIL"
fi

echo "======================================"
echo "Tests Complete!"
```

---

## Continuous Integration Recommendation

Add to your CI/CD pipeline:

```yaml
# .github/workflows/test-resume.yml
name: Test Resume Functionality

on: [push, pull_request]

jobs:
  test-resume:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - run: pip install -r requirements.txt
      - run: bash test_resume.sh
```
