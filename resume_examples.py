"""
Practical examples for resuming training with TOTNet.
This script demonstrates different resume scenarios.
"""

import os
import subprocess
import sys


def run_command(command, description=""):
    """Run a shell command and display output."""
    if description:
        print(f"\n{'='*70}")
        print(f"  {description}")
        print(f"{'='*70}\n")
    
    print(f"Command: {command}\n")
    try:
        result = subprocess.run(command, shell=True, capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def example_1_initial_training():
    """Example 1: Initial training run."""
    command = """
    cd /home/ubuntu/Documents/TOTNet/src && python main.py \
        --saved_fn TOTNet_test \
        --model_choice TOTNet \
        --num_epochs 50 \
        --batch_size 8 \
        --num_workers 4 \
        --seed 2024
    """
    run_command(command, "Example 1: Initial Training Run")


def example_2_resume_latest():
    """Example 2: Resume from latest checkpoint."""
    command = """
    cd /home/ubuntu/Documents/TOTNet/src && python main.py \
        --saved_fn TOTNet_test \
        --model_choice TOTNet \
        --num_epochs 50 \
        --batch_size 8 \
        --num_workers 4 \
        --resume
    """
    run_command(command, "Example 2: Resume from Latest Checkpoint")


def example_3_resume_best():
    """Example 3: Resume from best checkpoint."""
    command = """
    cd /home/ubuntu/Documents/TOTNet/src && python main.py \
        --saved_fn TOTNet_test \
        --model_choice TOTNet \
        --num_epochs 50 \
        --batch_size 8 \
        --num_workers 4 \
        --resume \
        --resume_from best
    """
    run_command(command, "Example 3: Resume from Best Checkpoint")


def example_4_resume_specific_epoch():
    """Example 4: Resume from specific epoch."""
    command = """
    cd /home/ubuntu/Documents/TOTNet/src && python main.py \
        --saved_fn TOTNet_test \
        --model_choice TOTNet \
        --num_epochs 50 \
        --batch_size 8 \
        --num_workers 4 \
        --resume \
        --resume_from epoch_25
    """
    run_command(command, "Example 4: Resume from Specific Epoch (Epoch 25)")


def example_5_distributed_training():
    """Example 5: Resume with distributed training (4 GPUs)."""
    command = """
    cd /home/ubuntu/Documents/TOTNet/src && torchrun --nproc_per_node=4 main.py \
        --saved_fn TOTNet_distributed \
        --model_choice TOTNet \
        --num_epochs 50 \
        --batch_size 32 \
        --num_workers 4 \
        --resume
    """
    run_command(command, "Example 5: Resume with Distributed Training (4 GPUs)")


def list_checkpoints():
    """List all available checkpoints."""
    checkpoints_dir = "/home/ubuntu/Documents/TOTNet/checkpoints"
    
    print(f"\n{'='*70}")
    print("  Available Checkpoints in Directory")
    print(f"{'='*70}\n")
    
    if os.path.exists(checkpoints_dir):
        for model_dir in os.listdir(checkpoints_dir):
            model_path = os.path.join(checkpoints_dir, model_dir)
            if os.path.isdir(model_path):
                print(f"\nModel: {model_dir}")
                print("-" * 70)
                checkpoint_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
                if checkpoint_files:
                    for checkpoint in sorted(checkpoint_files):
                        file_path = os.path.join(model_path, checkpoint)
                        size = os.path.getsize(file_path) / (1024**2)  # Convert to MB
                        print(f"  - {checkpoint:<50} ({size:.2f} MB)")
                else:
                    print("  No checkpoints found")
    else:
        print(f"Checkpoints directory not found: {checkpoints_dir}")


def check_checkpoint_metadata():
    """Check metadata of a specific checkpoint."""
    import torch
    
    print(f"\n{'='*70}")
    print("  Checkpoint Metadata Inspection")
    print(f"{'='*70}\n")
    
    checkpoints_dir = "/home/ubuntu/Documents/TOTNet/checkpoints"
    
    # Find a checkpoint file
    for model_dir in os.listdir(checkpoints_dir):
        model_path = os.path.join(checkpoints_dir, model_dir)
        if os.path.isdir(model_path):
            checkpoint_files = [f for f in os.listdir(model_path) if '_best.pth' in f]
            if checkpoint_files:
                checkpoint_path = os.path.join(model_path, checkpoint_files[0])
                print(f"Checkpoint: {checkpoint_path}\n")
                
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    
                    print("Checkpoint Contents:")
                    print("-" * 70)
                    
                    if 'epoch' in checkpoint:
                        print(f"  Epoch: {checkpoint['epoch']}")
                    
                    if 'best_val_loss' in checkpoint:
                        print(f"  Best Validation Loss: {checkpoint['best_val_loss']:.6f}")
                    
                    if 'earlystop_count' in checkpoint:
                        print(f"  Early Stop Count: {checkpoint['earlystop_count']}")
                    
                    if 'state_dict' in checkpoint:
                        print(f"  Model Parameters: {len(checkpoint['state_dict'])}")
                    
                    if 'optimizer' in checkpoint:
                        print(f"  Optimizer State Keys: {list(checkpoint['optimizer'].keys())}")
                    
                    if 'lr_scheduler' in checkpoint:
                        print(f"  LR Scheduler State Keys: {list(checkpoint['lr_scheduler'].keys())}")
                    
                    print("\nAll Keys in Checkpoint:")
                    for key in checkpoint.keys():
                        print(f"  - {key}")
                    
                except Exception as e:
                    print(f"Error loading checkpoint: {e}")
                
                return
    
    print("No checkpoints found to inspect")


def show_usage_tips():
    """Display usage tips for resume functionality."""
    print(f"\n{'='*70}")
    print("  Resume Training - Quick Reference")
    print(f"{'='*70}\n")
    
    tips = """
RESUME OPTIONS:
1. Resume from Latest Checkpoint (Recommended)
   $ python main.py --saved_fn model_name --model_choice tracknet --resume

2. Resume from Best Checkpoint
   $ python main.py --saved_fn model_name --model_choice tracknet \\
     --resume --resume_from best

3. Resume from Specific Epoch
   $ python main.py --saved_fn model_name --model_choice tracknet \\
     --resume --resume_from epoch_15

KEY POINTS:
• Use the same --batch_size and model configuration as original training
• --resume flag enables checkpoint loading
• --resume_from specifies which checkpoint to use (optional)
• For distributed training, use torchrun with --resume
• Checkpoints are saved in: checkpoints/{saved_fn}/

CHECKPOINT INFORMATION PRESERVED:
✓ Model weights from the exact state when checkpoint was saved
✓ Optimizer state (momentum, moving averages, etc.)
✓ Learning rate scheduler state
✓ Epoch number (training continues from next epoch)
✓ Best validation loss
✓ Early stopping counter

USEFUL COMMANDS:
• List checkpoints:
  ls -lh checkpoints/model_name/

• Check checkpoint details:
  python -c "import torch; \\
  ckpt = torch.load('checkpoints/model/model_best.pth'); \\
  print(f'Epoch: {ckpt[\"epoch\"]}, Best Loss: {ckpt[\"best_val_loss\"]:.4f}')"

• Remove old checkpoints (keep only best):
  rm checkpoints/model_name/*_epoch_*.pth

TROUBLESHOOTING:
• "Checkpoint not found" → Check saved_fn matches the directory name
• "Size mismatch" → Ensure model configuration matches original training
• Training starts from epoch 1 → Add --resume flag
• Lost best model → Use --resume_from best if best.pth exists
    """
    
    print(tips)


def main():
    """Main function to run examples."""
    print("\n" + "="*70)
    print("  TOTNet Resume Training - Practical Examples")
    print("="*70)
    
    print("\nThis script demonstrates how to resume training with TOTNet.")
    print("You can:")
    print("  1. View available checkpoints")
    print("  2. Check checkpoint metadata")
    print("  3. See usage tips")
    print("  4. Run example commands (commented out for safety)")
    
    # Show available checkpoints
    list_checkpoints()
    
    # Show checkpoint metadata if available
    try:
        check_checkpoint_metadata()
    except Exception as e:
        print(f"\nNote: Could not inspect checkpoints: {e}")
    
    # Show usage tips
    show_usage_tips()
    
    print("\n" + "="*70)
    print("  Example Commands (Uncomment in script to run)")
    print("="*70)
    
    print("""
# Example 1: Initial training run
# example_1_initial_training()

# Example 2: Resume from latest checkpoint
# example_2_resume_latest()

# Example 3: Resume from best checkpoint
# example_3_resume_best()

# Example 4: Resume from specific epoch
# example_4_resume_specific_epoch()

# Example 5: Distributed training with resume
# example_5_distributed_training()
    """)
    
    print("\n" + "="*70)
    print("  Quick Start Guide")
    print("="*70)
    
    quick_start = """
Step 1: Start initial training
  cd /home/ubuntu/Documents/TOTNet/src
  python main.py --saved_fn my_model --model_choice TOTNet --num_epochs 100

Step 2: If interrupted, resume training with:
  python main.py --saved_fn my_model --model_choice TOTNet --resume

Step 3: To resume from best checkpoint:
  python main.py --saved_fn my_model --model_choice TOTNet --resume --resume_from best

Step 4: Check available checkpoints:
  ls -lh ../checkpoints/my_model/

That's it! The resume mechanism handles all the complexity internally.
    """
    
    print(quick_start)


if __name__ == "__main__":
    main()
