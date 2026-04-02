#!/bin/bash
# Example scripts for resuming training with TOTNet

echo "=========================================="
echo "TOTNet Resume Training Examples"
echo "=========================================="
echo ""

# Example 1: Resume from latest checkpoint
echo "Example 1: Resume from latest checkpoint"
echo "Command:"
echo "python main.py \\"
echo "    --saved_fn my_model \\"
echo "    --model_choice tracknet \\"
echo "    --num_epochs 100 \\"
echo "    --batch_size 8 \\"
echo "    --resume"
echo ""

# Example 2: Resume from best checkpoint
echo "Example 2: Resume from best checkpoint"
echo "Command:"
echo "python main.py \\"
echo "    --saved_fn my_model \\"
echo "    --model_choice tracknet \\"
echo "    --num_epochs 100 \\"
echo "    --batch_size 8 \\"
echo "    --resume \\"
echo "    --resume_from best"
echo ""

# Example 3: Resume from specific epoch
echo "Example 3: Resume from specific epoch (epoch 15)"
echo "Command:"
echo "python main.py \\"
echo "    --saved_fn my_model \\"
echo "    --model_choice tracknet \\"
echo "    --num_epochs 100 \\"
echo "    --batch_size 8 \\"
echo "    --resume \\"
echo "    --resume_from epoch_15"
echo ""

# Example 4: Resume from best checkpoint with distributed training
echo "Example 4: Resume with distributed training (4 GPUs)"
echo "Command:"
echo "torchrun --nproc_per_node=4 main.py \\"
echo "    --saved_fn my_model \\"
echo "    --model_choice tracknet \\"
echo "    --num_epochs 100 \\"
echo "    --batch_size 32 \\"
echo "    --resume \\"
echo "    --resume_from best"
echo ""

# Example 5: List available checkpoints
echo "Example 5: List available checkpoints"
echo "Command:"
echo "ls -lh checkpoints/my_model/"
echo ""

# Example 6: Check checkpoint metadata
echo "Example 6: Check checkpoint metadata"
echo "Command:"
echo "python -c \"import torch; ckpt = torch.load('checkpoints/my_model/my_model_best.pth'); print(f'Epoch: {ckpt[\\\"epoch\\\"]}, Best Loss: {ckpt[\\\"best_val_loss\\\"]:.4f}')\""
echo ""

echo "=========================================="
echo "Tips for Resuming Training:"
echo "=========================================="
echo "1. Use the same batch size and model configuration as the original training"
echo "2. The resume mechanism preserves: model weights, optimizer state, LR scheduler, epoch count"
echo "3. For distributed training, use the same number of GPUs and torchrun configuration"
echo "4. Check that checkpoint files exist in the checkpoints/{saved_fn}/ directory"
echo "5. Use --resume_from best to continue from the best validation loss checkpoint"
echo "6. Use --resume_from epoch_X to continue from a specific epoch"
echo ""
