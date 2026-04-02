"""
Utility functions for resuming training from checkpoints.
"""

import os
import torch
import time


def get_checkpoint_path(checkpoints_dir, saved_fn, resume_from=None):
    """
    Get the checkpoint file path to resume from.
    
    Args:
        checkpoints_dir (str): Directory containing checkpoints
        saved_fn (str): Base filename for checkpoints
        resume_from (str, optional): Specific checkpoint to resume from.
                                    Can be:
                                    - None: latest checkpoint
                                    - 'best': best checkpoint
                                    - 'epoch_X': specific epoch
                                    - filename: specific checkpoint file
    
    Returns:
        str: Full path to checkpoint file, or None if not found
    """
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoint directory does not exist: {checkpoints_dir}")
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoints_dir}")
        return None
    
    # If resume_from is specified
    if resume_from is not None:
        if resume_from == 'best':
            checkpoint_file = f'{saved_fn}_best.pth'
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                print(f"Using best checkpoint: {checkpoint_path}")
                return checkpoint_path
            else:
                print(f"Best checkpoint not found: {checkpoint_path}")
                return None
        elif resume_from.startswith('epoch_'):
            checkpoint_file = f'{saved_fn}_{resume_from}.pth'
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                print(f"Using checkpoint from {resume_from}: {checkpoint_path}")
                return checkpoint_path
            else:
                print(f"Checkpoint for {resume_from} not found: {checkpoint_path}")
                return None
        else:
            # Assume it's a filename
            checkpoint_path = os.path.join(checkpoints_dir, resume_from)
            if os.path.exists(checkpoint_path):
                print(f"Using checkpoint: {checkpoint_path}")
                return checkpoint_path
            else:
                print(f"Checkpoint file not found: {checkpoint_path}")
                return None
    
    # Find the latest checkpoint (excluding best)
    regular_checkpoints = [f for f in checkpoint_files if '_best' not in f]
    if regular_checkpoints:
        # Sort by epoch number
        def extract_epoch(filename):
            try:
                # Extract epoch number from filename like 'model_epoch_10.pth'
                epoch_str = filename.split('_epoch_')[-1].replace('.pth', '')
                return int(epoch_str)
            except (ValueError, IndexError):
                return 0
        
        latest_checkpoint = max(regular_checkpoints, key=extract_epoch)
    else:
        # Fall back to best checkpoint if no regular checkpoints found
        if any('_best' in f for f in checkpoint_files):
            latest_checkpoint = f'{saved_fn}_best.pth'
        else:
            latest_checkpoint = max(checkpoint_files)
    
    checkpoint_path = os.path.join(checkpoints_dir, latest_checkpoint)
    print(f"Using latest checkpoint: {checkpoint_path}")
    return checkpoint_path if os.path.exists(checkpoint_path) else None


def resume_checkpoint(model, optimizer, lr_scheduler, checkpoint_path, device, configs=None):
    """
    Resume training from a checkpoint.
    
    Args:
        model (torch.nn.Module): Model to load checkpoint into
        optimizer (torch.optim.Optimizer): Optimizer to restore state
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        checkpoint_path (str): Path to checkpoint file
        device (torch.device): Device to load checkpoint to
        configs (object, optional): Configuration object to update with checkpoint values
    
    Returns:
        tuple: (epoch, best_val_loss, earlystop_count) - Information from checkpoint
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return 0, float('inf'), 0
    
    print(f"\n{'='*60}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"{'='*60}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0, float('inf'), 0
    
    # Load model state dict
    try:
        if hasattr(model, 'module'):
            # Handle DataParallel models
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        print("✓ Model state loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model state: {e}")
        return 0, float('inf'), 0
    
    # Load optimizer state
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("✓ Optimizer state loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not load optimizer state: {e}")
    
    # Load learning rate scheduler state
    try:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print("✓ Learning rate scheduler state loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not load lr_scheduler state: {e}")
    
    # Extract training information
    epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    earlystop_count = checkpoint.get('earlystop_count', 0)
    
    print(f"\n✓ Checkpoint Summary:")
    print(f"  - Last saved epoch: {epoch}")
    print(f"  - Best validation loss: {best_val_loss:.6f}")
    print(f"  - Early stopping count: {earlystop_count}")
    print(f"{'='*60}\n")
    
    return epoch, best_val_loss, earlystop_count


def check_for_existing_checkpoints(logs_dir, checkpoints_dir, saved_fn):
    """
    Check if logs and checkpoints exist for the given model.
    This helps determine if training should be resumed or started fresh.
    
    Args:
        logs_dir (str): Path to logs directory
        checkpoints_dir (str): Path to checkpoints directory
        saved_fn (str): Base filename for checkpoints
    
    Returns:
        dict: Contains:
            - 'has_logs': bool - Whether logs directory has content
            - 'has_checkpoints': bool - Whether checkpoints exist
            - 'checkpoint_path': str or None - Path to latest checkpoint if exists
            - 'num_checkpoints': int - Number of checkpoints found
            - 'latest_epoch': int - Latest epoch from checkpoints, or -1 if none
    """
    result = {
        'has_logs': False,
        'has_checkpoints': False,
        'checkpoint_path': None,
        'num_checkpoints': 0,
        'latest_epoch': -1,
    }
    
    # Check for existing logs
    if os.path.exists(logs_dir):
        log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log') or f.endswith('.txt')]
        if log_files:
            result['has_logs'] = True
    
    # Check for existing checkpoints
    if os.path.exists(checkpoints_dir):
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
        result['num_checkpoints'] = len(checkpoint_files)
        
        if checkpoint_files:
            result['has_checkpoints'] = True
            
            # Find the latest checkpoint
            checkpoint_path = get_checkpoint_path(checkpoints_dir, saved_fn, resume_from=None)
            if checkpoint_path:
                result['checkpoint_path'] = checkpoint_path
                
                # Extract epoch number
                try:
                    epoch_str = os.path.basename(checkpoint_path).split('_epoch_')[-1].replace('.pth', '')
                    result['latest_epoch'] = int(epoch_str)
                except (ValueError, IndexError):
                    if '_best' in os.path.basename(checkpoint_path):
                        result['latest_epoch'] = 0  # Best checkpoint, epoch unknown
    
    return result


def should_resume_training(logs_dir, checkpoints_dir, saved_fn):
    """
    Determine if training should be resumed based on existing logs and checkpoints.
    
    Args:
        logs_dir (str): Path to logs directory
        checkpoints_dir (str): Path to checkpoints directory
        saved_fn (str): Base filename for checkpoints
    
    Returns:
        tuple: (should_resume: bool, checkpoint_path: str or None, info_message: str)
    """
    check_result = check_for_existing_checkpoints(logs_dir, checkpoints_dir, saved_fn)
    
    should_resume = check_result['has_checkpoints']
    checkpoint_path = check_result['checkpoint_path']
    
    # Create informative message
    message = "\n" + "="*70 + "\n"
    message += "CHECKPOINT DETECTION REPORT\n"
    message += "="*70 + "\n"
    
    if check_result['has_logs']:
        message += f"✓ Existing logs found in: {logs_dir}\n"
    else:
        message += f"✗ No existing logs found\n"
    
    if check_result['has_checkpoints']:
        message += f"✓ Found {check_result['num_checkpoints']} checkpoint(s)\n"
        if check_result['latest_epoch'] >= 0:
            message += f"✓ Latest epoch: {check_result['latest_epoch']}\n"
        if checkpoint_path:
            message += f"✓ Will resume from: {os.path.basename(checkpoint_path)}\n"
    else:
        message += f"✗ No checkpoints found\n"
        message += "→ Starting training from scratch\n"
    
    message += "="*70 + "\n"
    
    return should_resume, checkpoint_path, message
