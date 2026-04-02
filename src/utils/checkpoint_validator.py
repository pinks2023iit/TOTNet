"""
Checkpoint validation utilities to detect interrupted training sessions.
"""

import os
import time


def validate_checkpoint_sequence(checkpoints_dir, saved_fn, max_time_gap_minutes=60):
    """
    Validate that checkpoints form a continuous sequence based on timestamps.
    This prevents resuming from stale checkpoints when training was interrupted.
    
    Args:
        checkpoints_dir (str): Directory containing checkpoints
        saved_fn (str): Base filename for checkpoints
        max_time_gap_minutes (int): Maximum allowed time gap between consecutive checkpoints
                                   If gap is larger, indicates incomplete/interrupted training
    
    Returns:
        dict: Contains:
            - 'is_valid': bool - Whether checkpoint sequence is valid
            - 'valid_checkpoint_path': str or None - Latest valid checkpoint
            - 'valid_epoch': int - Latest valid epoch
            - 'warning': str - Warning message if any issues found
            - 'gap_detected': bool - Whether a large time gap was detected
    """
    result = {
        'is_valid': False,
        'valid_checkpoint_path': None,
        'valid_epoch': -1,
        'warning': '',
        'gap_detected': False,
    }
    
    if not os.path.exists(checkpoints_dir):
        result['warning'] = f"Checkpoint directory does not exist: {checkpoints_dir}"
        return result
    
    # Get all checkpoint files with their modification times
    checkpoint_files = {}
    for f in os.listdir(checkpoints_dir):
        if f.endswith('.pth') and '_best' not in f:  # Exclude best checkpoint for sequence validation
            filepath = os.path.join(checkpoints_dir, f)
            try:
                # Extract epoch number
                epoch_str = f.split('_epoch_')[-1].replace('.pth', '')
                epoch = int(epoch_str)
                mod_time = os.path.getmtime(filepath)
                checkpoint_files[epoch] = {
                    'path': filepath,
                    'mtime': mod_time,
                    'filename': f
                }
            except (ValueError, IndexError):
                continue
    
    if not checkpoint_files:
        result['warning'] = "No valid epoch checkpoints found"
        return result
    
    # Sort by epoch
    sorted_epochs = sorted(checkpoint_files.keys())
    
    # Analyze the checkpoint pattern
    latest_valid_epoch = sorted_epochs[0]
    
    for i, epoch in enumerate(sorted_epochs):
        current_time = checkpoint_files[epoch]['mtime']
        
        if i > 0:
            prev_epoch = sorted_epochs[i - 1]
            prev_time = checkpoint_files[prev_epoch]['mtime']
            time_diff = current_time - prev_time
            
            # If there's a large time gap, suspect interrupted training
            if time_diff > (max_time_gap_minutes * 60):
                result['gap_detected'] = True
                result['warning'] = (
                    f"⚠ Large time gap detected between epoch {prev_epoch} and {epoch}: "
                    f"{time_diff/60:.1f} minutes. Training may have been interrupted.\n"
                    f"→ Resuming from epoch {prev_epoch} instead of latest checkpoint."
                )
                latest_valid_epoch = prev_epoch
                break
        
        latest_valid_epoch = epoch
    
    if latest_valid_epoch >= 0:
        result['is_valid'] = True
        result['valid_epoch'] = latest_valid_epoch
        result['valid_checkpoint_path'] = checkpoint_files[latest_valid_epoch]['path']
    
    return result


def find_safe_checkpoint(checkpoints_dir, saved_fn, max_time_gap_minutes=60):
    """
    Find the most recent "safe" checkpoint based on continuous time progression.
    
    Args:
        checkpoints_dir (str): Directory containing checkpoints
        saved_fn (str): Base filename for checkpoints
        max_time_gap_minutes (int): Maximum allowed time gap between consecutive checkpoints
    
    Returns:
        tuple: (checkpoint_path, epoch, is_safe)
            - checkpoint_path: Path to safe checkpoint or None
            - epoch: Epoch number of safe checkpoint or -1
            - is_safe: Whether the checkpoint is considered safe to resume from
    """
    validation_result = validate_checkpoint_sequence(checkpoints_dir, saved_fn, max_time_gap_minutes)
    
    if validation_result['is_valid']:
        return (
            validation_result['valid_checkpoint_path'],
            validation_result['valid_epoch'],
            not validation_result['gap_detected']
        )
    else:
        return None, -1, False


def get_checkpoint_creation_time(checkpoint_path):
    """Get creation time of a checkpoint."""
    if os.path.exists(checkpoint_path):
        return os.path.getmtime(checkpoint_path)
    return None


def check_checkpoint_freshness(checkpoint_path, max_age_hours=24):
    """
    Check if a checkpoint is "fresh" (recently created).
    
    Args:
        checkpoint_path (str): Path to checkpoint
        max_age_hours (int): Maximum age in hours for a "fresh" checkpoint
    
    Returns:
        bool: True if checkpoint is fresh, False otherwise
    """
    if not os.path.exists(checkpoint_path):
        return False
    
    mod_time = os.path.getmtime(checkpoint_path)
    current_time = time.time()
    age_hours = (current_time - mod_time) / 3600
    
    return age_hours <= max_age_hours
