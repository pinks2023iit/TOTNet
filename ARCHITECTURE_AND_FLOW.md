# TOTNet Resume Training - Architecture & Flow

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TOTNet Training System                        │
└─────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  main.py (Entry Point)                                    │
│  ├─ Parses arguments (--resume, --resume_from)           │
│  ├─ Initializes distributed settings (if needed)         │
│  └─ Calls main_worker()                                  │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│  main_worker() Function                                   │
│  ├─ Creates model and moves to device                    │
│  ├─ Makes model data parallel                            │
│  ├─ Creates optimizer and scheduler                      │
│  ├─ [NEW] Resume checkpoint if --resume flag set        │
│  └─ Starts training loop                                 │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  Is --resume set?    │
                └──────────────────────┘
                    │              │
                   YES             NO
                    │              │
                    ▼              ▼
        ┌───────────────────┐  ┌──────────────────┐
        │ Resume Module     │  │ Start Fresh      │
        │ (resume_utils.py) │  │ epoch = 1        │
        └───────────────────┘  └──────────────────┘
                    │
         ┌──────────┼──────────┐
         │          │          │
         ▼          ▼          ▼
    ┌────────┐ ┌────────┐ ┌─────────────┐
    │ Latest │ │ Best   │ │ Specific    │
    │ (auto) │ │ (best) │ │ (epoch_X)   │
    └────────┘ └────────┘ └─────────────┘
         │          │          │
         └──────────┼──────────┘
                    │
                    ▼
    ┌───────────────────────────────────┐
    │  get_checkpoint_path()            │
    │  Finds checkpoint file            │
    └───────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────┐
    │  resume_checkpoint()              │
    │  ├─ Loads model state_dict       │
    │  ├─ Restores optimizer state     │
    │  ├─ Restores scheduler state     │
    │  └─ Returns: epoch, best_loss    │
    └───────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────┐
    │  Training continues from          │
    │  epoch + 1                        │
    └───────────────────────────────────┘
```

## Data Flow: Checkpoint Save & Load

```
CHECKPOINT SAVE (During Training)
═══════════════════════════════════════

End of Epoch N:
   │
   ├─ Model weights (state_dict)
   ├─ Optimizer state (momentum, lr, etc.)
   ├─ LR Scheduler state
   ├─ Epoch number
   ├─ Best validation loss
   └─ Early stopping counter
   │
   ▼
┌─────────────────────────────────────┐
│  get_saved_state()                  │
│  Creates checkpoint dictionary      │
└─────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────┐
│  save_checkpoint()                  │
│  Saves to:                          │
│  - model_best.pth (if best)         │
│  - model_epoch_N.pth (periodic)     │
└─────────────────────────────────────┘


CHECKPOINT LOAD (Resume Training)
════════════════════════════════════

Checkpoint File (model_best.pth or model_epoch_N.pth):
   │
   ├─ state_dict (model weights)
   ├─ optimizer (optimizer state)
   ├─ lr_scheduler (scheduler state)
   ├─ epoch (training epoch)
   ├─ best_val_loss (best loss so far)
   └─ earlystop_count (early stop progress)
   │
   ▼
┌─────────────────────────────────────┐
│  get_checkpoint_path()              │
│  Locates checkpoint file            │
└─────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────┐
│  torch.load()                       │
│  Loads checkpoint into memory       │
└─────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────┐
│  resume_checkpoint()                │
├─ model.load_state_dict()            │
├─ optimizer.load_state_dict()        │
├─ lr_scheduler.load_state_dict()     │
│  Returns: new_epoch, best_loss,     │
│           earlystop_count           │
└─────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────┐
│  Training continues from epoch + 1  │
└─────────────────────────────────────┘
```

## Resume Decision Tree

```
                      ┌─────────────┐
                      │  --resume?  │
                      └─────────────┘
                          │
                 ┌────────┴────────┐
                 │                 │
                NO                YES
                 │                 │
                 ▼                 ▼
           ┌──────────┐      ┌─────────────────┐
           │Start     │      │ --resume_from?  │
           │Fresh     │      └─────────────────┘
           │epoch=1   │          │
           └──────────┘    ┌─────┼─────┐
                           │     │     │
                        None   best   epoch_X
                           │     │     │
              ┌─────────────┘     │     └──────────────┐
              │                   │                    │
              ▼                   ▼                    ▼
        ┌──────────┐        ┌──────────┐      ┌────────────────┐
        │Find      │        │Find      │      │Find Specific   │
        │Latest    │        │Best      │      │Epoch           │
        │Checkpoint│        │Checkpoint│      │Checkpoint      │
        └──────────┘        └──────────┘      └────────────────┘
              │                   │                    │
              └─────────────────┬─────────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │Checkpoint Found? │
                        └──────────────────┘
                            │        │
                           YES      NO
                            │        │
                            ▼        ▼
                    ┌─────────────┐ ┌──────────────┐
                    │Load & Resume│ │Start Fresh   │
                    │from next    │ │epoch=1       │
                    │epoch        │ │(with warning)│
                    └─────────────┘ └──────────────┘
```

## Module Interaction

```
┌────────────────────────────────────────────────────────────┐
│                     config.py                              │
│  ├─ --resume (action='store_true')                        │
│  └─ --resume_from (type=str)                              │
│                                                            │
│  Accessed in: main.py via configs.resume                 │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                     main.py                                │
│  ├─ Imports: resume_utils                                │
│  ├─ Calls: get_checkpoint_path()                         │
│  ├─ Calls: resume_checkpoint()                           │
│  └─ Updates: configs.start_epoch                         │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                  resume_utils.py (NEW)                     │
│  ├─ get_checkpoint_path()                                │
│  │  ├─ Searches checkpoints directory                    │
│  │  ├─ Handles: latest, best, epoch_X                    │
│  │  └─ Returns: full checkpoint path                     │
│  │                                                        │
│  └─ resume_checkpoint()                                  │
│     ├─ torch.load()                                      │
│     ├─ model.load_state_dict()                           │
│     ├─ optimizer.load_state_dict()                       │
│     ├─ lr_scheduler.load_state_dict()                    │
│     └─ Returns: epoch, best_loss, earlystop_count       │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                   train_utils.py (Existing)                │
│  ├─ save_checkpoint()  ◄─ Called at end of each epoch     │
│  ├─ get_saved_state()  ◄─ Prepares checkpoint data       │
│  └─ create_optimizer() ◄─ Used by resume                 │
└────────────────────────────────────────────────────────────┘
```

## State Transitions

```
Training Lifecycle with Resume:
═════════════════════════════════

START
  │
  ├─ Parse Arguments (--resume, --resume_from)
  │
  ├─ Initialize Model, Optimizer, Scheduler
  │
  ├─ [DECISION] Is --resume set?
  │  │
  │  ├─YES─► Load Checkpoint
  │  │       │
  │  │       ├─ Model State: Training State 1 → Training State N
  │  │       ├─ Optimizer: Paused State → Resumed State
  │  │       ├─ Scheduler: Previous Position → Current Position
  │  │       └─ Epoch: N → Continue from N+1
  │  │
  │  └─NO──► Fresh Start
  │          │
  │          └─ Start from Epoch 1
  │
  ├─ Training Loop (Epoch N+1, N+2, ...)
  │  ├─ Forward Pass
  │  ├─ Backward Pass
  │  ├─ Optimizer Step
  │  └─ End of Epoch: Save Checkpoint
  │
  ├─ [DECISION] Early Stop or Epoch Limit?
  │  ├─YES─► FINISH
  │  └─NO──► Continue to Next Epoch
  │
  └─ END


Checkpoint States:
══════════════════

┌─────────────────────────────────────────┐
│ SAVED STATE (Checkpoint File)           │
│                                         │
│ epoch: 45                               │
│ model_state: {...}  (1000+ layers)     │
│ optimizer: {                            │
│   'state': {...},                       │
│   'param_groups': [...]                 │
│ }                                       │
│ lr_scheduler: {...}                     │
│ best_val_loss: 0.0456                   │
│ earlystop_count: 2                      │
└─────────────────────────────────────────┘
         │
         │ torch.save()
         ▼
   checkpoint.pth (≈500 MB)
         │
         │ torch.load()
         ▼
┌─────────────────────────────────────────┐
│ RESTORED STATE (In-Memory)              │
│                                         │
│ Model weights: Restored to epoch 45     │
│ Optimizer state: Restored with momentum │
│ LR Scheduler: At step N                 │
│ Epoch counter: Set to 46 (next)         │
└─────────────────────────────────────────┘
         │
         │ Training continues
         ▼
    Epoch 46 Training...
```

## Error Handling Flow

```
Resume Process with Error Handling:
═════════════════════════════════════

Try: Load Checkpoint
  │
  ├─ Checkpoint exists?
  │  ├─NO─► Print warning, start fresh (epoch 1)
  │  └─YES─► Continue
  │
  ├─ Checkpoint valid (parseable)?
  │  ├─NO─► Print error, start fresh (epoch 1)
  │  └─YES─► Continue
  │
  ├─ Model state matches?
  │  ├─NO─► Print "Size mismatch", start fresh
  │  └─YES─► Continue
  │
  ├─ Optimizer state loadable?
  │  ├─NO─► Print warning, continue with fresh optimizer
  │  └─YES─► Continue
  │
  ├─ LR Scheduler loadable?
  │  ├─NO─► Print warning, continue with fresh scheduler
  │  └─YES─► Success!
  │
  └─ SUCCESS: Resume from epoch N+1
     with full state restoration
```

## File Organization

```
/home/ubuntu/Documents/TOTNet/
│
├── src/
│   ├── main.py
│   │   └─ [MODIFIED] Added resume logic
│   │
│   ├── config/
│   │   └── config.py
│   │       └─ [MODIFIED] Added --resume, --resume_from args
│   │
│   └── utils/
│       ├── train_utils.py
│       │   └─ [EXISTING] save_checkpoint(), get_saved_state()
│       │
│       └── resume_utils.py
│           └─ [NEW] get_checkpoint_path(), resume_checkpoint()
│
├── checkpoints/
│   └── {saved_fn}/
│       ├── model_best.pth
│       ├── model_epoch_1.pth
│       ├── model_epoch_2.pth
│       └── ...
│
├── RESUME_GUIDE.md
│   └─ Quick reference guide
│
├── RESUME_COMPLETE_GUIDE.md
│   └─ Comprehensive documentation
│
├── resume_examples.sh
│   └─ Shell script examples
│
└── resume_examples.py
    └─ Python examples
```
