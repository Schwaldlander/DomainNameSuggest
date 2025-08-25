Each training run writes `runs/<name>/metadata.json` with:
- base model
- training hyperparameters
- dataset path & hash
- timestamp and run hash

Model checkpoints are stored in the run folder. Use Git-LFS or DVC for large files.
