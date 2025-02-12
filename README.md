# PINN_DTP_PMSM_OFD

This repository contains scripts for testing and evaluating the trained model.  
The dataset and pre-trained models are stored in **Google Drive**:https://drive.google.com/file/d/1uFNa5oo6fiM_YRFZZBkESjOOslQZFZr6/view?usp=drive_link.  

## Running Tests

### 1. Robustness Testing & Standard Testing
To perform robustness testing or standard evaluation, run:
python test_noload.py

### 2. Uncertainty Analysis  
For uncertainty analysis, execute:
```bash
python test_uncertainty_noload.py

Ensure all dependencies are installed, and that the dataset and model checkpoints are properly linked before execution.
