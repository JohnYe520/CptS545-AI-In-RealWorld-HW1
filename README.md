# CPTS545-HW1

## Overview
This repository contains the coding part of HW1 for **CPTS 545: AI in Real-World**.

## Project Structure
```text
Ye/
├── JohnYe_HW1_Q3a.py            # Bayesian Optimization on 6D Hartmann (continuous)
├── JohnYe_HW1_Q3b.py            # Discrete candidate BO using UCI Airfoil dataset
│
└── README.md                    # Instructions for running the code
```

## Requirements

Before running the project, make sure your Python version is 3.12 or higher.

Install the required Python libraries:

```bash
pip install torch botorch gpytorch matplotlib pandas numpy
```

## How to Run
```bash
# For Q3a
python JohnYe_HW1_Q3a.py

# For Q3b
python JohnYe_HW1_Q3b.py
```

The UCI Airfoil Self-Noise dataset will be automatically downloaded to the current folder when running JohnYe_HW1_Q3b.py for the first time.
