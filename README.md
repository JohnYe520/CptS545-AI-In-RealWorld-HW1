# CPTS545-HW1

## Overview
This is HW1 for **CPTS 545: AI in Real-World** course.  

## Project Structure
```text
11581172_Ye/
├── JohnYe_HW1_Q3a.py            # Bayesian Optimization on 6D Hartmann (continuous)
├── JohnYe_HW1_Q3b.py            # Discrete candidate BO using UCI Airfoil dataset
├── Q3a_AblationTable.png        # Ablation table image for Q3(a)
├── Q3a_Figure.png               # Best-so-far performance plot for Q3(a)
├── Q3b_AblationTable.png        # Ablation table image for Q3(b)
├── Q3b_Figure.png               # Best-so-far performance plot for Q3(b)
│
├── 11581172_Ye_HW1.pdf          # PDF file containing written part from Q1, Q2 and figures from Q3
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