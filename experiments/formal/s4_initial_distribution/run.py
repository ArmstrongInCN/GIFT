"""Evaluate fresh Gaussian-trained models with exactly the M2 numerical route.

The original M2 measurements are a fixed plotting reference, never a training
input. S4's test IDs are 2260-2439; no original-population model is accepted as
a Gaussian-trained prerequisite. Every model is trained independently first.
"""
from experiments.formal.m2_recursive_prediction.run import main


if __name__ == '__main__':
    main(experiment='S4')
