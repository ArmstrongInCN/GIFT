"""Evaluate fresh Gaussian-trained models with exactly the W1 numerical route.

The original W1 measurements are a fixed plotting reference, never a training
input. S3's test IDs are 2260-2439; no original-population model is accepted as
a Gaussian-trained prerequisite. Every model is trained independently first.
"""
from experiments.formal.w1_recursive_prediction.run import main


if __name__ == '__main__':
    main(experiment='S3')
