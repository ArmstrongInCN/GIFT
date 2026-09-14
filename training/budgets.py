"""Shared epoch budget for the four supervised prediction baselines.

GIFT retains its separate generator and correction stages. Equal epochs here
mean equal passes over 1,000 trajectories, not equal updates or compute.
"""

PREDICTION_EPOCHS = 500
