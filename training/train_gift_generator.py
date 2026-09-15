"""Train only the full-data prediction generator; M1 uses its own entry point."""

from training.gift_prediction_control import main_generator

if __name__ == "__main__":
    main_generator()
