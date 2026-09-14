# Native PINN terminal bundles

Each condition/library folder contains `result.json`, `terminal_state.npz` and
`COMPLETED.json`. Keep these three files together. The JSON and NPZ hashes,
59 native TensorFlow variables, coefficient/mask consistency, full training
budget and observation population are checked by the NumPy-only reader.

```shell
python -m adapters.pinn_reference --condition noise_000 --mode known
```

This reads learned coefficients; it does not train or run TensorFlow inference.
The full coefficient vector is retained, including additional terms. Three
physical parameter estimates do not establish successful sparse-structure
recovery or low error. Consult the experiment results for actual accuracy.

The terminal bundle is not a standalone resumable journal. Training commands
save separate same-run model/optimizer/random/phase boundaries in their output
directory. Continue an interrupted training run using that directory and
`--resume`, not by inserting these downloaded terminal weights into a new run.

See `artifacts/CHECKPOINTS.json` for the available completed bundles. Missing
conditions are not substituted by another condition or a diagnostic checkpoint.
