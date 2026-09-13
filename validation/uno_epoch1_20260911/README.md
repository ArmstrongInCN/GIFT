# U-NO: valid formal epoch1 pause, small recorded loss difference

The fresh run used current controller `7CDA8F5E...`, released data and native
backward. It completed one full epoch of the unchanged150-epoch budget:
1000 trajectories,20-step closed loop,batch16 (last8),63 upstream-Adam updates.
Process exit0 took66.069s; the epoch itself54.298s. No published weights were used.

One actual CPU-only audit took10.184s. Source/data/runtime/receipt bindings and
own epoch0→1 journal integrity passed; each state has36 model tensors, the initial
optimizer is empty and epoch1 has72 moment tensors with all36 cursors at63.
Scheduler and complete Python/NumPy/Torch CPU/CUDA RNG state passed inspection.
Initial tensor hashes all match the prior fresh initialization. No model was
constructed, no RNG/optimizer restored, and no forward, training or CUDA context
was created by this audit. The original native-backward warning remains recorded.

There is **one** independent reference training loss. The new stored per-trajectory
20-step sum is11.766310146331787; dividing by20 gives0.5883155073165893 versus
original0.5883155637741089, an absolute difference5.6457519548303026e-8.
This is not bit-exact. The division is a reporting-unit conversion, not recovery
of the unsaved pre-rounding accumulator. No5% or other new training-loss gate
was applied, and no automatic tuning/retraining occurred.

[Verification and original hashes](verification.json) preserve the actual current
profile separately from missing historical environment fields. Verifier code2
means valid pause with a differing scalar, not failed journal integrity.

This is a historical epoch1 snapshot, not150-epoch or experiment acceptance,
and not an executed same-run GPU resume test. Later authorized continuation may
advance LATEST; it does not invalidate the preserved epoch1 evidence. No weights,
journal tensors, private paths or third-party source are packaged here.
