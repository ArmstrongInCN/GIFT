# U-Net: actual formal four-step first-batch bridge PASS

Two independent, sequential GPU child processes compared the actual B7 original control with current candidate controller7CDA8F5E… and loaderB79B8D75…. Both initialized seed0 from scratch; no published trained weight was loaded. They used the complete first physical batch20,46 history frames,4 recursive steps, accumulation1 and one native Adam update. The B7 pre-Torch precision guard and thread1/convTF32-on/matmulTF32-off profile were retained.

All14 comparison fields match exactly: all60 initialization states; CPU/CUDA RNG;1000 trajectory/order entries and1000 window starts; runtime; actual batch; loss/full metric; all4 forward outputs;36 complete gradient tensors;60 updated state tensors; complete Adam state and unadvanced StepLR state. Comparison uses SHA256 of every tensor's full byte sequence with dtype/shape, not a numerical tolerance.

Original and candidate child exit codes were0, with elapsed5.109768s and5.084747s. Both recorded552,814,080 bytes peak allocated GPU memory. Loss95.12290954589844 and full metric23.802324295043945 agree.

[Comparison](comparison.json) retains the actual original bytes; [evidence](evidence.json) binds source/data/runtime and actual record hashes; [matching observations](matching_observations.json) stores their identical tensor descriptors/metadata once, without raw weights. No third-party source, UUID or private absolute path is included. Packaging read existing results only.

The earlier B7 campaign's500 scientific log rows and60 terminal tensor matches remain separate historical full-training evidence. This new result binds the current controller's formal four-step first batch only: not its complete training CLI,50-batch epoch,scheduler100-epoch boundary,500-epoch training or M2 evaluation. It does not implement a new pre-import guard inside the candidate CLI or certify different hardware/data profiles.
