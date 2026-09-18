# cs336
My personal implementation of  `cs336 [2026 spring]`

It's currently updating, the lastest version is [2026.09.18]

In this implementation, I only simply do:

1. Train 50M LLM from scratch

· hand-write the flash-attn (fwd, bwd) based on triton 

· hand-clean open-sourced dataset FineWeb-Edu 27GB 

· hand-write the components of the LLM, such as amp, FSDP, ckpt... 

· hand-see the profiling, debug (according to the requirements of `hw2`, `hw3` in `cs336`) 

· hand-write report of LLM training based on the logs. 

