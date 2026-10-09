# Rejected partial run

This corrected-physics trajectory was interrupted at approximately 419 µm. While it was running, the reversible cyclic driver and its restart dependency closure were merged into the worktree. Those changes are inactive for this monotonic case, but the in-memory source identity no longer exactly matched the future commit.

The partial result is excluded from all regression claims. The accepted long regression starts from zero after the source commit is created.
