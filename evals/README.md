# Evaluations

The executable evaluation order is:

1. synthetic generic suite across all three scenarios;
2. clean bootstrap and full repository verification;
3. sealed path/hash verification only;
4. machine-readable readiness gate;
5. independent ground-truth motion-recovery gate;
6. independent reference analysis and reconstruction;
7. clean rerun and transfer on a different source;
8. comparison report and candidate editing-knowledge packet.

Copy `reference.lock.example.json` to the ignored `reference.lock.json`, then
replace every placeholder with the operator-provided paths, sizes, and hashes.
The lock identifies inputs without decoding their content. The reference video
hash is permanently forbidden in render lineage. The source image becomes a
permitted reconstruction input only after a passing readiness gate.

Runtime receipts, the operator lock, and reference-derived media live in
untracked repo-local state. Only the safe lock template and evaluation
definitions are checked in; shared provider runtimes and model caches remain
external owner surfaces.
