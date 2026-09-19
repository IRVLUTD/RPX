# Superseded Molmo runtime

The active VQA branch replaced Molmo with InternVL 3.5 in commit `68da9f7`.
This snapshot preserves the separate Molmo isolation fix from branch
`naren/vqa-molmo-no-tensorflow` at `715213b` without restoring superseded
models to the current evaluation roster. Its isolated Transformers 4.49 Docker
recipe, entrypoint, matrix, backend and tests are preserved here as reference.
This subset is not a standalone checkout; use the source branch for that runtime.
The historical tests are not part of the active benchmark test suite.
