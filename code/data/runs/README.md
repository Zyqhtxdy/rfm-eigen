# Recorded run exports

The `experiment3` and `experiment4` directories contain selected outputs from
the original sealed runs. Their `output_manifest.json` files are preserved
unchanged. Every packaged output matches its entry in the original manifest.
The individual completion records under `calls/` are omitted from these
exports (160 for Example 3 and 20 for Example 4); the corresponding consolidated
rows are included.

`audit.py` verifies each packaged file and reports omitted completion records
separately. Other missing files and any changed file fail the audit. These
exports are evidence for the reported results, not resumable working runs.
Fresh experiment runs under the output root retain their completion records
and are sealed only after all outputs are written.
