# Migration receipts

Migration inventory, cutover, and cleanup reports can contain operator paths,
device topology, media hashes, project identifiers, and capacity measurements.
They are runtime evidence, not portable source.

Write new receipts under the ignored `var/artifacts/migrations` tree. The
tracked relocation code, schemas, and synthetic tests define and verify the
portable behavior; raw receipts remain with their operator-owned source.
