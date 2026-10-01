# Restore a committed snapshot

```powershell
embeddingvc checkout main
embeddingvc checkout HEAD~1
embeddingvc checkout <commit-id> --force
embeddingvc checkout HEAD
```

A branch name keeps HEAD attached to that branch. A commit ID, unique prefix or
ancestor expression produces a detached HEAD. No branch reference is moved.
An unborn branch cannot be checked out because it has no saved snapshot.

A normal switch checks staged state, working configuration and tracked source
bytes against the current commit. It refuses changes until you save them or
use `--force`. Force replaces the index and working configuration, including
invalid working JSON/YAML, but **never restores, deletes or overwrites source
files**. Original source bytes are not stored in MVP history. After restoring
an older snapshot, `embeddingvc status` may therefore show source differences.
Untracked files outside tracked roots are preserved and do not block checkout.

`checkout HEAD` is a synchronization repair. It uses the current commit's stored
configuration and vectors, preserving working YAML, index and source edits even
when the working YAML/index are invalid. `--force` does not change this special
behavior. To discard working metadata while remaining at the same snapshot,
check out the current branch name or explicit commit ID with `--force`.

Restoration validates the manifest, configuration and all referenced chunk and
vector objects. It builds and verifies a fresh Chroma collection from stored
vectors; deleted occurrences are absent and model dimensions may change.
Checkout never downloads or invokes an embedding model. Missing or corrupt
objects fail explicitly. Chroma support requires the `vector-store` extra.

## Interrupted checkout recovery

The existing repository lock serializes all mutating commands. Checkout first
builds a replacement collection without changing the active synchronization
record. If building fails, HEAD, working YAML, index and the previous sync record
remain intact. A failed attempt may leave an unused collection; old collections
are retained so previous snapshots remain recoverable.

Once the replacement is verified, checkout records a durable decision at
`.embeddingvc/transactions/checkout.json`. It marks synchronization pending,
atomically replaces YAML/index/HEAD individually, publishes the ready sync
record last, and removes the transaction. These individual file replacements
are recoverable, not a single filesystem-wide atomic operation.

If publication is interrupted, run `embeddingvc checkout HEAD` again. Every
mutating command first finishes a pending checkout under the same lock, without
regenerating vectors. The pending transaction takes precedence over interpreting
the new command: a HEAD repair after interruption repairs the recovered target.
Read-only status reports that checkout recovery is required.

Recovery refuses to overwrite metadata edited after interruption, a target
branch moved to a different commit, or a corrupt target. It retains the
transaction and reports the conflicting file. Preserve your edits, restore the
conflicting metadata to its pre-checkout or intended target state, and retry.
The transaction retains the previous metadata bytes for diagnosis/recovery;
do not delete it to bypass an incomplete publication.

After checking out a detached revision, create and check out a branch before
committing:

```powershell
embeddingvc branch experiment
embeddingvc checkout experiment --force
# Source files remain as they were; inspect status before embedding/committing.
embeddingvc status
```
