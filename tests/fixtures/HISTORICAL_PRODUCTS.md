# Historical-product test fixtures

The 24 affected tests explicitly request `historical_product`; source-only
tests in the same modules do not. There is no fallback to the mutable
repository `analysis_outputs` trees.

Three small V5.2.1 JSON records support two tests. They are exact Git-blob
copies from archival commit `1816b4f73a7fb671aad29cb03948b10fc8a62b08`,
whose message attests byte identity to the original `e413005...` record.
The original object itself is unavailable locally. Their SHA-256 values are
published in the fixture manifest and independently checked against those
archival Git blobs on use; these are newly recorded hashes, not a claim of
an earlier SHA-256 publication.

The remaining 22 tests need an optional verified historical product pack
that is not installed here. Its absence produces
`HISTORICAL_PRODUCT_FIXTURE_UNAVAILABLE`, not a passing scientific assertion.
Some original assertions additionally need compact archives, source snapshots,
published figure files, or checkpoint payloads. They have not been replaced
with synthetic stand-ins or had those assertions removed.

To install a reviewed external pack, set both
`PF_HISTORICAL_PRODUCT_FIXTURE_ROOT` and
`PF_HISTORICAL_PRODUCT_MANIFEST_SHA256`. The latter must come independently
from its published provenance, not from blindly hashing an untrusted download.
Its root must contain `historical_product_fixture_manifest.json` with
`members` (relative path to SHA-256 and size_bytes) and `tests` (exact node ID
to path `bindings`). Required binding names are fixed by the bundled catalog.
Preserve repository-relative layout inside this compact pack; do not point
the loader at a large simulation repository. Every member, including any
archive, snapshot or image actually needed by an assertion, must be manifested.
The prefix runtime test additionally needs `source_root_relative`, pointing
to its manifested portable checkpoint/ledger subset.

An explicitly configured absent or partial pack, hash mismatch, extra member,
path escape, unknown test or assertion failure is an error and cannot skip.
All original scientific assertions remain in the tests.
