# Final fracture study package

This compact release contains the external-reader manuscript, Supporting Information, figures and numerical figure data. The Word file is self-contained and has embedded PNG figures, editable Word tables, and native structured OOXML equations. Its package and relationships are validated in `reproducibility/DOCX_VALIDATION.json`; a `textutil` parse passed, but a page-by-page Microsoft Word render is not claimed because the installed Word session did not open either the release or a known control DOCX.

Full native meshes, histories, and checkpoints remain in the hashed evidence roots listed by the manifest. Run `campaign/run_or_resume.py --audit-only` from the repository root to audit campaign status. HTML previews are in `previews/`. Regenerating the structured equations requires `latex2mathml` (pinned in `reproducibility/requirements-docx.txt`), `xsltproc`, and Microsoft Word's installed `mathml2omml.xsl`; the generated DOCX itself has no runtime dependency.

The paper uses only original full spatial PF or original FEM/CZM numerical evidence. Historical field plates retained for geometry/provenance are labeled explicitly and are not pooled with the current quantitative curves.
