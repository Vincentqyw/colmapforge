# Changelog

All notable changes to this project are documented in this file.

## 1.3.0 - 2026-08-01

### Features
- Add a "Launch COLMAP GUI" button that opens the last built database and
  images in COLMAP at any time; enabled after a successful build.

### Fixes
- Disabled buttons now have a distinct base color instead of blending into
  the card background in dark mode.
- The segmentation custom-class hint wraps to multiple lines so the left
  sidebar no longer spills under the splitter handle with a horizontal
  scrollbar.
- Lay out Build / Stop as a full-width primary action with Launch COLMAP
  and Open Output side by side beneath it.

## 1.2.0 - 2026-08-01

### Features
- Restrict SAM support to SAM3, the only SAM family with a native text
  encoder; class prompts are now passed straight to it as text.  Removed
  SAM1 / SAM2 / SAM2.1 backends and the YOLO-World detection bridge.
- Register SkyWater in `models.yaml` alongside SAM3 (first entry, so it is
  the default selection).  Unify model download through a single
  `download_model_entry` that infers zip-vs-single-file from the URL.
- Launch COLMAP GUI after build now defaults to off.

### Fixes
- Segmentation prompts now take effect: SAM3 uses text directly, and the
  pipeline refuses configs lacking a language encoder instead of silently
  producing blank masks.
- Rebuilding the COLMAP database no longer fails with a
  `UNIQUE constraint failed: images.name` — the old DB file is dropped
  before each build.
- Image names in the database are stored relative to the image directory,
  so COLMAP resolves them without doubling the `images` path segment.
- Stopping the pipeline now fully terminates it — workers no longer emit
  `finished` when cancelled, so the next stage never starts.
- Preview panel: removed the frame-count overlay badge; the info label now
  keeps a consistent format (dimensions always shown once known) instead of
  jumping between layouts while navigating.
- The progress bar shows the current status message during long model
  downloads instead of a bare percentage.
