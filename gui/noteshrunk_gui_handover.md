# noteshrunk GUI — Project Handover & Build Instructions

## Context

This document is a complete handover brief for building `noteshrunk_gui.py`, a PySide6-based
graphical front-end for the command-line document-scanning tool `noteshrunk.py`. The attached
`noteshrunk.py` is the **unmodified** back-end and must not be changed. The GUI is a completely
separate, standalone script that imports from it.

---

## Project Goal

Create `noteshrunk_gui.py` — a single-file PySide6 application that exposes all `noteshrunk`
functionality through a graphical interface, with a live split-screen preview, a thumbnail page
strip, and full settings control.

---

## Architecture

### Three-Layer Design

```
noteshrunk_gui.py
├── UI Layer          — PySide6 widgets, layout, signals/slots
├── Bridge Layer      — translates GUI state → argparse.Namespace,
│                       calls noteshrunk functions, returns results
└── noteshrunk.py     — called as an external process, completely untouched
```

### Subprocess / Bridge Design

The GUI calls noteshrunk like this for preview (single page):

```
python noteshrunk.py <input_file> -o <temp_output.png> [all current flags]
```

Read back the temp PNG and display it in the preview panel. Use a system temp directory; clean up after each preview render.

For export (all pages, final PDF):

```
python noteshrunk.py <file1> <file2> ... -o <output.pdf> [all current flags]
```

Page order is taken from the current order in the page strip (respecting any drag-reorder the user performed).

Run subprocess.run() inside a threading.Thread (or a QThread) to prevent it from blocking the Qt event loop. When the subprocess finishes, emit a PySide6 signal from the worker thread to notify the main thread, passing the output temp file path or an error string.

---

## Main Window Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Menu bar  (File → Open, Save As, Quit)                      │
├────────────┬───────────────────────────┬─────────────────────┤
│            │  processed  │  original   │                     │
│  Page      │       ◄────splitter────►  │  Settings           │
│  strip     │                           │  panel              │
│            │   (split-screen preview)  │                     │
│  (thumb-   │                           │  (grouped,          │
│   nails,   │                           │   collapsible)      │
│   vertical │                           │                     │
│   scroll,  │                           │  [x] Auto-preview   │
│   D&D      │                           │                     │
│   reorder) │                           │                     │
├────────────┴───────────────────────────┴─────────────────────┤
│  Status label  │  ████████░░  Page 3/12  │ [Preview]  [Run]  │
└──────────────────────────────────────────────────────────────┘
```

---

## Panel Specifications

### Left Panel — Page Strip

- Vertical `QListWidget` (or custom widget) with scrolling.
- One thumbnail per input file. Thumbnails show the **original** image.
- A small overlay badge per thumbnail indicates state: `idle` / `processing` / `done` / `skipped`
  (skipped = page removed by `--skip-empty`).
- Clicking a thumbnail sets it as the active page and triggers a preview render.
- **Drag-and-drop reordering** of pages within the strip. The order in the strip determines the
  output PDF page order on export.
- Files can also be added by dragging from the OS file manager into the strip.
- The strip is populated via **File → Open** (multi-select file dialog) or drag-and-drop.

### Center Panel — Split-Screen Preview

- A single `QGraphicsView` (preferred) or `QLabel` rendering a composite image: left half is the
  **processed** result, right half is the **original**. They are stitched together into one
  `QPixmap` in Python before display — no side-by-side widget split is used.
- A thin **draggable vertical splitter line** overlaid on the image allows the user to change the
  processed/original ratio freely. The splitter defaults to 50/50.
- Supports zoom (mouse wheel) and pan (click-drag).
- A **fit-to-window** button resets zoom.
- While processing, the panel shows a semi-transparent spinner overlay. The last rendered result
  remains visible underneath.
- If a page is marked as skipped (empty), display a "Page removed (empty)" message overlay.

### Right Panel — Settings Panel

All `noteshrunk` CLI flags are exposed here, grouped into collapsible `QGroupBox` sections:

| Group | Flags |
|---|---|
| **Output** | `--ncolors`, `--dpi`, `--quality` |
| **Background** | `--white-background`, `--black`, `--normalize`, `--saturate` |
| **Palette** | `--local-palette` / global (radio or checkbox), `--palette` (hex input), `--percentage` |
| **Filtering** | `--denoise-median`, `--median-strength`, `--denoise-closing`, `--closing-strength`, `--denoise-opening`, `--opening-strength`, `--unsharp-mask`, `--unsharp-amount`, `--unsharp-radius` |
| **Binarize** | `--binarize`, `--threshold-binarize` |
| **Thresholds** | `--threshold-saturation`, `--threshold-value`, `--threshold-empty` |
| **Empty pages** | `--skip-empty` |

Widget types follow naturally from the argparse definitions (bool → `QCheckBox`, int/float range →
`QSpinBox`/`QDoubleSpinBox`, string → `QLineEdit`). Inspect `parse_args()` to keep widget ranges
in sync with argparse constraints.

**Auto-preview checkbox** lives at the bottom of the settings panel (or the top — your call):
- **Checked (default on)**: any widget change starts a debounced re-render of the current page
  (~300 ms, implemented with `QTimer.singleShot`). The `[Preview]` button in the bottom bar is
  disabled.
- **Unchecked**: widget changes accumulate silently. The `[Preview]` button is enabled and
  triggers rendering on demand.
- This state is persisted to the config file.

### Bottom Bar

Left to right:
- **Status label**: `Idle` / `Processing page N…` / `Done` / `Error: <message>`
- **Progress bar**: shows page X of N during a full export run; hidden or empty when idle
- **`[Preview]` button**: disabled when auto-preview is on
- **`[Run / Export]` button**: triggers the full pipeline for all pages in strip order, then calls
  `merge_pdfs()` and saves the output PDF. Triggers a Save As dialog if no output path is set.

---

## Global Palette Handling

This is non-trivial and must be handled explicitly.

- When `--local-palette` is **off** (default), the global palette is computed from all loaded
  images together via `create_palette(images, args, use_global_palette=True)`.
- Compute the global palette **once** in a background thread whenever the file list changes (files
  added, removed, or reordered).
- Cache the result. The live preview for each individual page uses this cached palette.
- Display a small status indicator in the settings panel near the palette group:
  `● Global palette: ready` (green) / `○ Global palette: outdated` (amber) / `— Not computed`
  (grey, shown when `--local-palette` is on).
- When `--local-palette` is on, per-page palettes are computed on demand during preview.
- Make the limitation clear: the preview with a global palette is only correct if the palette was
  computed from the **current** file list with the **current** settings. If settings change, the
  palette is marked outdated and recomputed before the next preview.

---

## Config File

Path: `~/.config/noteshrunk_gui.conf`
Format: INI via Python's `configparser`

### What to persist

- All settings panel widget values (one key per flag, using the long flag name without `--`)
- Window geometry (`width`, `height`, `x`, `y`)
- Splitter position (processed/original ratio, 0.0–1.0)
- Auto-preview on/off
- Last used output directory

### Behaviour

- Load on application start; apply to all widgets before any signal connections fire (to avoid
  spurious re-renders on startup).
- Save on clean exit (`QApplication.aboutToQuit` signal).
- If the config file is missing or corrupt, silently fall back to defaults (never crash on startup).

---

## Code Structure

Suggested class layout within `noteshrunk_gui.py`:

```python
class PreviewWidget(QGraphicsView):
    """Split-screen preview with draggable splitter, zoom, pan."""

class PageStripItem(QListWidgetItem):
    """Single thumbnail entry in the page strip."""

class PageStrip(QListWidget):
    """Scrollable thumbnail strip with drag-and-drop reorder."""

class SettingsPanel(QScrollArea):
    """All noteshrunk flags as widgets, grouped in QGroupBoxes."""
    settings_changed = Signal()  # emitted on any widget change

class BridgeWorker(QObject):
    """Runs process_image() / create_palette() in a thread."""
    finished = Signal(int, object)   # (page_index, np.ndarray result)
    error    = Signal(int, str)      # (page_index, error_message)
    palette_ready = Signal(object)   # global palette tuple

class MainWindow(QMainWindow):
    """Assembles all panels, bottom bar, menu, config I/O."""
```

---

## PEP 8 & Code Style Rules

- Python ≥ 3.10 assumed.
- All functions do one thing. Keep functions small.
- F-strings for all string formatting.
- List/dict comprehensions over explicit loops where readable.
- No unnecessary abstractions or convoluted control flow.
- Standard scientific libraries (`numpy`, `PIL`) are available and should be used without
  re-implementing anything they already provide.
- Brief comments on non-obvious logic only.

---

## Explicit Non-Goals

- Do **not** modify `noteshrunk.py`.
- Do **not** implement a custom multiprocessing or threading pool in the GUI; noteshrunk handles
  this internally.
- Do **not** support formats other than those already supported by noteshrunk.

---

## Deliverable

A single file: `noteshrunk_gui.py`, importable and runnable as `python noteshrunk_gui.py`,
with `noteshrunk.py` present in the same directory (or on `PYTHONPATH`).
