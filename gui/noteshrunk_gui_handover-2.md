# noteshrunk GUI — Project Handover Brief

## Project Goal
Build a standalone GUI frontend (`noteshrunk_gui.py`) for the existing CLI tool
`noteshrunk.py`. The two files must remain **fully decoupled** — the GUI calls
noteshrunk exclusively via `subprocess`, never by importing it. Do not modify
`noteshrunk.py` in any way.

---

## Architecture

Three strict layers:

1. **UI layer** — PySide6 widgets, layout, event handlers
2. **Bridge layer** — translates GUI state into a noteshrunk CLI command, executes
   it via `subprocess.run()` inside a worker thread, reads results back via temp files
3. **noteshrunk** — called as an external process, completely untouched

---

## Framework

**PySide6** (not PyQt6). Reason: PySide6 is LGPL; PyQt6 is GPL. API is identical.

---

## Main Window Layout

```
┌──────────────────────────────────────────────────────────┐
│  Menu bar (File → Open, Save As, Quit)                   │
├───────────┬──────────────────────────┬───────────────────┤
│           │  processed │  original   │                   │
│  Page     │      ◄─────┼─────►       │  Settings         │
│  strip    │   (draggable splitter)   │  panel            │
│  (thumb-  │                          │  (grouped,        │
│  nails,   │                          │   collapsible     │
│  vertical │                          │   sections)       │
│  scroll,  │                          │                   │
│  drag to  │                          │  [x] Auto-preview │
│  reorder) │                          │                   │
├───────────┴──────────────────────────┴───────────────────┤
│  Status label  |  ████░░ Page 3/12  |  [Preview]  [Run] │
└──────────────────────────────────────────────────────────┘
```

---

## Panel Details

### Left — Page Strip

- Vertical scrollable list of thumbnail widgets, one per input file
- Thumbnails show the **original** image
- A small badge overlaid on the thumbnail indicates state:
  `idle` / `processing` / `done` / `skipped (empty)`
- Click a thumbnail → set as the active preview page
- **Drag-and-drop reordering** of pages (changes the final PDF page order on export)
- Files can be added by drag-dropping onto the strip or via File → Open

### Center — Preview Panel

- Renders a **single composite image**: left half = processed output, right half = original
- The dividing line is a **draggable splitter** (defaults to 50/50)
- Zoom (mouse wheel) and pan (click-drag) support; fit-to-window button
- While processing: show a spinner or dimmed overlay on the processed half
- The preview always shows only the **currently selected page**

### Right — Settings Panel

Collapsible groups mapping directly to noteshrunk CLI argument groups:

| Group      | Arguments                                                                              |
|------------|----------------------------------------------------------------------------------------|
| Output     | `--ncolors`, `--dpi`, `--quality`                                                      |
| Background | `--white-background`, `--black`, `--normalize`, `--saturate`                           |
| Palette    | `--local-palette` / `--palette` (hex input) / global, `--percentage`                  |
| Filtering  | `--denoise-median`, `--denoise-closing`, `--denoise-opening` + strengths;              |
|            | `--unsharp-mask` + `--unsharp-amount`, `--unsharp-radius`                              |
| Binarize   | `--binarize`, `--threshold-binarize`                                                   |
| Thresholds | `--threshold-saturation`, `--threshold-value`, `--threshold-empty`                     |
| Skip empty | `--skip-empty`                                                                         |

Every widget maps 1:1 to a CLI flag — the bridge simply reads widget values and
assembles a command list.

**Auto-preview checkbox** (state persisted in config):

- **Checked**: any widget change triggers a debounced re-render (~300 ms via
  `QTimer.singleShot`). The `[Preview]` button is disabled.
- **Unchecked**: changes accumulate silently; `[Preview]` button is active and
  triggers render on demand.

### Bottom Bar

- Status label (idle / processing page N / done / error message)
- Progress bar (page X of N), active during full export runs
- `[Preview]` button — disabled when auto-preview is on
- `[Run]` button — triggers full pipeline on all pages and saves the output PDF

---

## Subprocess / Bridge Design

### Preview (single page)

```
python noteshrunk.py <input_file> -o <temp_output.png> [all current flags]
```

Read back the temp PNG and display it in the preview panel. Use a system temp
directory; clean up after each preview render.

### Export (all pages, final PDF)

```
python noteshrunk.py <file1> <file2> ... -o <output.pdf> [all current flags]
```

Page order is taken from the current order in the page strip (respecting any
drag-reorder the user performed).

### Threading Model

`subprocess.run()` is blocking. To prevent it from freezing the Qt event loop,
run it inside a `QThread` (or `threading.Thread`). When the subprocess finishes,
emit a PySide6 signal from the worker thread to notify the main thread, passing
the output temp file path or an error string. There is no IPC — the only
communication between the GUI process and the noteshrunk process is the output
file written to disk.

```
Main thread (GUI)          Worker thread
──────────────────         ──────────────────────────────
user triggers render  →    QThread starts
GUI stays responsive       subprocess.run(["python", "noteshrunk.py", ...])
                           ... blocks until noteshrunk exits ...
                           emit signal(output_path) or signal(error_str)
                      ←    signal received on main thread
display result / error     thread ends
```

### Global Palette Limitation

When `--local-palette` is **off**, the real global palette is built from all
input images together — this is only possible on a full export run (where all
files are passed to noteshrunk at once). Single-page preview subprocess calls
each compute their own palette, meaning **preview colours may differ slightly
from the final export**.

Make this limitation **visible to the user** with a non-intrusive info label in
the settings panel near the palette section, e.g.:

> ⓘ *Preview uses a per-page palette. Final export uses the global palette.*

Show this label only when `--local-palette` is off and more than one file is loaded.

---

## Settings Persistence

- File: `~/.config/noteshrunk_gui.conf`
- Format: INI via Python's `configparser`
- Persist on clean exit, restore on launch
- Contents: all settings panel widget values, window geometry, splitter position,
  auto-preview checkbox state, last-used output path

---

## Code Structure

```
noteshrunk_gui.py
│
├── main()
│       Entry point; constructs QApplication and MainWindow
│
├── class MainWindow(QMainWindow)
│   ├── __init__                   Assembles the three-panel layout and bottom bar
│   ├── _build_page_strip()        Creates the left panel (PageStrip widget)
│   ├── _build_preview()           Creates the center panel (PreviewWidget)
│   ├── _build_settings()          Creates the right panel (SettingsPanel widget)
│   ├── _build_bottom_bar()        Status label, progress bar, Preview/Run buttons
│   ├── _load_config()             Reads ~/.config/noteshrunk_gui.conf via configparser
│   ├── _save_config()             Writes config on clean exit (closeEvent)
│   ├── _on_page_selected()        Slot: active page changed → trigger preview if auto
│   ├── _on_setting_changed()      Slot: any setting widget changed → debounce timer
│   ├── _on_preview_requested()    Slot: Preview button or debounce timer fired
│   └── _on_run_requested()        Slot: Run button → full export
│
├── class PageStrip(QWidget)
│   ├── Scrollable QListWidget with custom PageThumbnail items
│   ├── Supports drag-and-drop internal reordering
│   ├── add_files(paths: list[Path])
│   ├── current_file() -> Path
│   ├── ordered_files() -> list[Path]
│   └── set_badge(index: int, state: str)
│           Updates the overlay badge on a thumbnail
│
├── class PreviewWidget(QWidget)
│   ├── Composite left/right split view (processed | original)
│   ├── Draggable splitter line (drawn on a QLabel or via QGraphicsView)
│   ├── Zoom + pan logic
│   ├── set_original(image: QPixmap)
│   ├── set_processed(image: QPixmap)
│   └── show_spinner(visible: bool)
│
├── class SettingsPanel(QWidget)
│   ├── Collapsible QGroupBox sections (one per argument group)
│   ├── to_args() -> list[str]
│   │       Assembles the full CLI argument list from current widget state
│   ├── from_config(cfg: dict)
│   │       Restores widget state from a config dict
│   └── to_config() -> dict
│           Serialises widget state for config persistence
│
└── class RenderWorker(QThread)
    ├── Signals: finished(str), error(str), progress(int, int)
    ├── __init__(input_files: list[Path], output_path: Path,
    │            args: list[str], preview_mode: bool)
    ├── run()
    │       Calls subprocess.run(); emits finished(output_path) or error(msg)
    └── cancel()
            Sets a flag; kills the subprocess if still running
```

---

## Coding Standards

- PEP 8 throughout
- F-strings for all string formatting
- Functions do one thing; keep them small
- Prefer established library functions over custom logic
- No unnecessary abstractions; keep control flow straightforward
- Python 3.10+ — use `match` statements where appropriate
- Type hints on all function signatures
