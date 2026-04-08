# Plan 5: Label Panel — File List, Annotation List, Properties & Assembly

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the label panel — the main annotation workspace composed of a file list (left), canvas (center), annotation list + properties (right), and a toolbar (top). Wire up image switching, auto-save, undo/redo, and keyboard shortcuts. Integrate into MainWindow.

**Architecture:** Three sub-widgets assembled via QSplitter. FileListWidget manages image list with status colors and lazy thumbnails. AnnotationPanel shows current image's annotations and properties. LabelPanel composes them with a toolbar and connects signals. MainWindow adds the label panel as a tab when a project is opened.

**Tech Stack:** Python 3.10+, PyQt5, pytest

---

## File Structure

```
auto-labeling-v3/
├── src/
│   └── ui/
│       ├── file_list.py           # FileListWidget
│       ├── properties.py          # AnnotationPanel (list + properties)
│       └── label_panel.py         # LabelPanel assembly + toolbar
├── tests/
│   └── ui/
│       ├── test_file_list.py
│       ├── test_properties.py
│       └── test_label_panel.py
```

---

## Task 1: FileListWidget
## Task 2: AnnotationPanel
## Task 3: LabelPanel Assembly + Toolbar
## Task 4: MainWindow Integration
## Task 5: Full Test Suite Verification
