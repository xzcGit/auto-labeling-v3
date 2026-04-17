"""Format registry — plugin-style exporter/importer management."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.core.annotation import ImageAnnotation


class ExporterInfo:
    """Metadata for a registered exporter."""

    def __init__(
        self,
        name: str,
        label: str,
        export_fn: Callable,
        needs_classes: bool = True,
        output_is_file: bool = False,
    ):
        self.name = name
        self.label = label
        self.export_fn = export_fn
        self.needs_classes = needs_classes
        self.output_is_file = output_is_file


class ImporterInfo:
    """Metadata for a registered importer."""

    def __init__(
        self,
        name: str,
        label: str,
        import_fn: Callable,
        input_is_file: bool = False,
        file_filter: str = "",
    ):
        self.name = name
        self.label = label
        self.import_fn = import_fn
        self.input_is_file = input_is_file  # True=select file, False=select directory
        self.file_filter = file_filter  # e.g. "JSON files (*.json)" for file dialog


class ExportRegistry:
    """Registry of available export formats."""

    def __init__(self):
        self._exporters: dict[str, ExporterInfo] = {}

    def register(
        self,
        name: str,
        label: str,
        export_fn: Callable,
        needs_classes: bool = True,
        output_is_file: bool = False,
    ) -> None:
        self._exporters[name] = ExporterInfo(
            name=name, label=label, export_fn=export_fn,
            needs_classes=needs_classes, output_is_file=output_is_file,
        )

    def get(self, name: str) -> ExporterInfo | None:
        return self._exporters.get(name)

    def list_names(self) -> list[str]:
        return list(self._exporters.keys())

    def list_labels(self) -> list[str]:
        return [e.label for e in self._exporters.values()]

    def export(
        self,
        name: str,
        annotations: list[ImageAnnotation],
        output_dir: Path | str,
        classes: list[str] | None = None,
        only_confirmed: bool = False,
    ) -> None:
        info = self._exporters.get(name)
        if info is None:
            raise ValueError(f"Unknown export format: {name}")
        output = Path(output_dir)
        if info.output_is_file:
            output = output / f"{name.lower()}.json"
        kwargs = {"only_confirmed": only_confirmed}
        if info.needs_classes and classes is not None:
            kwargs["classes"] = classes
        info.export_fn(annotations, output, **kwargs)


class ImportRegistry:
    """Registry of available import formats."""

    def __init__(self):
        self._importers: dict[str, ImporterInfo] = {}

    def register(
        self,
        name: str,
        label: str,
        import_fn: Callable,
        input_is_file: bool = False,
        file_filter: str = "",
    ) -> None:
        self._importers[name] = ImporterInfo(
            name=name, label=label, import_fn=import_fn,
            input_is_file=input_is_file, file_filter=file_filter,
        )

    def get(self, name: str) -> ImporterInfo | None:
        return self._importers.get(name)

    def list_names(self) -> list[str]:
        return list(self._importers.keys())

    def list_info(self) -> list[ImporterInfo]:
        return list(self._importers.values())


# Global registry instances
_registry = ExportRegistry()
_import_registry = ImportRegistry()


def get_export_registry() -> ExportRegistry:
    """Get the global export format registry."""
    return _registry


def get_import_registry() -> ImportRegistry:
    """Get the global import format registry."""
    return _import_registry


def _register_builtin_formats() -> None:
    """Register all built-in export and import formats."""
    from src.core.formats.yolo import export_yolo_detection, import_yolo_detection
    from src.core.formats.coco import export_coco, import_coco
    from src.core.formats.labelme import export_labelme, import_labelme

    _registry.register("YOLO", "YOLO (txt)", export_yolo_detection, needs_classes=True)
    _registry.register("COCO", "COCO (json)", export_coco, needs_classes=True, output_is_file=True)
    _registry.register("labelme", "labelme (json)", export_labelme, needs_classes=False)

    _import_registry.register(
        "YOLO", "YOLO (txt)", import_yolo_detection,
        input_is_file=False,
    )
    _import_registry.register(
        "COCO", "COCO (json)", import_coco,
        input_is_file=True, file_filter="JSON 文件 (*.json)",
    )
    _import_registry.register(
        "labelme", "labelme (json)", import_labelme,
        input_is_file=False,
    )


_register_builtin_formats()
