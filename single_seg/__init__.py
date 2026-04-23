"""Standalone single-object segmentation package."""

__all__ = ["SingleObjectPointCloudSegmenter", "SingleSegConfig"]


def __getattr__(name: str):
    if name == "SingleObjectPointCloudSegmenter":
        from .single_object_segmenter import SingleObjectPointCloudSegmenter

        return SingleObjectPointCloudSegmenter
    if name == "SingleSegConfig":
        from .single_object_segmenter import SingleSegConfig

        return SingleSegConfig
    raise AttributeError(name)
