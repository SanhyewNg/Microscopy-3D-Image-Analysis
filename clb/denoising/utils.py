from typing import Tuple


def select_stride(
    image_shape: Tuple[int, int], max_stride_size: Tuple[int, int] = (64, 64), patch_size: Tuple[int, int] = (256, 256)
):
    """Select correct stride size that enable proper tileing"""
    row = image_shape[0] - patch_size[0]
    col = image_shape[1] - patch_size[1]
    stride_size = list(max_stride_size)
    assert row > 0 and col > 0, f"Given patch_size: {patch_size} is grater than image_shape: {image_shape}"
    assert (
        patch_size[0] > max_stride_size[0] and patch_size[1] > max_stride_size[1]
    ), f"Given max_stride_size is grater than patch_size"

    while stride_size[0] > 0:
        if row % stride_size[0] == 0:
            break
        stride_size[0] -= 1

    while stride_size[1] > 0:
        if col % stride_size[1] == 0:
            break
        stride_size[1] -= 1

    return tuple(stride_size)
