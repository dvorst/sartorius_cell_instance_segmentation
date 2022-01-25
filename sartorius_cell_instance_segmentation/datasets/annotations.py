import torch
from numba import njit
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms.functional
from PIL import Image
import cProfile
import pstats
import dataclasses
import PIL
import matplotlib.pyplot as plt


@dataclasses.dataclass
class Annotation:
    mask: PIL.Image
    bounds: PIL.Image
    touch: PIL.Image


def to_annotation(img_data: pd.DataFrame, idx: str):
    """
    Obtain annotations and remove overlap

    Wherever two annotations overlap, each overlapping pixel will be
    assigned to the closest annotation and removed from the other.
    """

    # decompress annotations
    annotations = _decompress(img_data, idx)

    # correct for overlap
    annotations = _correct_overlap(annotations)

    # determine boundaries of each annotation, and the boundaries that touch
    bounds, touch = _bounds(annotations)

    # calculate mask from the annotations and touching boundaries
    mask = _mask(annotations, touch)

    return Annotation(mask=_to_pil(mask), bounds=_to_pil(bounds), touch=_to_pil(touch))


def from_mask(mask):
    """ exports the provided mask to the challenges desired format. """
    pass


def export():
    pass


def _correct_overlap(annotations):
    # calculate center of each annotation
    #     after pixels are added/removed, the center changes, the choice
    #     is made not to recalculate the center in such case due to the
    #     computational overhead involved
    # todo: find a more computational efficient way of implementing this

    # elements of overlapping pixels
    overlap = _overlapping_pixels(annotations)

    # if there are no overlapping pixels, nothing has to be corrected
    if len(overlap) == 0:
        return annotations

    # annotation to which each overlapping pixel lies closest to
    closest_annotation = _annotation_closest_to_pixel(annotations, overlap)

    # change the annotations accordingly
    # todo: find a faster method for this
    for idx_pixel, idx_annotation in enumerate(closest_annotation):
        x, y = overlap[idx_pixel]
        # set pixel x, y to zero for all annotations except the
        # annotation that the particular pixel belongs to
        annotations[:, x, y] = 0
        annotations[idx_annotation, x, y] = 1

    return annotations


def _annotation_closest_to_pixel(annotations, pixels):
    """
    Find the annotation that lies closest to the given pixels.

    Distance between center of each annotation and each pixel is calculated.
    The annotation-elements are returned with the closest distance for each given pixel.

    Note that squared_distance is used, practically this involves omitting the root when calculating the
    euclidean distance, reducing the computational requirement. Since only the closest lying annotation is of interest,
    calculating the squared distance will have the same result at a reduced computational effort.
    """
    return torch.argmin(_squared_distances(pixels, _centers(annotations)), dim=0)


def _squared_distances(a, b):
    # create meshgrid for faster computation
    a = a.repeat(b.shape[0], 1, 1)
    b = b.repeat(a.shape[1], 1, 1).transpose(0, 1)

    # return squared distance
    return (a - b) ** 2


def _decompress(data, idx):
    return torch.stack([
        _decompress_single(r['annotation'], r['width'], r['height'])
        for _, r in data.loc[data['id'] == idx].iterrows()
    ])


def _decompress_single(data: str, width, height):
    # convert annotation string to matrix:
    # [column 1: pixel index, column 2: length]
    data = torch.tensor([int(elm) for elm in data.split(' ')]).reshape([-1, 2])

    # create empty 1D image
    annotation = torch.zeros((height * width), dtype=torch.uint8)

    # draw white pixels in the image as indicated by the annotation
    for idx, length in data:
        annotation[idx:(idx + length)] = 1

    # reshape image to 2D
    annotation = annotation.reshape((height, width))

    # return image
    return annotation


def _center(annotation):
    center = torch.nonzero(annotation)  # ids of non-zero elements
    center = torch.sum(center, dim=0) / center.shape[0]  # average id, type: float
    center = torch.round(center).to(torch.int16)  # convert tot int16 (uint8 is too small)
    return center


def _centers(annotations):
    return torch.stack([_center(a) for a in annotations])


def _overlapping_pixels(annotations):
    """
    finds all overlapping pixels

    annotations have a pixel value equal to either 0 or 1. So if two or
    more annotations overlap, the sum of these is >1. Overlap is
    returned as torch tensor with values that lie between 0 and 1.
    """
    overlap_ids = torch.sum(annotations, dim=0, dtype=torch.uint8) > 1
    overlap_ids = torch.nonzero(overlap_ids)
    return overlap_ids


def _to_pil(t: torch.tensor):
    """ transform tensor with values between 0 and 1 to PIL image """
    return torchvision.transforms.functional.to_pil_image(
        (t * 255).type(torch.uint8)
    )


def _bounds(annotations):
    """ Image of touching boundaries from neighbouring annotations """

    bounds = torch.sum(_gradient(annotations), dim=0, dtype=torch.uint8)

    # bounds that touch
    touch = (bounds > 1).to(torch.uint8)

    #  clamp bounds to create mask
    bounds = bounds.clamp(0, 1)

    # return bounds and touch
    return bounds, touch


def _gradient(a: torch.tensor):
    n, y, x = a.shape
    dx = torch.empty((n, y, x + 2), dtype=torch.uint8)
    dy = torch.empty((n, y + 2, x), dtype=torch.uint8)
    dx[:, :, 1:-1] = a
    dx[:, :, -1:1] = 0
    dy[:, 1:-1, :] = a
    dy[:, -1:1, :] = 0
    dx = (dx[:, :, 2:] - dx[:, :, :-2]) ** 2  # squaring is slightly faster than abs, dx^2 is either 0 or 1
    dy = (dy[:, 2:, :] - dy[:, :-2, :]) ** 2

    # use gradient to calculate bounds
    bounds = dx + dy  # bounds element value is either 0, 1 or 2
    bounds = bounds.clamp(0, 1)  # clip bounds such that each element is either 0 or 1

    return bounds


def _mask(annotations, touch):
    """
    create mask from annotations and bounds (i.e. touching boundaries
    between annotations).

    any touching boundaries between annotations will be considered
    'background' such that the annotations remain semantically
    segmented. The mask is boolean, i.e. the pixel values are either 0
    or 1.
    """
    mask = torch.sum(annotations, dim=0, dtype=torch.uint8) - touch
    mask = torch.clamp(mask, min=0, max=1)
    return mask
