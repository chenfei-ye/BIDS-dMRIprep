#!/usr/bin/env python

import re
import numpy as np
from nibabel.tmpdirs import InTemporaryDirectory
from dipy.core.geometry import vector_norm


def parse_fsl_affine(file):
    with open(file) as f:
        lines = f.readlines()
    entries = [l.split() for l in lines]
    entries = [row for row in entries if len(row) > 0]  # remove empty rows
    return np.array(entries).astype(np.float32)


def read_bvecs(this_fname):
    """
    Adapted from dipy.io.read_bvals_bvecs
    """
    with open(this_fname, 'r') as f:
        content = f.read()
    # We replace coma and tab delimiter by space
    with InTemporaryDirectory():
        tmp_fname = "tmp_bvals_bvecs.txt"
        with open(tmp_fname, 'w') as f:
            f.write(re.sub(r'(\t|,)', ' ', content))
        return np.squeeze(np.loadtxt(tmp_fname)).T


def rotate(bvecs_in, affine_in, bvecs_out):

    bvecs = read_bvecs(bvecs_in)

    affine = parse_fsl_affine(affine_in)
    affine = affine[:3, :3]

    # Get rotation component of affine transformation
    len = np.linalg.norm(affine, axis=0)
    rotation = np.zeros((3,3))
    rotation[:, 0] = affine[:, 0] / len[0]
    rotation[:, 1] = affine[:, 1] / len[1]
    rotation[:, 2] = affine[:, 2] / len[2]

    # Apply rotation to bvecs
    bvecs = np.array(bvecs).T
    rotated_bvecs = np.matmul(rotation, bvecs)
    norm_bvecs = vector_norm(rotated_bvecs.T)
    rotated_bvecs = np.divide(rotated_bvecs, norm_bvecs.T, 
                out=np.zeros_like(rotated_bvecs), 
                where=norm_bvecs.T!=0) # force unit bvec, modified by Chenfei

    np.savetxt(bvecs_out, rotated_bvecs, fmt='%1.16f')

