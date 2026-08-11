# homogeneous-segmentation <!-- omit in toc -->

Methods for homogenous segmentation of linear spatial data, such as pavement
performance indicators and traffic volumes.

This python package contains two of the three segmentation methods ported from the [R package named HS](https://cran.r-project.org/web/packages/HS/index.html).

- [1. Introduction](#1-introduction)
  - [1.1. Aim](#11-aim)
  - [1.2. Background](#12-background)
  - [1.3. Relevance for Austroads Pavement Design](#13-relevance-for-austroads-pavement-design)
- [2. Installation](#2-installation)
- [3. Usage](#3-usage)
- [4. See Also](#4-see-also)

## 1. Introduction

This python package has been ported from an
[R package - also called HS](https://cran.r-project.org/web/packages/HS/index.html).
The author of the original R package is Yongze Song and the `HS` package is related to
the following paper:

> Song, Yongze, Peng Wu, Daniel Gilmore, and Qindong Li.
> "[A spatial heterogeneity-based segmentation model for analyzing road deterioration network data in multi-scale infrastructure systems.](https://ieeexplore.ieee.org/document/9123684)"
> IEEE Transactions on Intelligent Transportation Systems (2020).

This project is not a complete port of the original R version;
Only the following two methods have been ported:

- Segmentation using `Spatial Heterogeneity Segmentation (SHS)`
- Segmentation using `Minimize Coefficient of Variation (MCV)`

The following is not implemented

- Segmentation using the `Cumulative Difference Approach (CDA)`

### 1.1. Aim

The aim of this package is to help break apart and group road segments based on
one or more road condition variables such that each segment can be reasonably
represented by a single characteristic value.

> Note: I am not yet convinced that existing methods can do a good job of
> segmentation when basing the segmentation on multiple variables.
> Although the functions in this package do accept multiple condition
> variables, they do not for example, support strategies such as
>
> - Weighting each variable based on importance to the segmentation
> - Segmenting first by each variable independently then combining
>   segmentations, etc.

### 1.2. Background

Segmentation refers to the spatial-linear indexing of road data to the physical
road network. Sometimes data is available at constant interval length (e.g.
every 10 metres for roughness), and must be grouped into larger intervals.
Sometimes data has uneven intervals (eg local government area) and must be split
and regrouped.

### 1.3. Relevance for Austroads Pavement Design

Austroads "Guide to Pavement Technology Part 5: Pavement Evaluation and
Treatment Design"
[AGPT05-19](https://austroads.com.au/publications/pavement/agpt05) section
`9.2.5 Selection of Homogeneous Sections` recommends

- Homogenous Sections should be longer than 100 metres
- Homogenous Sections should have a coefficient of variation (standard deviation
  divided by the mean) no greater than 0.25 to be considered homogeneous

`AGPT05-19 Section 9.2.5` recommends the use of the cumulative difference
approach outlined in `Appendix D` of the same document to identify Homogeneous
Sections for the design of pavement overlays.

The paper cited above describes the `Cumulative Difference Approach (CDA)` and
its limitations, and compares its output to the `MCV` and `SHS` algorithms.

## 2. Installation

```bash
pip install homogeneous-segmentation
```

## 3. Usage

The following example shows how to use the the `segment_ids_to_maximize_spatial_heterogeneity()` and `segment_ids_to_minimize_coefficient_of_variation()` methods;

> Note: that in this example the `SHS` and `MCV` methods produce the same
> output. This is not always the case, but seems to be fairly typical for small
> sections of data. I had trouble finding a section of data to demonstrate the
> difference in outputs.

```python
from pyroads.homogeneous_segmentation import (
    segment_ids_to_maximize_spatial_heterogeneity,
    segment_ids_to_minimize_coefficient_of_variation
)
import pandas as pd
from io import StringIO

df = pd.read_csv(StringIO("""road,slk_from,slk_to,cwy,deflection,dirn
H001,0.00,0.01,L,179.37,L
H001,0.01,0.02,L,177.12,L
H001,0.02,0.03,L,179.06,L
H001,0.03,0.04,L,212.65,L
H001,0.04,0.05,L,175.35,L
H001,0.05,0.06,L,188.66,L
H001,0.06,0.07,L,188.31,L
H001,0.07,0.08,L,174.48,L
H001,0.08,0.09,L,210.28,L
H001,0.09,0.10,L,260.05,L
H001,0.10,0.11,L,228.83,L
H001,0.11,0.12,L,226.33,L
H001,0.12,0.13,L,245.53,L
H001,0.13,0.14,L,315.77,L
H001,0.14,0.15,L,373.86,L
H001,0.15,0.16,L,333.56,L"""))

df["seg.shs"] = segment_ids_to_maximize_spatial_heterogeneity(
    data                         = df,
    measure                      = ("slk_from", "slk_to"),
    variable_column_names        = ["deflection"],
    allowed_segment_length_range = (0.030, 0.080)
)

df["seg.mcv"] = segment_ids_to_minimize_coefficient_of_variation(
    data                         = df,
    measure                      = ("slk_from", "slk_to"),
    variable_column_names        = ["deflection"],
    allowed_segment_length_range = (0.030, 0.080)
)

expected_result          = pd.read_csv(StringIO("""road,slk_from,slk_to,cwy,deflection,dirn,seg.shs,seg.mcv
H001,0.00,0.01,L,179.37,L,1,1
H001,0.01,0.02,L,177.12,L,1,1
H001,0.02,0.03,L,179.06,L,1,1
H001,0.03,0.04,L,212.65,L,1,1
H001,0.04,0.05,L,175.35,L,2,2
H001,0.05,0.06,L,188.66,L,2,2
H001,0.06,0.07,L,188.31,L,2,2
H001,0.07,0.08,L,174.48,L,2,2
H001,0.08,0.09,L,210.28,L,2,2
H001,0.09,0.10,L,260.05,L,3,3
H001,0.10,0.11,L,228.83,L,3,3
H001,0.11,0.12,L,226.33,L,3,3
H001,0.12,0.13,L,245.53,L,3,3
H001,0.13,0.14,L,315.77,L,3,3
H001,0.14,0.15,L,373.86,L,3,3
H001,0.15,0.16,L,333.56,L,3,3
"""))

# check the result matches the expected result
pd.testing.assert_frame_equal(
    left       = df,
    right      = expected_result,
    check_like = True # ignore column and row order
)

```

## 4. See Also

See the python package `https://github.com/thehappycheese/segmenter` for further
methods to help with road segments, including tools to break longer segments
into shorter ones. At the time of writing segmenter has not yet been published
to PyPI
