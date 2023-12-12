import time
import numpy as np
import capytaine as cpt
import scipy
from capytaine.io.mesh_writers import write_STL
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import vtk
import logging

import xarray as xr
from capytaine.io.xarray import merge_complex_values
# 频域计算所需要导入的库
# 1、capytaine
# 2、python科学计算工具及绘图工具
# 3、自编函数，用于模块化计算
from Geometry import Create_geometry
from Calculate import hydro
from capytaine.post_pro import rao
logging.basicConfig(level=logging.INFO, format='%(levelname)-8s: %(message)s')