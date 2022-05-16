#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Thu Apr 07 2022
@author: Sylvain Brisson sylvain.brisson@ens.fr
"""

# importations
import sys,os

path_bsl_toolbox = "/home/sylvain/documents/Geosciences/stage-BSL/tools/bsl_toolbox"
sys.path.append(path_bsl_toolbox)

from plotting.plot_A3Dmodel_map_greatCircles import plot_model, plot_hotspots, plot_plates
from common.setup import models_path_default

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import argparse

import warnings
warnings.filterwarnings('ignore')

def plot_box(lonlat1, lonlat2, ax):
    """Plot the rectangular box delimited by two opposite points given in (lon-lar)
    """
    
    lon1,lat1 = lonlat1 
    lon2,lat2 = lonlat2
    
    lons = [lon1,lon1,lon2,lon2,lon1]
    lats = [lat1,lat2,lat2,lat1,lat1]
    
    ax.plot(lons, lats, "m", transform=ccrs.Geodetic(), lw=3, zorder=10, label="Box boundaries")


# to have better defined great circles
# https://stackoverflow.com/questions/40270990/cartopy-higher-resolution-for-great-circle-distance-line
class LowerThresholdOthographic(ccrs.Orthographic):
    @property
    def threshold(self):
        return 1e3


if __name__ == "__main__":
    
    # +++ config +++
    box_name = 'Indian ocean'
    box = (
        (40.0, -60.0), 
        (120.0, 0.0)
        )
    lon0,lat0 = 80.0,-30.0
    depth = 2800.
    # +++ config +++
    
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('-o',dest="out_file",type=str,help='output figure name')
    args = parser.parse_args()
    
    
    fig = plt.figure(figsize=(8,8))
    proj = LowerThresholdOthographic(
        central_latitude=lat0,
        central_longitude=lon0
    ) 
    
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.gridlines(linestyle=":", color="k")  
    
    ax.set_global()
    
    # plot model in background
    plot_model(
        model_name = "2.6S6X4",
        model_file = os.path.join(models_path_default, "Model-2.6S6X4-ZeroMean.A3d"),
        parameter = "S", 
        grid_file = os.path.join(models_path_default, "grid.6"), 
        rmean = False, 
        depth_and_level = [depth,3.0,7.265], 
        ref_model_file = os.path.join(models_path_default, "Model-2.6S6X4-ZeroMean_1D"),
        ax = ax    
        )
    
    # add coastlines
    ax.coastlines()
    
    # plot plates and hostpots
    plot_hotspots(ax)
    plot_plates(ax)
    
    # plot box limits
    plot_box(*box, ax)
    
    plt.title(f"{box_name} box")
    plt.legend()
    
    # showing / saving
    if args.out_file:
        if os.path.exists(args.out_file):
            if input(f"File {args.out_file} exists, overwrite ? [y/n] ") != "y":
                print(f"No file saved.")
                exit()
        print(f"Saving {args.out_file}")
        plt.savefig(args.out_file, dpi=500, bbox_inches='tight')
    else:
        plt.show()