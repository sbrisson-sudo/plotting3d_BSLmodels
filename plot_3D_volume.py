#!/usr/bin/env python

import json
import pickle
from time import time

import numpy as np
from mayavi import mlab

import sys
sys.path.append('/home/sylvain/documents/Geosciences/stage-BSL/tools/ucbpy')

from UCBColorMaps import cmapSved

print(">> Loading modules ok")

#########
# const #
#########

# earth radius
RN = 6371.0

# default surface of the extracted model volumes
RSURF = RN - 50.0

# path (relative or absolute) to plate boundaries and hotspot locations
TECTONICS_PATH = './tectonics'

##########
# config #
##########

# This config is was used for the superswell plot (Nature Fig. 2)
config = {
    # figure properties
    'figure': {
        'fgcolor': (1.0, 1.0, 1.0),
        'bgcolor': (0.0, 0.0, 0.0),
        'size': (1024, 768),
    },

    # settings for saved figures
    'savefig': {
        'save': True,
        'ext': 'png',
    },

    # unique common name for saved figures
    'set_name': '20150718-SPSS',

    # name of the exported volume (i.e. will read in volume.%s.pkl)
    'box_name': 'indian_ocean',

    # plotting geometry
    'cut_top': True,
    'cut_top_depth': 1000.0,
    'cut_bot': False,
    'cut_bot_depth': None,
    'vert_exag': 2.0,

    # config for iso_surface
    'iso_kwargs': {
        'contours': [1.0, -0.7, -1.0, -1.3],
        'vmin': -2.0,
        'vmax': +2.0,
    },

    # whether to plot 2D Vs on the bottom of the box
    'box_bottom_vs': True,

    # misc tectonics, etc.
    'coastlines': True,
    'plates': True,
    'hotspots': True,
    'hotspots_at_cut_depth': True,  # otherwise plotted at earth's surface

    # labels and annotations ...
    # .. apply full lat / lon grid lines to surface
    'full_surf_grid': False,
    # .. apply depth-tick labels
    'tick_labels': False,
    # .. plot colorbar; include labels?
    'colorbar': True,
    'colorbar_labels': True,

    # views to show (i.e. each will be saved)
    'views': [
        {'azimuth': -60.0, 'elevation': 40.0, 'distance': 220.0},
    ],

    # general troubleshooting ...
    # .. whether to force only positive lons (i.e. if box straddles 180E/180W)
    'ensure_positive_lons': False,
    # .. try offscreen rendering (necessary for some versions of mayavi)
    'render_offscreen': False,
}

## This config was used for the Hawaii plot (Nature Fig. 3)
#config = {
#    # figure properties
#    'figure': {
#        'fgcolor': (1.0,1.0,1.0),
#        'bgcolor': (0.0,0.0,0.0),
#        'size': (1024,768),
#    },
#
#    # settings for saved figures
#    'savefig': {
#        'save': True,
#        'ext': 'png',
#    },
#
#    # unique common name for saved figures
#    'set_name': '20150718-Hawaii',
#
#    # name of the exported volume (i.e. will read in volume.%s.pkl)
#    'box_name': 'Hawaii-Small',
#
#    # plotting geometry
#    'cut_top': True,
#    'cut_top_depth': 150.0,
#    'cut_bot': False,
#    'cut_bot_depth': None,
#    'vert_exag': 2.0,
#
#    # config for iso_surface
#    'iso_kwargs': {
#        'contours': [-1.0, -0.7, 0.7],
#        'vmin': -2.0,
#        'vmax': +2.0,
#    },
#
#    # whether to plot 2D Vs on the bottom of the box
#    'box_bottom_vs': True,
#
#    # misc tectonics, etc.
#    'coastlines': True,
#    'plates': True,
#    'hotspots': True,
#    'hotspots_at_cut_depth': False, # otherwise plotted at earth's surface
#
#    # labels and annotations ...
#    # .. apply full lat / lon grid lines to surface
#    'full_surf_grid': False,
#    # .. apply depth-tick labels
#    'tick_labels': False,
#    # .. plot colorbar; include labels?
#    'colorbar': True,
#    'colorbar_labels': True,
#
#    # views to show (i.e. each will be saved)
#    'views': [
#        {'azimuth': -42.5, 'elevation': 70.0, 'distance': 120.0},
#    ],
#
#    # general troubleshooting ...
#    # .. whether to force only positive lons (i.e. if box straddles 180E)
#    'ensure_positive_lons': False,
#    # .. try offscreen rendering (necessary for some versions of mayavi)
#    'render_offscreen': False,
#}

###############################################################################
###############################################################################

#########################
# colormap manipulation #
#########################

def reverse_current_colormap(o):
    lut = o.module_manager.scalar_lut_manager.lut.table.to_array()
    o.module_manager.scalar_lut_manager.lut.table = lut[-1::-1,:]
    mlab.draw()

def use_mpl_colormap(cmap_in, o):
    nlevels_mlab = 255
    lut = 255 * cmap_in(1.0 / (nlevels_mlab - 1) * np.arange(nlevels_mlab))
    o.module_manager.scalar_lut_manager.lut.table = lut
    mlab.draw()

####################
# plotting support #
####################

def plot_hotspots(m, fac, xlims, ylims, r_min, r_plot = None, **kwargs):
        
    with open('%s/hotspots.pkl' % (TECTONICS_PATH), 'rb') as f:
        hotspots = pickle.load(f)
        
    xs, ys = hotspots[1][:,0], hotspots[1][:,1]
    
    if config['ensure_positive_lons']:
        xs = (xs + 360.0) % 360.0
        
    if r_plot is None:
        r_plot = 0.5 * (RN + RSURF)
        
    for x, y in zip(xs, ys):

        if x >= xlims[0] and x <= xlims[1] and y >= ylims[0] and y <= ylims[1]:
            
            # plot cone at the surface
            o = m.points3d([x], [y], [r_plot / fac], **kwargs)
            o.glyph.glyph_source.glyph_source.direction = (0,0,1)
            o.glyph.glyph_source.glyph_source.height = 1.25 * o.glyph.glyph_source.glyph_source.radius
            print(o.glyph.glyph_source.glyph_source)

            # plot line to the CMB
            color = kwargs.get("color")
            line_width = 1e6
            tube_radius = 1e-1
            
            mlab.plot3d([x,x],[y,y],[r_min / fac,r_plot / fac], color=color, tube_radius=tube_radius)
            
            

def plot_plates(m, fac, xlims, ylims, r_plot = None, **kwargs):
    
    for bound in ['ridge', 'transform', 'trench']:
        
        with open('%s/%s.pkl' % (TECTONICS_PATH, bound), 'rb') as f:
            name, segs = pickle.load(f)
            
        if config['ensure_positive_lons']:
            segs[:,0] = (segs[:,0] + 360) % 360.0
            
        ikeep, = np.nonzero(np.all(
            np.vstack([segs[:,0] >= xlims[0], segs[:,0] <= xlims[1],
                       segs[:,1] >= ylims[0], segs[:,1] <= ylims[1]]),
            axis=0))
        
        if r_plot is None:
            r_plot = RSURF
            
        segs_reg = []
        
        for i in range(ikeep.size):
            
            if segs_reg and ikeep[i] != ikeep[i-1] + 1:
                segs_reg.append(np.array([np.nan,np.nan]))
                
            segs_reg.append(segs[ikeep[i],:])
            
        segs_reg = [np.array([np.nan,np.nan])] + segs_reg
        segs_reg.append(np.array([np.nan,np.nan]))
        segs_reg = np.array(segs_reg)
        inan, = np.nonzero(np.isnan(segs_reg[:,0]))
        
        for i,j in zip(inan[:-1], inan[1:]):
            
            srx, sry = segs_reg[i+1:j,0], segs_reg[i+1:j,1]
            if srx.size > 0:
                m.plot3d(srx, sry, r_plot / fac * np.ones(j - i - 1), **kwargs)

def plot_coast(m, fac, xlims, ylims, r_plot = None, **kwargs):
    coast = np.loadtxt('%s/coast.dat' % (TECTONICS_PATH))
    if config['ensure_positive_lons']:
        coast[:,0] = (coast[:,0] + 360) % 360.0
    ikeep, = np.nonzero(np.all(
        np.vstack([coast[:,0] >= xlims[0], coast[:,0] <= xlims[1],
                   coast[:,1] >= ylims[0], coast[:,1] <= ylims[1]]),
        axis=0))
    if r_plot is None:
        r_plot = RSURF
    coast_reg = []
    for i in range(ikeep.size):
        if coast_reg and ikeep[i] != ikeep[i-1] + 1:
            coast_reg.append(np.array([np.nan,np.nan]))
        coast_reg.append(coast[ikeep[i],:])
    coast_reg = [np.array([np.nan,np.nan])] + coast_reg
    coast_reg.append(np.array([np.nan,np.nan]))
    coast_reg = np.array(coast_reg)
    inan, = np.nonzero(np.isnan(coast_reg[:,0]))
    for i,j in zip(inan[:-1], inan[1:]):
        m.plot3d(coast_reg[i+1:j,0], coast_reg[i+1:j,1], r_plot / fac * np.ones(j - i - 1), **kwargs)

################
# main routine #
################

def plot_volume_isosurface():

    ##################
    # data load / prep
    
    print(">> loading serialized data")

    # load the data
    x,y,r,m0,m = pickle.load(open('volume.%s.pkl' % (config['box_name']), 'rb'))
    if config['cut_top']:
        m['S'] = m['S'][:,:,r <= RN - config['cut_top_depth']]
        r_cut = r[r <= RN - config['cut_top_depth']].max()
        r = r[r <= RN - config['cut_top_depth']]
    if config['cut_bot']:
        m['S'] = m['S'][:,:,r >= RN - config['cut_bot_depth']]
        r = r[r >= RN - config['cut_bot_depth']]

    # ensure positive longitudes only
    if config['ensure_positive_lons']:
        x = (x + 360.0) % 360.0

    # figure out the exageration factor for the vertical
    fac = 111.0 / config['vert_exag']

    # to percent
    m['S'] *= 100

    ##########
    # plotting
    
    print(">> mayavi environement initialization")

    # initialize the figure
    # if config['render_offscreen']:
    #     mlab.options.offscreen = True
    f = mlab.figure(1, **config['figure'])

    # isocontours
    src = mlab.pipeline.scalar_field(m['S'])
    o = mlab.pipeline.iso_surface(
        src,
        opacity=1.0,
        colormap='jet', # using jet as a dummy value for now
        extent=[x[0,0], x[-1,0], y[0,0], y[0,-1], r[0] / fac, r_cut / fac],
        **config['iso_kwargs'])

    # set the actual color map to that of SEMum
    use_mpl_colormap(cmapSved(512), o)

    # include a colorbar
    if config['colorbar']:
        if config['colorbar_labels']:
            mlab.colorbar(orientation='vertical', title='dlnVs', nb_labels=5)
        else:
            mlab.colorbar(orientation='vertical', nb_labels=0)

    # manually draw the scale (support for this in mayavi is rather lacking ...)
    scale_color = config['figure']['fgcolor']
    scale_lw = 1.5
    mlab.outline(line_width=scale_lw, color=scale_color)
    # tick intervals
    dtick = 10     # lat / lon (degrees)
    dtickr = 250   # depth (km)
    tickwidthr = 1 # length of a tick line (degrees)
    # build tick lists
    rs = []
    rt = RN
    while rt > r[0]:
        if rt < RN - config['cut_top_depth']:
            rs.append(rt)
        rt -= dtickr
    rs = np.asarray(rs)
    lon_tick_min = int(x[0,0] / dtick) * dtick
    lat_tick_min = int(y[0,0] / dtick) * dtick
    lons = np.arange(lon_tick_min, x[-1,0] + 1e-5 * dtick, dtick)
    lats = np.arange(lat_tick_min, y[0,-1] + 1e-5 * dtick, dtick)
    # lat / lon grid ticks
    grid_tick_width = 2.0 # (degrees)
    for lon in lons:
        mlab.plot3d(
            [lon,lon], [y[0,0],y[0,0] + grid_tick_width], [r[-1] / fac,r[-1] / fac],
            color=scale_color, tube_radius=None, line_width=scale_lw)
        mlab.plot3d(
            [lon,lon], [y[-1,-1],y[-1,-1] - grid_tick_width], [r[-1] / fac,r[-1] / fac],
            color=scale_color, tube_radius=None, line_width=scale_lw)
    for lat in lats:
        mlab.plot3d(
            [x[0,0],x[0,0] + grid_tick_width], [lat,lat], [r[-1] / fac,r[-1] / fac],
            color=scale_color, tube_radius=None, line_width=scale_lw)
        mlab.plot3d(
            [x[-1,-1],x[-1,-1] - grid_tick_width], [lat,lat], [r[-1] / fac,r[-1] / fac],
            color=scale_color, tube_radius=None, line_width=scale_lw)
    if config['full_surf_grid']:
        for lon in lons:
            mlab.plot3d(
                [lon,lon], [y[0,0],y[0,-1]], [r[-1] / fac,r[-1] / fac],
                color=scale_color, tube_radius=None, line_width=scale_lw)
        for lat in lats:
            mlab.plot3d(
                [x[0,0],x[-1,0]], [lat,lat], [r[-1] / fac,r[-1] / fac],
                color=scale_color, tube_radius=None, line_width=scale_lw)
        mlab.plot3d([x[0,-1],x[0,-1]], [y[0,-1],y[0,-1]], [r[0] / fac,r[-1] / fac],
                color=scale_color, tube_radius=None, line_width=scale_lw)
        mlab.plot3d([x[-1,-1],x[-1,-1]], [y[-1,-1],y[-1,-1]], [r[0] / fac,r[-1] / fac],
                color=scale_color, tube_radius=None, line_width=scale_lw)
        mlab.plot3d([x[0,0],x[0,0]], [y[0,0],y[0,0]], [r[0] / fac,r[-1] / fac],
                color=scale_color, tube_radius=None, line_width=scale_lw)
        mlab.plot3d([x[-1,0],x[-1,0]], [y[-1,0],y[-1,0]], [r[0] / fac,r[-1] / fac],
                color=scale_color, tube_radius=None, line_width=scale_lw)
    # radial ticks
    for rt in rs:
        mlab.plot3d([x[0,0],x[0,0] + tickwidthr], [y[0,0],y[0,0]], [rt / fac, rt / fac],
            color=scale_color, tube_radius=None, line_width=scale_lw)
        mlab.plot3d([x[0,0],x[0,0]], [y[0,0],y[0,0] + tickwidthr], [rt / fac, rt / fac],
            color=scale_color, tube_radius=None, line_width=scale_lw)
        mlab.plot3d([x[0,-1],x[0,-1] + tickwidthr], [y[0,-1],y[0,-1]], [rt / fac, rt / fac],
            color=scale_color, tube_radius=None, line_width=scale_lw)
        mlab.plot3d([x[0,-1],x[0,-1]], [y[0,-1],y[0,-1] - tickwidthr], [rt / fac, rt / fac],
            color=scale_color, tube_radius=None, line_width=scale_lw)

        mlab.plot3d([x[-1,-1],x[-1,-1] - tickwidthr], [y[-1,-1],y[-1,-1]], [rt / fac, rt / fac],
            color=scale_color, tube_radius=None, line_width=scale_lw)
        mlab.plot3d([x[-1,-1],x[-1,-1]], [y[-1,-1],y[-1,-1] - tickwidthr], [rt / fac, rt / fac],
            color=scale_color, tube_radius=None, line_width=scale_lw)

        mlab.plot3d([x[-1,0],x[-1,0] - tickwidthr], [y[-1,0],y[-1,0]], [rt / fac, rt / fac],
            color=scale_color, tube_radius=None, line_width=scale_lw)
        mlab.plot3d([x[-1,0],x[-1,0]], [y[-1,0],y[-1,0] + tickwidthr], [rt / fac, rt / fac],
            color=scale_color, tube_radius=None, line_width=scale_lw)

        if config['tick_labels']:
            print(int(RN - rt) % 250)
            if int(RN - rt) % 250 == 0:
                o = mlab.text3d(x[-1,-1], y[-1,-1], rt / fac, '%.0f' % (RN - rt))

    xlims = x.min(),x.max()
    ylims = y.min(), y.max()


    # add hotspots, coastlines, and plate boundaries
    if config['hotspots']:
        plot_hotspots(
            mlab, fac, xlims, ylims,
            color=(0,0.8,0.1),
            mode='cone',
            r_min=r[0],  
            r_plot=r_cut + 30.0 if config['hotspots_at_cut_depth'] else None,
            resolution=32,
            scale_factor=6)
    if config['coastlines']:
        plot_coast(
            mlab, fac, xlims, ylims,
            color=(1.0,1.0,1.0),
            r_plot = r[-1],
            tube_radius=None,
            line_width=2.0)
    if config['plates']:
        plot_plates(
            mlab, fac, xlims, ylims,
            color=(1.0,0,0.5),
            r_plot = r[-1],
            tube_radius=None,
            line_width=5.0)

    # add structure at bottom of box
    if config['box_bottom_vs']:
        o = mlab.surf(
            m['S'][:,:,0].reshape(x.shape),
            vmin=-2, vmax=2,
            colormap='jet',
            extent=[x[0,0], x[-1,0], y[0,0], y[0,-1], r[0] / fac, r[1] / fac])
        use_mpl_colormap(cmapSved(512), o)

    ##################
    # build and orient

    # save config for later reference
    with open('config.%s.json' % (config['set_name']), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    i = 0
    for view in config['views']:
        mlab.view(view['azimuth'], view['elevation'], distance=view['distance'])
        if config['savefig']['save']:
            t0 = time()
            f.scene.save('frame_%4.4i.%s.%s' % (i, config['set_name'], config['savefig']['ext']))
            t1 = time()
            print('saved frame %4.4i (%s) in %.1f s' % (i, repr(view), t1 - t0))
        i += 1
    print('... done (simply close the window when you are done)')
    mlab.show()


def main():
    plot_volume_isosurface()

if __name__ == '__main__':
    main()
