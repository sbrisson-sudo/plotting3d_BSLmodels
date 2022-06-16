# Short documentation

In order to run the scripts you need to :
1. set up the associated conda environement
```bash
conda env create -n mayavi-env --file=environment.yml
```
2. dowload the UCB package `usbpy` (available at https://github.com/sbrisson-sudo/ucbpy.git), you may need to recompile the C interfaces
3. update the paths to this package in the codes


# Example codes for 3D plotting

Here, you can find generalized versions of the code I've used for 3D plotting
of the SEMUCB-WM1 model. Specifically, there are two different codes:
1. `extract_volume_cartesian.py` : extracts a cartesian (lat, lon, depth) box
   from the selected A3d model; and
2. `plot_3D_volume.py` : generates a 3D a plot of the exported volume.

## Dependencies

Both codes depend on the `ucbpy` set of python modules (specifically, for A3d
model interpolation routines and UCB colormaps) and numpy / scipy, while
`plot_3D_volume.py` also depends on Mayavi's `mlab` matplotlib-like plotting
interface. The latter can be rather difficult to build independently, so I
would recommend obtaining it as part of a complete python environment, such as
Enthought Canopy or Anaconda.

## Usage

Both codes are fairly well commented and largely self-explanatory. Both codes
are configured by hand in the section labeled

    ##########
    # config #
    ##########

as opposed to being driven by command line arguments.

For the volume extraction code, you will need to configure:
1. the dimensions of the box you are extracting (lat, lon, depth limits);
2. the step sizes in both the lat / lon and depth dimensions; and
3. 1D reference and A3d model details (model file names, spherical spline grid
   knot files, etc.)
Note that for large export volumes, you will want to run this on a machine with
a fair amount of RAM (e.g. the `lx4` or `lx5` compute servers).

For the plotting code, you have a lot more knobs to turn. Specifically, all
relevant configuration parameters are set in the global `config` dict object.
The motivation behind placing all of the configuration in a single object is so
that it can easily be saved (in this case, as json) when also saving your plot
files (e.g. png files) to disk. This makes it easy to reproduce figures later
on if needed. These are saved as `config.$set_name.json` where `$set_name` is
simply a user-defined field in the `config` intended to be meaningful to you
for keep track of different configurations (your saved plot files will have the
same field appear in their respective filenames).

In both python scripts, you can find two pre-defined configurations: one for
the Superswell figure (Nature Fig. 2) and one for the Hawaii figure (Nature
Fig. 3a).

The plotting script is designed to loop through multiple 'views' of the same
rendering, saving each to disk if desired  (see the `views` list in the
`config` dict). Your mileage with this may vary ... Mayavi is great for many
reasons, but it is also kind of a buggy, poorly documented mess. You may find
that this "multi-view" approach does not work, and you need to run the script
for each view individually (i.e. multiple configurations).
