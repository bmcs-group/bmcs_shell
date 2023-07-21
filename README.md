# BMCS Shells

![alt text](assets/wb_concrete_shell.png "Screenshot from the interactive waterbomb geometries generator app")

Development framework for geometry development, numerical modeling, and production of brittle-matrix composite shells.
Currently, the focus is on developing flexible foldable shell geometries based on origami waterbomb patterns.

## Content
### Geometry
* **Waterbomb-based geometries**
  * Shells based on symmetric 4 parameters waterbomb base
  ![alt text](assets/4p_waterbomb_shell_app_screenshot.png "Screenshot from the interactive waterbomb geometries generator app")

  * Slabs based on Symmetric 4 parameters waterbomb base\
  ![alt text](assets/4p_waterbomb_slab_app_screenshot.png "Screenshot from the interactive waterbomb geometries generator app")

  * Shells based on semi-symmetric 5 parameters waterbomb base 
  ![alt text](assets/5p_waterbomb_shell_app_screenshot.png "Screenshot from the interactive waterbomb geometries generator app")

### Numerical structural analysis
Pending

### Production processes
Pending

## Installation

### Dependencies

Dependencies on `gmsh` and `pygmsh` included in `environment.yml` 
Anaconda versioning seems not up to date.

Problem with dependencies in `pyface` with a missing 
dependency on `anaconda` cloud. As a fix,`importlib-resources`
that was reported as missing has been included in environment.

