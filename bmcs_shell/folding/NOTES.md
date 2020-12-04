

# Description of the package structure

The model `WBCell` in `wb_dell.py` implements the 
waterbomb cell folding kinematics.

The model `WBShell` in `wb_shell.py` captures the 
diagonal periodicity of the crease pattern. 
`WBShell` contains one instance of `WBCell` and
shares the cell design parameters `a, b, c` and the 
angle `alpha`.




# Tasks

- Verify the implementation using the tested waterbomb
  shell. This needs an adaption/specialization of the 
  Shell Model
 
- Add a parameter `d` to the WBCell.
  
- Class Structure: Separate the base class for 
  the WB shell generator. E.g. the TwoWBCell shell
  already tested.
  
- 
  
