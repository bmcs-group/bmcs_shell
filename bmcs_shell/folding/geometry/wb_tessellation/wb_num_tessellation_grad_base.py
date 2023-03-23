import random

import k3d
import numpy as np
import traits.api as tr
from scipy.optimize import minimize

import bmcs_utils.api as bu

from bmcs_shell.folding.geometry.wb_cell.wb_cell_4p import WBCell4Param
from bmcs_shell.folding.geometry.wb_tessellation.wb_tessellation_base import WBTessellationBase


"""
A class for a waterbomb cell tessellation that uses a list of different but compatible wb cells arranged gradually
along the x axis to form the tessellation.
"""
class WBNumTessellationGradBase(WBTessellationBase):
    name = 'WBGradNumTessellationBase'

    wb_cells = bu.List([WBCell4Param(), WBCell4Param()], GEO=True)

    # If the distance between the corners of the base cell and the cell above it also needs to be considered in the
    # num. minimization problem. If cells compatbility exists, this is not needed. However, if no compatibility exists
    # it might be useful to see the approximate tessellataion
    minimize_for_upper_cell_too = tr.Bool(True)

    ipw_view = bu.View(
        bu.Item('wb_cells'),
    )

    def get_sol(self, base_cell_X_Ia, glued_cell_X_Ia, side='r'):
        """
        :param side: 'r' or 'l', on which side should the cell be glued, 'r' for ur and br, and 'l' for ul and bl
        """
        print('---------------------------') if self.debug else None
        sol = self.minimize_dist(base_cell_X_Ia, glued_cell_X_Ia, side=side)
        # Transfer angles to range [-pi, pi] (to avoid having angle > 2pi so we can do the comparison that follows)
        sol = np.arctan2(np.sin(sol), np.cos(sol))
        print('num_sol=', sol)
        # if you find solution to a left cell using _get_bl_X_Ia and _get_ul_X_Ia, then sol is simply correct, not -sol
        if self.minimize_for_upper_cell_too:
            return sol
        else:
            return -sol if side == 'l' else sol

    def minimize_dist(self, base_cell_X_Ia, glued_cell_X_Ia, side='r'):
        x0 = np.array([np.pi, np.pi, np.pi]) if self.minimize_for_upper_cell_too else np.array([np.pi, np.pi])
        try:
            res = minimize(self.rotate_and_get_diff, x0, args=(base_cell_X_Ia, glued_cell_X_Ia, side), tol=1e-4)
        except:
            print('Error while minimizing!')
            return np.array([0, 0, 0]) if self.minimize_for_upper_cell_too else np.array([0, 0])
        smallest_dist = res.fun
        print('smallest_dist=', smallest_dist)
        sol = res.x
        return sol

    def rotate_and_get_diff(self, rotations, base_cell_X_Ia, glued_cell_X_Ia, side='r'):
        if self.minimize_for_upper_cell_too:
            return self.rotate_and_get_diff_with_upper_cell_compatibility(rotations, base_cell_X_Ia, glued_cell_X_Ia,
                                                                          side)
        else:
            if side == 'r':
                br_X_Ia_rot = self._get_br_X_Ia(base_cell_X_Ia, rot=rotations[0], X2_Ia=glued_cell_X_Ia)
                ur_X_Ia_rot = self._get_ur_X_Ia(base_cell_X_Ia, rot=rotations[1], X2_Ia=glued_cell_X_Ia)
            elif side == 'l':
                # finding the angle on the left with base_cell_X_Ia as fist cell is also possible. However, this is an
                # equivliant variant which is better because the solution will be similar the other
                br_X_Ia_rot = self._get_br_X_Ia(glued_cell_X_Ia, rot=rotations[0], X2_Ia=base_cell_X_Ia)
                ur_X_Ia_rot = self._get_ur_X_Ia(glued_cell_X_Ia, rot=rotations[1], X2_Ia=base_cell_X_Ia)
            else:
                raise ValueError(str(side) + ' is not valid value for "side"')
            diff1 = ur_X_Ia_rot[1] - br_X_Ia_rot[3]
            dist = np.sqrt(np.sum(diff1 * diff1))
            #     print('dist=', dist)
            return dist

    def rotate_and_get_diff_with_upper_cell_compatibility(self, rotations, base_cell_X_Ia, glued_cell_X_Ia, side='r'):
        if side == 'r':
            br_X_Ia_rot = self._get_br_X_Ia(base_cell_X_Ia, rot=rotations[0], X2_Ia=glued_cell_X_Ia)
            ur_X_Ia_rot = self._get_ur_X_Ia(base_cell_X_Ia, rot=rotations[1], X2_Ia=glued_cell_X_Ia)
            u_X_Ia_rot = self._get_ul_X_Ia(ur_X_Ia_rot, rot=rotations[2], X2_Ia=base_cell_X_Ia)
            diff1 = ur_X_Ia_rot[1] - br_X_Ia_rot[3]
            diff2 = u_X_Ia_rot[2] - base_cell_X_Ia[4]
        elif side == 'l':
            bl_X_Ia_rot = self._get_bl_X_Ia(base_cell_X_Ia, rot=rotations[0], X2_Ia=glued_cell_X_Ia)
            ul_X_Ia_rot = self._get_ul_X_Ia(base_cell_X_Ia, rot=rotations[1], X2_Ia=glued_cell_X_Ia)
            u_X_Ia_rot = self._get_ur_X_Ia(ul_X_Ia_rot, rot=rotations[2], X2_Ia=base_cell_X_Ia)
            diff1 = ul_X_Ia_rot[2] - bl_X_Ia_rot[4]
            diff2 = base_cell_X_Ia[3] - u_X_Ia_rot[1]
        else:
            raise ValueError(str(side) + ' is not valid value for "side"')
        dist = np.sqrt(np.sum(diff1 * diff1)) + np.sqrt(np.sum(diff2 * diff2))
        #     print('dist=', dist)
        return dist
