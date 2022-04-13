import random

import k3d
import numpy as np
import traits.api as tr
from scipy.optimize import minimize

from bmcs_shell.folding.geometry.wb_tessellation.wb_tessellation_base import WBTessellationBase


class WBNumTessellationBase(WBTessellationBase):
    name = 'WB Num. Tessellation Base'

    sol = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_sol(self):
        print('---------------------------')
        sol = self.minimize_dist()
        # Transfer angles to range [-pi, pi] (to avoid having angle > 2pi so we can do the comparison that follows)
        sol = np.arctan2(np.sin(sol), np.cos(sol))
        print('num_sol=', sol)
        return sol

    def minimize_dist(self):
        x0 = np.array([np.pi, np.pi])
        try:
            res = minimize(self.rotate_and_get_diff, x0, tol=1e-4)
        except:
            print('Error while minimizing!')
            return np.array([0, 0])
        smallest_dist = res.fun
        print('smallest_dist=', smallest_dist)
        sol = res.x
        return sol

    def rotate_and_get_diff(self, rotations):
        br_X_Ia_rot = self._get_br_X_Ia(self.X_Ia, rot=rotations[0])
        ur_X_Ia_rot = self._get_ur_X_Ia(self.X_Ia, rot=rotations[1])
        diff = ur_X_Ia_rot[1] - br_X_Ia_rot[3]
        dist = np.sqrt(np.sum(diff * diff))
        #     print('dist=', dist)
        return dist
