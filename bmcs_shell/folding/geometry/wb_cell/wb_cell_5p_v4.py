import bmcs_utils.api as bu
import k3d
import traits.api as tr
import numpy as np
from bmcs_shell.folding.geometry.wb_cell.wb_cell import WBCell
from numpy import sin, cos, sqrt, tan
from scipy.optimize import fsolve, least_squares , minimize

class WBCell5ParamV4(WBCell):
    name = 'WBCell5ParamV4'

    plot_backend = 'k3d'

    gamma = bu.Float(np.pi / 6, GEO=True)
    phi = bu.Float(0.1, GEO=True)

    a = bu.Float(1000., GEO=True)
    b = bu.Float(1000., GEO=True)
    c = bu.Float(1000., GEO=True)
    symmetric = bu.Bool(True, GEO=True)

    continuous_update = True

    ipw_view = bu.View(
        bu.Item('gamma', latex=r'\gamma', editor=bu.FloatRangeEditor(
            low=1e-6, high=np.pi / 2, n_steps=201, continuous_update=continuous_update)),
        bu.Item('phi', latex=r'\phi', editor=bu.FloatRangeEditor(
            low=-np.pi / 2, high=np.pi / 2, n_steps=201, continuous_update=continuous_update)),
        bu.Item('a', latex='a', editor=bu.FloatRangeEditor(
            low=1e-6, high=2000, n_steps=201, continuous_update=continuous_update)),
        bu.Item('b', latex='b', editor=bu.FloatRangeEditor(
            low=1e-6, high=2000, n_steps=201, continuous_update=continuous_update)),
        bu.Item('c', latex='c', editor=bu.FloatRangeEditor(
            low=1e-6, high=2000, n_steps=201, continuous_update=continuous_update)),
        bu.Item('symmetric'),
        *WBCell.ipw_view.content,
    )

    X_Ia = tr.Property(depends_on='+GEO')
    '''Array with nodal coordinates I - node, a - dimension
    '''

    @tr.cached_property
    def _get_X_Ia(self):
        a = self.a
        b = self.b
        c = self.c
        gamma = self.gamma
        phi = self.phi

        # Define a function that takes the unknowns as input and returns the equations as a numpy array
        def equations(vars):
            xr, yr, zr, xl, yl, zl = vars

            # # non-negativity constraints
            # if zr < 0 or zl < 0 or yr < 0 or yl < 0:
            #     return np.ones(6) * 1e6

            # # opposite sign constraint
            # if np.sign(xr) == np.sign(xl):
            #     return np.ones(6) * 1e6

            z = np.array([0, 0, 1])

            Ur_flat = np.array([a, b, 0])
            Ul_flat = np.array([-a, b, 0])
            Vr_flat = np.array([c, 0, 0])
            Vl_flat = np.array([-c, 0, 0])

            Ul = np.array([xl, yl, zl])
            Ur = np.array([xr, yr, zr])
            Vl = np.array([-c * np.sin(gamma), 0, c * np.cos(gamma)])
            Vr = np.array([c * np.sin(gamma), 0, c * np.cos(gamma)])

            yz_plane_n = np.array([1, 0, 0])
            OUu_plane_n = np.cross((Ur + Ul) / 2, z)

            eq1 = np.linalg.norm(Ur) ** 2 - np.linalg.norm(Ur_flat) ** 2 # OU_ur is constant
            eq2 = np.linalg.norm(Ul) ** 2 - np.linalg.norm(Ul_flat) ** 2 # OU_ul is constant
            eq3 = np.dot(Ul, Ur) - np.dot(Ul_flat, Ur_flat) # angle (OU_ur, OU_ul) is constant
            eq4 = np.dot(Ur, Vr) - np.dot(Ur_flat, Vr_flat) # angle (OU_ur, OVr) is constant
            eq5 = np.dot(Ul, Vl) - np.dot(Ul_flat, Vl_flat) # angle (OU_ul, OVl) is constant
            # eq6: angle between plane yz and plane spanned with [0, 0, 1] with O((U_ur + U_ul)/2) is phi
            eq6 = np.dot(yz_plane_n, OUu_plane_n)/ (np.linalg.norm(yz_plane_n) * np.linalg.norm(OUu_plane_n)) - np.cos(phi)

            return np.array([eq1, eq2, eq3, eq4, eq5, eq6])

        # Define initial guesses for the unknowns
        x0 = np.array([a, a, a, -a, a, a])

        # Call fsolve to solve the equations
        sol = fsolve(equations, x0)
        x_ur, y_ur, z_ur, x_ul, y_ul, z_ul = sol

        if (phi > 0 and (x_ur + x_ul) / 2 < 0) or (phi < 0 and (x_ur + x_ul) / 2 > 0):
            x_ur, y_ur, z_ur, x_ul, y_ul, z_ul = -x_ul, y_ul, z_ul, -x_ur, y_ur, z_ur

        U_ur = np.array([x_ur, y_ur, z_ur])
        U_lr = np.array([x_ur, -y_ur, z_ur])
        U_ul = np.array([x_ul, y_ul, z_ul])
        U_ll = np.array([x_ul, -y_ul, z_ul])
        V_r = np.array([c * sin(gamma), 0, c * cos(gamma)])
        V_l = np.array([-c * sin(gamma), 0, c * cos(gamma)])

        if not self.symmetric:
            U_ll, U_lr = U_lr, U_ll
            U_ll[0] = -U_ll[0]
            U_lr[0] = -U_lr[0]

        X_Ia = np.vstack((np.zeros(3), U_lr, U_ll, U_ur, U_ul, V_r, V_l)).astype(np.float32)
        return X_Ia