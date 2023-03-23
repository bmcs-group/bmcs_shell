import bmcs_utils.api as bu
import k3d
import traits.api as tr
import numpy as np
from bmcs_shell.folding.geometry.wb_cell.wb_cell import WBCell
from numpy import sin, cos, sqrt, tan
from scipy.optimize import fsolve

class WBCell2Gammas(WBCell):
    name = 'WBCell2Gammas'

    plot_backend = 'k3d'

    gamma_r = bu.Float(np.pi / 6, GEO=True)
    gamma_l = bu.Float(np.pi / 6, GEO=True)

    a = bu.Float(1000, GEO=True)
    b = bu.Float(1000, GEO=True)
    c = bu.Float(1000, GEO=True)
    sol = bu.Int(0, GEO=True)

    continuous_update = True

    ipw_view = bu.View(
        bu.Item('gamma_r', latex=r'\gamma_r', editor=bu.FloatRangeEditor(
            low=1e-6, high=np.pi / 2, n_steps=501, continuous_update=continuous_update)),
        bu.Item('gamma_l', latex=r'\gamma_l', editor=bu.FloatRangeEditor(
            low=1e-6, high=np.pi / 2, n_steps=501, continuous_update=continuous_update)),
        bu.Item('a', latex='a', editor=bu.FloatRangeEditor(
            low=1e-6, high=2000, n_steps=201, continuous_update=continuous_update)),
        bu.Item('b', latex='b', editor=bu.FloatRangeEditor(
            low=1e-6, high=2000, n_steps=201, continuous_update=continuous_update)),
        bu.Item('c', latex='c', editor=bu.FloatRangeEditor(
            low=1e-6, high=2000, n_steps=201, continuous_update=continuous_update)),
        bu.Item('sol', editor=bu.IntRangeEditor(
            low=0, high=3, n_steps=4, continuous_update=continuous_update)),
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
        gamma_l = self.gamma_l
        gamma_r = self.gamma_r

        # Define a function that takes the unknowns as input and returns the equations as a numpy array
        def equations(vars):
            xr, yr, zr, xl, yl, zl = vars

            # non-negativity constraints
            if zr < 0 or zl < 0 or yr < 0 or yl < 0:
                return np.ones(6) * 1e6

            # opposite sign constraint
            if np.sign(xr) == np.sign(xl):
                return np.ones(6) * 1e6

            Ur_flat = np.array([a, b, 0])
            Ul_flat = np.array([-a, b, 0])
            Vr_flat = np.array([c, 0, 0])
            Vl_flat = np.array([-c, 0, 0])

            Ul = np.array([xl, yl, zl])
            Ur = np.array([xr, yr, zr])
            Vl = np.array([-c * np.sin(gamma_l), 0, c * np.cos(gamma_l)])
            Vr = np.array([c * np.sin(gamma_r), 0, c * np.cos(gamma_r)])

            # other constraints
            if np.cross(Vr, Ur-Vr)[1] <= 0 or np.cross(Vr, Ur-Vr)[1] <= 0 :
                return np.ones(6) * 1e6

            eq1 = np.linalg.norm(Ur) ** 2 - np.linalg.norm(Ur_flat) ** 2
            eq2 = np.linalg.norm(Ul) ** 2 - np.linalg.norm(Ul_flat) ** 2
            eq3 = np.dot(Ul, Ur) - np.dot(Ul_flat, Ur_flat)
            eq4 = np.dot(Ur, Vr) - np.dot(Ur_flat, Vr_flat)
            eq5 = np.dot(Ul, Vl) - np.dot(Ul_flat, Vl_flat)
            # eq6 = np.linalg.norm(Ul-Vl)**2 - np.linalg.norm(Ul_flat-Vl_flat)**2

            eq6 = ((Ur + Ul)/2)[0]

            eq7 = np.linalg.norm(Ur-Vr)**2 - np.linalg.norm(Ur_flat-Vr_flat)**2
            eq8 = np.linalg.norm(Vr) ** 2 - np.linalg.norm(Vr_flat) ** 2
            eq9 = np.linalg.norm(Vl)**2 - np.linalg.norm(Vl_flat)**2

            return np.array([eq1, eq2, eq3, eq4, eq5, eq6])

        # Define initial guesses for the unknowns
        x0 = np.array([a, b, 0, -a, b, 0])

        # Call fsolve to solve the equations
        sol = fsolve(equations, x0)

        x_ur, y_ur, z_ur, x_ul, y_ul, z_ul = sol
        U_ur = np.array([x_ur, y_ur, z_ur])
        U_lr = np.array([x_ur, -y_ur, z_ur])
        U_ul = np.array([x_ul, y_ul, z_ul])
        U_ll = np.array([x_ul, -y_ul, z_ul])
        V_r = np.array([c * sin(gamma_r), 0, c * cos(gamma_r)])
        V_l = np.array([-c * sin(gamma_l), 0, c * cos(gamma_l)])
        X_Ia = np.vstack((np.zeros(3), U_lr, U_ll, U_ur, U_ul, V_r, V_l)).astype(np.float32)
        return X_Ia