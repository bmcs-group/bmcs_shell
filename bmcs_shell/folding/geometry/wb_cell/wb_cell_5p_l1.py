import bmcs_utils.api as bu
import k3d
import traits.api as tr
import numpy as np
from bmcs_shell.folding.geometry.wb_cell.wb_cell import WBCell
from numpy import sin, cos, sqrt
from scipy.optimize import root


class WBCell5ParamL1(WBCell):
    name = 'WBCell5ParamL1'

    plot_backend = 'k3d'

    debug = False

    v = bu.Float(0.5, GEO=True)
    w = bu.Float(0.5, GEO=True)
    a = bu.Float(1000, GEO=True)
    b = bu.Float(1500, GEO=True)
    c = bu.Float(800, GEO=True)
    u1_sqrt_positive = bu.Bool(True, GEO=True)

    continuous_update = True

    ipw_view = bu.View(
        bu.Item('v', latex='v', editor=bu.FloatRangeEditor(
            low=0, high=1, n_steps=101, continuous_update=continuous_update)),
        bu.Item('w', latex='w', editor=bu.FloatRangeEditor(
            low=0, high=1, n_steps=101, continuous_update=continuous_update)),
        bu.Item('a', latex='a', editor=bu.FloatRangeEditor(
            low=1e-6, high=2000, n_steps=101, continuous_update=continuous_update)),
        bu.Item('b', latex='b', editor=bu.FloatRangeEditor(
            low=1e-6, high=2000, n_steps=101, continuous_update=continuous_update)),
        bu.Item('c', latex='c', editor=bu.FloatRangeEditor(
            low=1e-6, high=2000, n_steps=101, continuous_update=continuous_update)),
        bu.Item('u1_sqrt_positive'),
        bu.Item('gamma', latex=r'\gamma', editor=bu.FloatEditor(), readonly=True),
        *WBCell.ipw_view.content,
    )

    gamma = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_gamma(self):
        a = self.a
        b = self.b
        c = self.c
        v_min = np.max([0., (a ** 2 - b ** 2) * c / (a ** 2 + b ** 2)])
        v_max = c
        v = v_min + (v_max - v_min) * self.v
        return np.round(np.arcsin(v / c), 4)

    X_Ia = tr.Property(depends_on='+GEO')
    '''Array with nodal coordinates I - node, a - dimension
    '''

    @tr.cached_property
    def _get_X_Ia(self):
        return self.get_cell_vertices()

    def get_cell_vertices(self):
        # Control geometry
        a = self.a
        b = self.b
        c = self.c

        # Control folding state
        v_min = np.max([0., (a ** 2 - b ** 2) * c / (a ** 2 + b ** 2)])
        v_max = c
        v = v_min + (v_max - v_min) * self.v
        k = np.sqrt(c ** 2 - v ** 2)

        w_min = (4 * v / c) * (a * v - b * k)
        w_max = (4 * v / c) * (a * v + b * k)
        w = w_min + (w_max - w_min) * self.w
        print('w = ', w)
        print('Sym cell when w = ', str(4 * a * v))

        u1_sign = 1 if self.u1_sqrt_positive else -1
        u1_tmp1 = w * (
                    8 * k ** 2 * a ** 2 - 4 * b ** 2 * c ** 2 + 8 * b ** 2 * v ** 2) + 32 * a ** 3 * c * v ** 2 - 2 * a * c * w ** 2
        u1_sqrt_tmp1 = 4 * a * c - w
        u1_sqrt_tmp2 = 4 * a ** 3 * c - 4 * a * b ** 2 * c - a ** 2 * w - b ** 2 * w
        u1_sqrt_tmp3 = 16 * v ** 4 * (
                    a ** 2 + b ** 2) - 16 * b ** 2 * c ** 2 * v ** 2 - 8 * a * c * v ** 2 * w + c ** 2 * w ** 2
        u1 = u1_tmp1 + u1_sign * 2 * np.sqrt(u1_sqrt_tmp1 * u1_sqrt_tmp2 * u1_sqrt_tmp3)
        u = u1 / (4 * b ** 2 * c ** 2 + 4 * a * c * w - w ** 2)

        M = np.array([0, 0, 0])  # Mittelpunkt
        Uro = (1 / (4 * v * k)) * np.array(
            [w * k,
             np.sqrt(16 * (a ** 2 + b ** 2) * v ** 2 * k ** 2 - w ** 2 * k ** 2 - (w - 4 * a * c) ** 2 * v ** 2),
             v * (4 * a * c - w)]
        )
        Ulu = np.array([-Uro[0], -Uro[1], Uro[2]])
        Ulo = (1 / (4 * v * k)) * np.array(
            [-u * k,
             np.sqrt(16 * (a ** 2 + b ** 2) * v ** 2 * k ** 2 - u ** 2 * k ** 2 - (u - 4 * a * c) ** 2 * v ** 2),
             v * (4 * a * c - u)]
        )
        Uru = np.array([-Ulo[0], -Ulo[1], Ulo[2]])

        # Vr und Vl liegen auf der x-Achse
        # TODO, is correct?
        Vr = np.array([v, 0, k])
        Vl = np.array([-v, 0, k])

        X_Ia = np.vstack((M, Uru, Ulu, Uro, Ulo, Vr, Vl)).astype(np.float32)

        return X_Ia
