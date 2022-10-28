import bmcs_utils.api as bu
import k3d
import traits.api as tr
import numpy as np
from bmcs_shell.folding.geometry.wb_cell.wb_cell import WBCell
from numpy import sin, cos, sqrt
from scipy.optimize import root


class WBCell5ParamV3(WBCell):
    name = 'waterbomb cell 5p v3'

    plot_backend = 'k3d'

    debug = False

    m = bu.Float(0.5, GEO=True)
    # l1 = bu.Float(1000, GEO=True)
    L1 = bu.Float(0.75, GEO=True)
    a = bu.Float(500, GEO=True)
    b = bu.Float(500, GEO=True)
    c = bu.Float(500, GEO=True)
    sol_index = bu.Int(0, GEO=True)
    yro_sign = bu.Bool(True, GEO=True)
    ylo_sign = bu.Bool(True, GEO=True)

    continuous_update = True

    ipw_view = bu.View(
        bu.Item('m', latex='m', editor=bu.FloatRangeEditor(
            low=0, high=1, n_steps=201, continuous_update=continuous_update)),
        bu.Item('L1', latex=r'L_1', editor=bu.FloatRangeEditor(
            low=0, high=1, n_steps=201, continuous_update=continuous_update)),
        # bu.Item('l1', latex=r'l_1', editor=bu.FloatRangeEditor(
        #     low=0, high=2000, n_steps=501, continuous_update=continuous_update)),
        bu.Item('a', latex='a', editor=bu.FloatRangeEditor(
            low=1e-6, high=2000, n_steps=101, continuous_update=continuous_update)),
        bu.Item('b', latex='b', editor=bu.FloatRangeEditor(
            low=1e-6, high=2000, n_steps=101, continuous_update=continuous_update)),
        bu.Item('c', latex='c', editor=bu.FloatRangeEditor(
            low=1e-6, high=2000, n_steps=101, continuous_update=continuous_update)),
        bu.Item('sol_index', latex='sol_{index}', editor=bu.IntRangeEditor(
            low=0, high=1, n_steps=1, continuous_update=continuous_update)),
        bu.Item('yro_sign'),
        bu.Item('ylo_sign'),
        *WBCell.ipw_view.content,
    )

    X_Ia = tr.Property(depends_on='+GEO')
    '''Array with nodal coordinates I - node, a - dimension
    '''

    @tr.cached_property
    def _get_X_Ia(self):
        return self.get_cell_vertices()

    def get_cell_vertices(self):
        sol_index = self.sol_index
        # Control geometry
        a = self.a
        b = self.b
        c = self.c

        # Control folding state
        m = self.m * self.c  # x of Vr, -x of Vl
        # l1 = self.l1

        # These are the two solutions obtained when solving ylo = sqrt(l1 ** 2 - (zlo - k) ** 2 - (xlo - m) ** 2) = 0
        # for l1 (see derivation in the jupyter notebook)
        # Note (28.10.2022): l1_max is working, but l1_min may need to be calcualted from
        l1_max = sqrt(a ** 2 * c - 2 * a * c ** 2 + 4 * a * m ** 2 + b ** 2 * c + 4 * b * m * sqrt(c - m) * sqrt(
            c + m) + c ** 3) / sqrt(c)
        l1_min = sqrt(a ** 2 * c - 2 * a * c ** 2 + 4 * a * m ** 2 + b ** 2 * c - 4 * b * m * sqrt(c - m) * sqrt(
            c + m) + c ** 3) / sqrt(c)
        l1 = (l1_max - l1_min) * self.L1 + l1_min
        print('l1, l1_min, l1_max:', l1, l1_min, l1_max)
        # Idea to switch sign without if else
        # i = 0, 1
        # sign = (-1) ** i

        # c = a -e
        e = a - c
        # b = sqrt(d**2 - e**2)
        d = sqrt(b ** 2 + e ** 2)
        # k is a short cut
        k = sqrt(c ** 2 - m ** 2)
        M = np.array([0, 0, 0])  # Mittelpunkt

        # l2 = 940.25
        # l2, yro_sign, ylo_sign = self.get_l2(a, b, c, d, m, k, l1)

        yro_sign = 1 if self.yro_sign else -1
        ylo_sign = 1 if self.ylo_sign else -1
        l2_ana_sols = [sqrt((
                                    a ** 6 + 3 * a ** 4 * b ** 2 - a ** 4 * c ** 2 - 2 * a ** 4 * l1 ** 2 - 8 * a ** 4 * m ** 2 - 16 * a ** 3 * c * m ** 2 + 3 * a ** 2 * b ** 4 - 6 * a ** 2 * b ** 2 * c ** 2 - 4 * a ** 2 * b ** 2 * l1 ** 2 - a ** 2 * c ** 4 - 4 * a ** 2 * c ** 2 * l1 ** 2 - 8 * a ** 2 * c ** 2 * m ** 2 + a ** 2 * l1 ** 4 + 8 * a ** 2 * l1 ** 2 * m ** 2 + 16 * a * b ** 2 * c ** 3 - 16 * a * b ** 2 * c * m ** 2 + b ** 6 - 5 * b ** 4 * c ** 2 - 2 * b ** 4 * l1 ** 2 + 8 * b ** 4 * m ** 2 - 5 * b ** 2 * c ** 4 + 8 * b ** 2 * c ** 2 * m ** 2 + b ** 2 * l1 ** 4 - 8 * b ** 2 * l1 ** 2 * m ** 2 + c ** 6 - 2 * c ** 4 * l1 ** 2 + c ** 2 * l1 ** 4 - 2 * sqrt(
                                a ** 10 * c ** 2 + 8 * a ** 9 * c * m ** 2 + 5 * a ** 8 * b ** 2 * c ** 2 - 4 * a ** 8 * c ** 4 - 4 * a ** 8 * c ** 2 * l1 ** 2 + 16 * a ** 8 * c ** 2 * m ** 2 + 16 * a ** 8 * m ** 4 - 8 * a ** 7 * b ** 2 * c ** 3 + 32 * a ** 7 * b ** 2 * c * m ** 2 - 8 * a ** 7 * c ** 3 * m ** 2 - 24 * a ** 7 * c * l1 ** 2 * m ** 2 + 64 * a ** 7 * c * m ** 4 + 10 * a ** 6 * b ** 4 * c ** 2 + 8 * a ** 6 * b ** 2 * c ** 4 - 16 * a ** 6 * b ** 2 * c ** 2 * l1 ** 2 - 32 * a ** 6 * b ** 2 * c ** 2 * m ** 2 + 64 * a ** 6 * b ** 2 * m ** 4 + 6 * a ** 6 * c ** 6 + 4 * a ** 6 * c ** 4 * l1 ** 2 - 32 * a ** 6 * c ** 4 * m ** 2 + 6 * a ** 6 * c ** 2 * l1 ** 4 - 32 * a ** 6 * c ** 2 * l1 ** 2 * m ** 2 + 96 * a ** 6 * c ** 2 * m ** 4 - 32 * a ** 6 * l1 ** 2 * m ** 4 - 24 * a ** 5 * b ** 4 * c ** 3 + 48 * a ** 5 * b ** 4 * c * m ** 2 + 8 * a ** 5 * b ** 2 * c ** 5 + 24 * a ** 5 * b ** 2 * c ** 3 * l1 ** 2 - 56 * a ** 5 * b ** 2 * c ** 3 * m ** 2 - 72 * a ** 5 * b ** 2 * c * l1 ** 2 * m ** 2 + 64 * a ** 5 * b ** 2 * c * m ** 4 - 8 * a ** 5 * c ** 5 * m ** 2 - 16 * a ** 5 * c ** 3 * l1 ** 2 * m ** 2 + 64 * a ** 5 * c ** 3 * m ** 4 + 24 * a ** 5 * c * l1 ** 4 * m ** 2 - 64 * a ** 5 * c * l1 ** 2 * m ** 4 + 10 * a ** 4 * b ** 6 * c ** 2 + 32 * a ** 4 * b ** 4 * c ** 4 - 24 * a ** 4 * b ** 4 * c ** 2 * l1 ** 2 - 128 * a ** 4 * b ** 4 * c ** 2 * m ** 2 + 96 * a ** 4 * b ** 4 * m ** 4 - 30 * a ** 4 * b ** 2 * c ** 6 - 36 * a ** 4 * b ** 2 * c ** 4 * l1 ** 2 + 32 * a ** 4 * b ** 2 * c ** 4 * m ** 2 + 18 * a ** 4 * b ** 2 * c ** 2 * l1 ** 4 + 96 * a ** 4 * b ** 2 * c ** 2 * l1 ** 2 * m ** 2 - 32 * a ** 4 * b ** 2 * c ** 2 * m ** 4 - 96 * a ** 4 * b ** 2 * l1 ** 2 * m ** 4 - 4 * a ** 4 * c ** 8 + 4 * a ** 4 * c ** 6 * l1 ** 2 + 16 * a ** 4 * c ** 6 * m ** 2 + 4 * a ** 4 * c ** 4 * l1 ** 4 - 32 * a ** 4 * c ** 4 * l1 ** 2 * m ** 2 + 16 * a ** 4 * c ** 4 * m ** 4 - 4 * a ** 4 * c ** 2 * l1 ** 6 + 16 * a ** 4 * c ** 2 * l1 ** 4 * m ** 2 - 32 * a ** 4 * c ** 2 * l1 ** 2 * m ** 4 + 16 * a ** 4 * l1 ** 4 * m ** 4 - 24 * a ** 3 * b ** 6 * c ** 3 + 32 * a ** 3 * b ** 6 * c * m ** 2 - 16 * a ** 3 * b ** 4 * c ** 5 + 48 * a ** 3 * b ** 4 * c ** 3 * l1 ** 2 + 40 * a ** 3 * b ** 4 * c ** 3 * m ** 2 - 72 * a ** 3 * b ** 4 * c * l1 ** 2 * m ** 2 - 64 * a ** 3 * b ** 4 * c * m ** 4 + 8 * a ** 3 * b ** 2 * c ** 7 + 16 * a ** 3 * b ** 2 * c ** 5 * l1 ** 2 - 48 * a ** 3 * b ** 2 * c ** 5 * m ** 2 - 24 * a ** 3 * b ** 2 * c ** 3 * l1 ** 4 + 48 * a ** 3 * b ** 2 * c * l1 ** 4 * m ** 2 + 8 * a ** 3 * c ** 7 * m ** 2 - 24 * a ** 3 * c ** 5 * l1 ** 2 * m ** 2 + 24 * a ** 3 * c ** 3 * l1 ** 4 * m ** 2 - 8 * a ** 3 * c * l1 ** 6 * m ** 2 + 5 * a ** 2 * b ** 8 * c ** 2 + 24 * a ** 2 * b ** 6 * c ** 4 - 16 * a ** 2 * b ** 6 * c ** 2 * l1 ** 2 - 96 * a ** 2 * b ** 6 * c ** 2 * m ** 2 + 64 * a ** 2 * b ** 6 * m ** 4 + 34 * a ** 2 * b ** 4 * c ** 6 - 52 * a ** 2 * b ** 4 * c ** 4 * l1 ** 2 + 32 * a ** 2 * b ** 4 * c ** 4 * m ** 2 + 18 * a ** 2 * b ** 4 * c ** 2 * l1 ** 4 + 160 * a ** 2 * b ** 4 * c ** 2 * l1 ** 2 * m ** 2 - 96 * a ** 2 * b ** 4 * c ** 2 * m ** 4 - 96 * a ** 2 * b ** 4 * l1 ** 2 * m ** 4 + 16 * a ** 2 * b ** 2 * c ** 8 - 40 * a ** 2 * b ** 2 * c ** 6 * l1 ** 2 - 64 * a ** 2 * b ** 2 * c ** 6 * m ** 2 + 32 * a ** 2 * b ** 2 * c ** 4 * l1 ** 4 + 128 * a ** 2 * b ** 2 * c ** 4 * l1 ** 2 * m ** 2 + 32 * a ** 2 * b ** 2 * c ** 4 * m ** 4 - 8 * a ** 2 * b ** 2 * c ** 2 * l1 ** 6 - 64 * a ** 2 * b ** 2 * c ** 2 * l1 ** 4 * m ** 2 - 64 * a ** 2 * b ** 2 * c ** 2 * l1 ** 2 * m ** 4 + 32 * a ** 2 * b ** 2 * l1 ** 4 * m ** 4 + a ** 2 * c ** 10 - 4 * a ** 2 * c ** 8 * l1 ** 2 + 6 * a ** 2 * c ** 6 * l1 ** 4 - 4 * a ** 2 * c ** 4 * l1 ** 6 + a ** 2 * c ** 2 * l1 ** 8 - 8 * a * b ** 8 * c ** 3 + 8 * a * b ** 8 * c * m ** 2 - 24 * a * b ** 6 * c ** 5 + 24 * a * b ** 6 * c ** 3 * l1 ** 2 + 88 * a * b ** 6 * c ** 3 * m ** 2 - 24 * a * b ** 6 * c * l1 ** 2 * m ** 2 - 64 * a * b ** 6 * c * m ** 4 - 24 * a * b ** 4 * c ** 7 + 48 * a * b ** 4 * c ** 5 * l1 ** 2 + 88 * a * b ** 4 * c ** 5 * m ** 2 - 24 * a * b ** 4 * c ** 3 * l1 ** 4 - 112 * a * b ** 4 * c ** 3 * l1 ** 2 * m ** 2 - 64 * a * b ** 4 * c ** 3 * m ** 4 + 24 * a * b ** 4 * c * l1 ** 4 * m ** 2 + 64 * a * b ** 4 * c * l1 ** 2 * m ** 4 - 8 * a * b ** 2 * c ** 9 + 24 * a * b ** 2 * c ** 7 * l1 ** 2 + 8 * a * b ** 2 * c ** 7 * m ** 2 - 24 * a * b ** 2 * c ** 5 * l1 ** 4 - 24 * a * b ** 2 * c ** 5 * l1 ** 2 * m ** 2 + 8 * a * b ** 2 * c ** 3 * l1 ** 6 + 24 * a * b ** 2 * c ** 3 * l1 ** 4 * m ** 2 - 8 * a * b ** 2 * c * l1 ** 6 * m ** 2 + b ** 10 * c ** 2 + 4 * b ** 8 * c ** 4 - 4 * b ** 8 * c ** 2 * l1 ** 2 - 16 * b ** 8 * c ** 2 * m ** 2 + 16 * b ** 8 * m ** 4 + 6 * b ** 6 * c ** 6 - 12 * b ** 6 * c ** 4 * l1 ** 2 - 32 * b ** 6 * c ** 4 * m ** 2 + 6 * b ** 6 * c ** 2 * l1 ** 4 + 32 * b ** 6 * c ** 2 * l1 ** 2 * m ** 2 + 32 * b ** 6 * c ** 2 * m ** 4 - 32 * b ** 6 * l1 ** 2 * m ** 4 + 4 * b ** 4 * c ** 8 - 12 * b ** 4 * c ** 6 * l1 ** 2 - 16 * b ** 4 * c ** 6 * m ** 2 + 12 * b ** 4 * c ** 4 * l1 ** 4 + 32 * b ** 4 * c ** 4 * l1 ** 2 * m ** 2 + 16 * b ** 4 * c ** 4 * m ** 4 - 4 * b ** 4 * c ** 2 * l1 ** 6 - 16 * b ** 4 * c ** 2 * l1 ** 4 * m ** 2 - 32 * b ** 4 * c ** 2 * l1 ** 2 * m ** 4 + 16 * b ** 4 * l1 ** 4 * m ** 4 + b ** 2 * c ** 10 - 4 * b ** 2 * c ** 8 * l1 ** 2 + 6 * b ** 2 * c ** 6 * l1 ** 4 - 4 * b ** 2 * c ** 4 * l1 ** 6 + b ** 2 * c ** 2 * l1 ** 8)) / (
                                    a ** 4 + 2 * a ** 2 * b ** 2 - 2 * a ** 2 * c ** 2 - 2 * a ** 2 * l1 ** 2 + b ** 4 - 2 * b ** 2 * c ** 2 - 2 * b ** 2 * l1 ** 2 + c ** 4 - 2 * c ** 2 * l1 ** 2 + l1 ** 4)),
                       sqrt((
                                    a ** 6 + 3 * a ** 4 * b ** 2 - a ** 4 * c ** 2 - 2 * a ** 4 * l1 ** 2 - 8 * a ** 4 * m ** 2 - 16 * a ** 3 * c * m ** 2 + 3 * a ** 2 * b ** 4 - 6 * a ** 2 * b ** 2 * c ** 2 - 4 * a ** 2 * b ** 2 * l1 ** 2 - a ** 2 * c ** 4 - 4 * a ** 2 * c ** 2 * l1 ** 2 - 8 * a ** 2 * c ** 2 * m ** 2 + a ** 2 * l1 ** 4 + 8 * a ** 2 * l1 ** 2 * m ** 2 + 16 * a * b ** 2 * c ** 3 - 16 * a * b ** 2 * c * m ** 2 + b ** 6 - 5 * b ** 4 * c ** 2 - 2 * b ** 4 * l1 ** 2 + 8 * b ** 4 * m ** 2 - 5 * b ** 2 * c ** 4 + 8 * b ** 2 * c ** 2 * m ** 2 + b ** 2 * l1 ** 4 - 8 * b ** 2 * l1 ** 2 * m ** 2 + c ** 6 - 2 * c ** 4 * l1 ** 2 + c ** 2 * l1 ** 4 + 2 * sqrt(
                                a ** 10 * c ** 2 + 8 * a ** 9 * c * m ** 2 + 5 * a ** 8 * b ** 2 * c ** 2 - 4 * a ** 8 * c ** 4 - 4 * a ** 8 * c ** 2 * l1 ** 2 + 16 * a ** 8 * c ** 2 * m ** 2 + 16 * a ** 8 * m ** 4 - 8 * a ** 7 * b ** 2 * c ** 3 + 32 * a ** 7 * b ** 2 * c * m ** 2 - 8 * a ** 7 * c ** 3 * m ** 2 - 24 * a ** 7 * c * l1 ** 2 * m ** 2 + 64 * a ** 7 * c * m ** 4 + 10 * a ** 6 * b ** 4 * c ** 2 + 8 * a ** 6 * b ** 2 * c ** 4 - 16 * a ** 6 * b ** 2 * c ** 2 * l1 ** 2 - 32 * a ** 6 * b ** 2 * c ** 2 * m ** 2 + 64 * a ** 6 * b ** 2 * m ** 4 + 6 * a ** 6 * c ** 6 + 4 * a ** 6 * c ** 4 * l1 ** 2 - 32 * a ** 6 * c ** 4 * m ** 2 + 6 * a ** 6 * c ** 2 * l1 ** 4 - 32 * a ** 6 * c ** 2 * l1 ** 2 * m ** 2 + 96 * a ** 6 * c ** 2 * m ** 4 - 32 * a ** 6 * l1 ** 2 * m ** 4 - 24 * a ** 5 * b ** 4 * c ** 3 + 48 * a ** 5 * b ** 4 * c * m ** 2 + 8 * a ** 5 * b ** 2 * c ** 5 + 24 * a ** 5 * b ** 2 * c ** 3 * l1 ** 2 - 56 * a ** 5 * b ** 2 * c ** 3 * m ** 2 - 72 * a ** 5 * b ** 2 * c * l1 ** 2 * m ** 2 + 64 * a ** 5 * b ** 2 * c * m ** 4 - 8 * a ** 5 * c ** 5 * m ** 2 - 16 * a ** 5 * c ** 3 * l1 ** 2 * m ** 2 + 64 * a ** 5 * c ** 3 * m ** 4 + 24 * a ** 5 * c * l1 ** 4 * m ** 2 - 64 * a ** 5 * c * l1 ** 2 * m ** 4 + 10 * a ** 4 * b ** 6 * c ** 2 + 32 * a ** 4 * b ** 4 * c ** 4 - 24 * a ** 4 * b ** 4 * c ** 2 * l1 ** 2 - 128 * a ** 4 * b ** 4 * c ** 2 * m ** 2 + 96 * a ** 4 * b ** 4 * m ** 4 - 30 * a ** 4 * b ** 2 * c ** 6 - 36 * a ** 4 * b ** 2 * c ** 4 * l1 ** 2 + 32 * a ** 4 * b ** 2 * c ** 4 * m ** 2 + 18 * a ** 4 * b ** 2 * c ** 2 * l1 ** 4 + 96 * a ** 4 * b ** 2 * c ** 2 * l1 ** 2 * m ** 2 - 32 * a ** 4 * b ** 2 * c ** 2 * m ** 4 - 96 * a ** 4 * b ** 2 * l1 ** 2 * m ** 4 - 4 * a ** 4 * c ** 8 + 4 * a ** 4 * c ** 6 * l1 ** 2 + 16 * a ** 4 * c ** 6 * m ** 2 + 4 * a ** 4 * c ** 4 * l1 ** 4 - 32 * a ** 4 * c ** 4 * l1 ** 2 * m ** 2 + 16 * a ** 4 * c ** 4 * m ** 4 - 4 * a ** 4 * c ** 2 * l1 ** 6 + 16 * a ** 4 * c ** 2 * l1 ** 4 * m ** 2 - 32 * a ** 4 * c ** 2 * l1 ** 2 * m ** 4 + 16 * a ** 4 * l1 ** 4 * m ** 4 - 24 * a ** 3 * b ** 6 * c ** 3 + 32 * a ** 3 * b ** 6 * c * m ** 2 - 16 * a ** 3 * b ** 4 * c ** 5 + 48 * a ** 3 * b ** 4 * c ** 3 * l1 ** 2 + 40 * a ** 3 * b ** 4 * c ** 3 * m ** 2 - 72 * a ** 3 * b ** 4 * c * l1 ** 2 * m ** 2 - 64 * a ** 3 * b ** 4 * c * m ** 4 + 8 * a ** 3 * b ** 2 * c ** 7 + 16 * a ** 3 * b ** 2 * c ** 5 * l1 ** 2 - 48 * a ** 3 * b ** 2 * c ** 5 * m ** 2 - 24 * a ** 3 * b ** 2 * c ** 3 * l1 ** 4 + 48 * a ** 3 * b ** 2 * c * l1 ** 4 * m ** 2 + 8 * a ** 3 * c ** 7 * m ** 2 - 24 * a ** 3 * c ** 5 * l1 ** 2 * m ** 2 + 24 * a ** 3 * c ** 3 * l1 ** 4 * m ** 2 - 8 * a ** 3 * c * l1 ** 6 * m ** 2 + 5 * a ** 2 * b ** 8 * c ** 2 + 24 * a ** 2 * b ** 6 * c ** 4 - 16 * a ** 2 * b ** 6 * c ** 2 * l1 ** 2 - 96 * a ** 2 * b ** 6 * c ** 2 * m ** 2 + 64 * a ** 2 * b ** 6 * m ** 4 + 34 * a ** 2 * b ** 4 * c ** 6 - 52 * a ** 2 * b ** 4 * c ** 4 * l1 ** 2 + 32 * a ** 2 * b ** 4 * c ** 4 * m ** 2 + 18 * a ** 2 * b ** 4 * c ** 2 * l1 ** 4 + 160 * a ** 2 * b ** 4 * c ** 2 * l1 ** 2 * m ** 2 - 96 * a ** 2 * b ** 4 * c ** 2 * m ** 4 - 96 * a ** 2 * b ** 4 * l1 ** 2 * m ** 4 + 16 * a ** 2 * b ** 2 * c ** 8 - 40 * a ** 2 * b ** 2 * c ** 6 * l1 ** 2 - 64 * a ** 2 * b ** 2 * c ** 6 * m ** 2 + 32 * a ** 2 * b ** 2 * c ** 4 * l1 ** 4 + 128 * a ** 2 * b ** 2 * c ** 4 * l1 ** 2 * m ** 2 + 32 * a ** 2 * b ** 2 * c ** 4 * m ** 4 - 8 * a ** 2 * b ** 2 * c ** 2 * l1 ** 6 - 64 * a ** 2 * b ** 2 * c ** 2 * l1 ** 4 * m ** 2 - 64 * a ** 2 * b ** 2 * c ** 2 * l1 ** 2 * m ** 4 + 32 * a ** 2 * b ** 2 * l1 ** 4 * m ** 4 + a ** 2 * c ** 10 - 4 * a ** 2 * c ** 8 * l1 ** 2 + 6 * a ** 2 * c ** 6 * l1 ** 4 - 4 * a ** 2 * c ** 4 * l1 ** 6 + a ** 2 * c ** 2 * l1 ** 8 - 8 * a * b ** 8 * c ** 3 + 8 * a * b ** 8 * c * m ** 2 - 24 * a * b ** 6 * c ** 5 + 24 * a * b ** 6 * c ** 3 * l1 ** 2 + 88 * a * b ** 6 * c ** 3 * m ** 2 - 24 * a * b ** 6 * c * l1 ** 2 * m ** 2 - 64 * a * b ** 6 * c * m ** 4 - 24 * a * b ** 4 * c ** 7 + 48 * a * b ** 4 * c ** 5 * l1 ** 2 + 88 * a * b ** 4 * c ** 5 * m ** 2 - 24 * a * b ** 4 * c ** 3 * l1 ** 4 - 112 * a * b ** 4 * c ** 3 * l1 ** 2 * m ** 2 - 64 * a * b ** 4 * c ** 3 * m ** 4 + 24 * a * b ** 4 * c * l1 ** 4 * m ** 2 + 64 * a * b ** 4 * c * l1 ** 2 * m ** 4 - 8 * a * b ** 2 * c ** 9 + 24 * a * b ** 2 * c ** 7 * l1 ** 2 + 8 * a * b ** 2 * c ** 7 * m ** 2 - 24 * a * b ** 2 * c ** 5 * l1 ** 4 - 24 * a * b ** 2 * c ** 5 * l1 ** 2 * m ** 2 + 8 * a * b ** 2 * c ** 3 * l1 ** 6 + 24 * a * b ** 2 * c ** 3 * l1 ** 4 * m ** 2 - 8 * a * b ** 2 * c * l1 ** 6 * m ** 2 + b ** 10 * c ** 2 + 4 * b ** 8 * c ** 4 - 4 * b ** 8 * c ** 2 * l1 ** 2 - 16 * b ** 8 * c ** 2 * m ** 2 + 16 * b ** 8 * m ** 4 + 6 * b ** 6 * c ** 6 - 12 * b ** 6 * c ** 4 * l1 ** 2 - 32 * b ** 6 * c ** 4 * m ** 2 + 6 * b ** 6 * c ** 2 * l1 ** 4 + 32 * b ** 6 * c ** 2 * l1 ** 2 * m ** 2 + 32 * b ** 6 * c ** 2 * m ** 4 - 32 * b ** 6 * l1 ** 2 * m ** 4 + 4 * b ** 4 * c ** 8 - 12 * b ** 4 * c ** 6 * l1 ** 2 - 16 * b ** 4 * c ** 6 * m ** 2 + 12 * b ** 4 * c ** 4 * l1 ** 4 + 32 * b ** 4 * c ** 4 * l1 ** 2 * m ** 2 + 16 * b ** 4 * c ** 4 * m ** 4 - 4 * b ** 4 * c ** 2 * l1 ** 6 - 16 * b ** 4 * c ** 2 * l1 ** 4 * m ** 2 - 32 * b ** 4 * c ** 2 * l1 ** 2 * m ** 4 + 16 * b ** 4 * l1 ** 4 * m ** 4 + b ** 2 * c ** 10 - 4 * b ** 2 * c ** 8 * l1 ** 2 + 6 * b ** 2 * c ** 6 * l1 ** 4 - 4 * b ** 2 * c ** 4 * l1 ** 6 + b ** 2 * c ** 2 * l1 ** 8)) / (
                                    a ** 4 + 2 * a ** 2 * b ** 2 - 2 * a ** 2 * c ** 2 - 2 * a ** 2 * l1 ** 2 + b ** 4 - 2 * b ** 2 * c ** 2 - 2 * b ** 2 * l1 ** 2 + c ** 4 - 2 * c ** 2 * l1 ** 2 + l1 ** 4))]
        l2 = l2_ana_sols[sol_index]

        if self.debug:
            print('l2=', l2)

        xro = (l2 ** 2 - d ** 2) / (4 * m)
        zro = (a * c - m * xro) / k
        yro = yro_sign * sqrt(d ** 2 - (zro - k) ** 2 - (xro - m) ** 2)

        xlo = (d ** 2 - l1 ** 2) / (4 * m)
        zlo = (a * c + m * xlo) / k
        ylo = ylo_sign * sqrt(d ** 2 - (zlo - k) ** 2 - (xlo + m) ** 2)

        # Vr und Vl liegen auf der x-Achse
        Uru = np.array([-xlo, -ylo, zlo])
        Ulu = np.array([-xro, -yro, zro])
        Uro = np.array([xro, yro, zro])
        Ulo = np.array([xlo, ylo, zlo])
        Vr = np.array([m, 0, k])
        Vl = np.array([-m, 0, k])

        if self.debug:
            print(str(round(np.abs(c - (a ** 2 + b ** 2) ** 0.5), 1)), '<= l1  <=',
                  str(round(c + (a ** 2 + b ** 2) ** 0.5, 1)))

        X_Ia = np.vstack((M, Uru, Ulu, Uro, Ulo, Vr, Vl)).astype(np.float32)

        return X_Ia

    def get_l2(self, a, b, c, d, m, k, l1):
        def get_G(l2, yro_sign=1, ylo_sign=1):
            xro = (l2 ** 2 - d ** 2) / (4 * m)  # checked
            zro = (a * c - k ** 2 - m * xro) / k  # checked

            yro_quad = d ** 2 - zro ** 2 - (xro - m) ** 2  # checked
            if yro_quad < 0:
                return np.nan
            yro = yro_sign * np.sqrt(d ** 2 - zro ** 2 - (
                    xro - m) ** 2)  # using np.abs(yro_quad) like before would introduce new wrong solutions

            xlo = (d ** 2 - l1 ** 2) / (4 * m)  # checked
            zlo = (a * c - k ** 2 + m * xlo) / k  # checked

            ylo_quad = d ** 2 - zlo ** 2 - (xlo + m) ** 2  # checked
            if ylo_quad < 0:
                return np.nan
            ylo = ylo_sign * np.sqrt(ylo_quad)  # using np.abs(yro_quad) like before would introduce new wrong solutions

            G = xro * xlo + yro * ylo + (zro + k) * (zlo + k) + (a ** 2 - b ** 2)  # checked
            return G

        l2, yro_sign, ylo_sign = np.nan, 1, 1
        l2 = self.solve_for_l2(get_G, yro_sign=1, ylo_sign=1)
        if np.isnan(l2):
            l2 = self.solve_for_l2(get_G, yro_sign=1, ylo_sign=-1)
        if np.isnan(l2):
            l2 = self.solve_for_l2(get_G, yro_sign=-1, ylo_sign=1)
        if np.isnan(l2):
            l2 = self.solve_for_l2(get_G, yro_sign=-1, ylo_sign=-1)
        print(l2, yro_sign, ylo_sign)
        return l2, yro_sign, ylo_sign

    def solve_for_l2(self, get_G, yro_sign, ylo_sign):
        l2 = np.nan
        l2_max = self.a * 8
        for l2_i in np.linspace(0, l2_max, 100):
            sol = root(get_G, l2_i, tol=1e-6, args=(yro_sign, ylo_sign))
            if sol.success and sol.x[0] >= 0:
                l2 = sol.x[0]
                break
        print('l2_case: yro_sign, ylo_sign=', yro_sign, ylo_sign)
        return l2
