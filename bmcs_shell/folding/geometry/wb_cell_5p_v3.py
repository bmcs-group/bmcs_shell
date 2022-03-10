import bmcs_utils.api as bu
import k3d
import traits.api as tr
import numpy as np
from bmcs_shell.folding.geometry.wb_cell import WBCell
from numpy import sin, cos, sqrt


class WBElem5ParamV3(WBCell):
    name = 'waterbomb cell 5p v3'

    plot_backend = 'k3d'

    debug = False

    m = bu.Float(250, GEO=True)
    l1 = bu.Float(750, GEO=True)
    a = bu.Float(500, GEO=True)
    b = bu.Float(500, GEO=True)
    c = bu.Float(500, GEO=True)

    continuous_update = True

    ipw_view = bu.View(
        bu.Item('m', latex='m', editor=bu.FloatRangeEditor(
            low=0, high=400, n_steps=100, continuous_update=continuous_update)), # TODO: dynamic 0, c ranges
        bu.Item('l1', latex=r'l_1', editor=bu.FloatRangeEditor(
            low=0, high=2000, n_steps=500, continuous_update=continuous_update)),
        bu.Item('a', latex='a', editor=bu.FloatRangeEditor(
            low=1e-6, high=2000, n_steps=100, continuous_update=continuous_update)),
        bu.Item('b', latex='b', editor=bu.FloatRangeEditor(
            low=1e-6, high=2000, n_steps=100, continuous_update=continuous_update)),
        bu.Item('c', latex='c', editor=bu.FloatRangeEditor(
            low=1e-6, high=2000, n_steps=100, continuous_update=continuous_update)),
        *WBCell.ipw_view.content,
    )

    X_Ia = tr.Property(depends_on='+GEO')
    '''Array with nodal coordinates I - node, a - dimension
    '''

    @tr.cached_property
    def _get_X_Ia(self):
        return self.get_cell_vertices()

    def get_cell_vertices(self, a=0.5, b=0.75, c=0.4, gamma=np.pi / 6, beta=np.pi / 3):
        # Control geometry
        a = self.a
        b = self.b
        c = self.c

        # Control folding state
        m = self.m  # x of Vr, -x of Vl
        l1 = self.l1

        # c = a -e
        e = a - c
        # b = sqrt(d**2 - e**2)
        d = sqrt(b ** 2 + e ** 2)
        # k is a short cut
        k = sqrt(c ** 2 - m ** 2)
        M = np.array([0, 0, -k])  # Mittelpunkt

        l2 = ((-8 * a ** 4 * l1 ** 2 + 8 * a ** 4 * d ** 2 + 8 * d ** 2 * e ** 4 - 4 * e ** 4 * l1 ** 2
               - 2 * d ** 4 * l1 ** 2 + d ** 2 * l1 ** 4 + d ** 6 + 16 * a ** 3 * e * l1 ** 2
               - 16 * a ** 3 * d ** 2 * e + 16 * a ** 2 * d ** 2 * e ** 2 - 12 * a ** 2 * e ** 2 * l1 ** 2
               - 16 * a * d ** 2 * e ** 3 + 8 * a * e ** 3 * l1 ** 2 - 32 * a ** 4 * m ** 2 + 32 * a ** 3 * e * m ** 2
               + 2 * a ** 2 * l1 ** 4 - 2 * a * e * l1 ** 4 - 4 * a ** 2 * d ** 2 * l1 ** 2
               + 4 * d ** 2 * e ** 2 * l1 ** 2 + 8 * e ** 2 * l1 ** 2 * m ** 2 - 8 * d ** 2 * l1 ** 2 * m ** 2
               + 8 * a ** 2 * l1 ** 2 * m ** 2
               + 2 * ((4 * a ** 4 - 4 * a ** 3 * e - 3 * a ** 2 * d ** 2 + 4 * a ** 2 * e ** 2 - a ** 2 * l1 ** 2
                       + 4 * a * d ** 2 * e - 4 * a * e ** 3 + d ** 4 - d ** 2 * e ** 2 - d ** 2 * l1 ** 2
                       + e ** 2 * l1 ** 2) * (4 * a ** 2 - 4 * a * e + d ** 2 - l1 ** 2) * (
                              a ** 2 * d ** 4 - 2 * a ** 2 * d ** 2 * l1 ** 2 - 8 * a ** 2 * d ** 2 * m ** 2
                              + 16 * a ** 2 * e ** 2 * m ** 2 + a ** 2 * l1 ** 4 - 8 * a ** 2 * l1 ** 2 * m ** 2
                              + 16 * a ** 2 * m ** 4 - 2 * a * d ** 4 * e + 4 * a * d ** 2 * e * l1 ** 2
                              + 24 * a * d ** 2 * e * m ** 2 - 32 * a * e ** 3 * m ** 2 - 2 * a * e * l1 ** 4
                              + 8 * a * e * l1 ** 2 * m ** 2 + d ** 4 * e ** 2 - 2 * d ** 2 * e ** 2 * l1 ** 2
                              - 16 * d ** 2 * e ** 2 * m ** 2 + 16 * d ** 2 * m ** 4 + 16 * e ** 4 * m ** 2
                              + e ** 2 * l1 ** 4 - 16 * e ** 2 * m ** 4)) ** (
                       1 / 2) + 10 * a * d ** 4 * e - 8 * a ** 2 * d ** 2 * m ** 2 - 8 * d ** 2 * e ** 2 * m ** 2
               + 8 * d ** 4 * m ** 2 - 2 * a ** 2 * d ** 4 - 8 * d ** 4 * e ** 2) / (
                      2 * a * e - 2 * a * l1 + d ** 2 - 2 * e ** 2 + 2 * e * l1 - l1 ** 2) / (
                      2 * a * e + 2 * a * l1 + d ** 2 - 2 * e ** 2 - 2 * e * l1 - l1 ** 2)) ** (1 / 2)
        if self.debug:
            print('l2=', l2)

        xro = (l2 ** 2 - d ** 2) / (4 * m)
        zro = (a * c - c ** 2 + m ** 2 - 1 / 4 * (l2 ** 2 - d ** 2)) / k
        yro = sqrt((c - a) ** 2 + b ** 2 - zro ** 2 - (xro - m) ** 2)

        xlo = -(l1 ** 2 - d ** 2) / (4 * m)
        zlo = (a * c - c ** 2 + m ** 2 + m * xlo) / k
        ylo = sqrt(l1 ** 2 - zlo ** 2 - (xlo - m) ** 2)

        # Vr und Vl liegen auf der x-Achse
        Uru = np.array([-xlo, -ylo, zlo])
        Ulu = np.array([-xro, -yro, zro])
        Uro = np.array([xro, yro, zro])
        Ulo = np.array([xlo, ylo, zlo])
        Vr = np.array([m, 0, 0])
        Vl = np.array([-m, 0, 0])

        if self.debug:
            print(str(round(np.abs(c - (a ** 2 + b ** 2) ** 0.5), 1)), '<= l1  <=', str(round(c + (a ** 2 + b ** 2) ** 0.5, 1)))

        X_Ia = np.vstack((M, Uru, Ulu, Uro, Ulo, Vr, Vl)).astype(np.float32)

        return X_Ia
