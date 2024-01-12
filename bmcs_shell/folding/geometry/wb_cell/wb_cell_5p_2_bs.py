import bmcs_utils.api as bu
import k3d
import traits.api as tr
import numpy as np
from bmcs_shell.folding.geometry.wb_cell.wb_cell import WBCell
from numpy import sin, cos, sqrt, tan
from scipy.optimize import fsolve, least_squares , minimize

# TODO
class WBCell5Param2Bs(WBCell4Param):
    name = 'WBCell5Param2Bs'

    plot_backend = 'k3d'

    gamma = bu.Float(np.pi / 6, GEO=True)
    a = bu.Float(1000., GEO=True)
    b1 = bu.Float(1000., GEO=True)
    b2 = bu.Float(1000., GEO=True)
    c = bu.Float(1000., GEO=True)
    symmetric = bu.Bool(True, GEO=True)

    continuous_update = True

    ipw_view = bu.View(
        bu.Item('gamma', latex=r'\gamma', editor=bu.FloatRangeEditor(
            low=1e-6, high=np.pi / 2, n_steps=201, continuous_update=continuous_update)),
        bu.Item('a', latex='a', editor=bu.FloatRangeEditor(
            low=1e-6, high=2000, n_steps=201, continuous_update=continuous_update)),
        bu.Item('b1', latex='b_1', editor=bu.FloatRangeEditor(
            low=1e-6, high=2000, n_steps=201, continuous_update=continuous_update)),
        bu.Item('b2', latex='b_2', editor=bu.FloatRangeEditor(
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
        gamma = self.gamma
        u_2 = self.symb.get_u_2_()
        u_3 = self.symb.get_u_3_()
        return np.array([
            [self.e_x, 0, 0],  # O_r
            [-self.e_x, 0, 0],  # O_l
            [self.e_x +self.a, u_2, u_3],  # U++
            [-self.e_x-self.a, u_2, u_3],  # U-+
            [self.e_x +self.a, -u_2, u_3],  # U+-
            [-self.e_x-self.a, -u_2, u_3],  # U--
            [self.e_x + self.c * np.sin(gamma), 0, self.c * np.cos(gamma)],  # W0+
            [-self.e_x -self.c * np.sin(gamma), 0, self.c * np.cos(gamma)]  # W0-
        ], dtype=np.float_
        )
