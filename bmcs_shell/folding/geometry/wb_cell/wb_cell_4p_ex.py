import bmcs_utils.api as bu
from bmcs_shell.folding.geometry.wb_cell.wb_cell import WBCell

import traits.api as tr
import numpy as np

from bmcs_shell.folding.geometry.wb_cell.wb_cell_4p import WBCell4Param


class WBCell4ParamEx(WBCell4Param):
    name = 'WBCell4ParamEx'

    c = bu.Float(800, GEO=True)
    e_x = bu.Float(200, GEO=True) # where e_x must be < c
    e_x_high = bu.Float(2000)

    ipw_view = bu.View(
        bu.Item('gamma', latex=r'\gamma', editor=bu.FloatRangeEditor(
            low=1e-6, high=np.pi / 2 - 0.001, n_steps=401, continuous_update=True)),
        bu.Item('a', latex='a^*', editor=bu.FloatRangeEditor(
            low=1e-6, high_name='a_high', n_steps=401, continuous_update=True)),
        bu.Item('b', latex='b', editor=bu.FloatRangeEditor(
            low=1e-6, high_name='b_high', n_steps=401, continuous_update=True)),
        bu.Item('c', latex='c', editor=bu.FloatRangeEditor(
            low=1e-6, high_name='c_high', n_steps=401, continuous_update=True)),
        bu.Item('e_x', latex='e_x', editor=bu.FloatRangeEditor(
            low=1e-6, high_name='e_x_high', n_steps=401, continuous_update=True)),
        *WBCell.ipw_view.content,
    )


    X_Ia = tr.Property(depends_on='+GEO')
    '''Array with nodal coordinates I - node, a - dimension
    '''

    @tr.cached_property
    def _get_X_Ia(self):
        print('X_Ia - WBCell4ParamEx')
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

    I_Fi = tr.Property
    '''Triangle mapping '''
    @tr.cached_property
    def _get_I_Fi(self):
        print('I_Fi - WBCell4ParamEx')
        return np.array([[0, 2, 1],
                         [1, 2, 3],
                         [0, 1, 4],
                         [1, 5, 4],
                         [0, 4, 6],
                         [0, 6, 2],
                         [1, 7, 5],
                         [1, 3, 7],
                         ]).astype(np.int32)

    delta_x = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_delta_x(self):
        print('delta_x - WBCell4ParamEx')
        return self.symb.get_delta_x() + 2 * self.e_x

