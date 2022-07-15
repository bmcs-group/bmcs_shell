import bmcs_utils.api as bu
from bmcs_shell.folding.geometry.wb_cell.wb_cell import WBCell

import traits.api as tr
import numpy as np

from bmcs_shell.folding.geometry.wb_cell.wb_cell_4p import WBCell4Param


class WBCell4ParamV2(WBCell4Param):
    name = 'Waterbomb cell 4p v2'

    c = bu.Float(800, GEO=True)
    a_o = bu.Float(200, GEO=True) # where a_0 must be < c
    a_o_high = bu.Float(2000)

    ipw_view = bu.View(
        bu.Item('gamma', latex=r'\gamma', editor=bu.FloatRangeEditor(
            low=1e-6, high=np.pi / 2, n_steps=401, continuous_update=True)),
        bu.Item('a', latex='a', editor=bu.FloatRangeEditor(
            low=1e-6, high_name='a_high', n_steps=401, continuous_update=True)),
        bu.Item('b', latex='b', editor=bu.FloatRangeEditor(
            low=1e-6, high_name='b_high', n_steps=401, continuous_update=True)),
        bu.Item('c', latex='c', editor=bu.FloatRangeEditor(
            low=1e-6, high_name='c_high', n_steps=401, continuous_update=True)),
        bu.Item('a_o', latex='a_o', editor=bu.FloatRangeEditor(
            low=1e-6, high_name='a_o_high', n_steps=401, continuous_update=True)),
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
            [self.a_o, 0, 0],  # O_r
            [-self.a_o, 0, 0],  # O_l
            [self.a, u_2, u_3],  # U++
            [-self.a, u_2, u_3],  # U-+
            [self.a, -u_2, u_3],  # U+-
            [-self.a, -u_2, u_3],  # U--
            [self.a_o + self.c * np.sin(gamma), 0, self.c * np.cos(gamma)],  # W0+
            [-self.a_o -self.c * np.sin(gamma), 0, self.c * np.cos(gamma)]  # W0-
        ], dtype=np.float_
        )

    I_Fi = tr.Property
    '''Triangle mapping '''
    @tr.cached_property
    def _get_I_Fi(self):
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
        return self.symb.get_delta_x() + self.a_o
