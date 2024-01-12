import bmcs_utils.api as bu
import numpy as np
import traits.api as tr

from bmcs_shell.folding.geometry.wb_cell.wb_cell import WBCell
from bmcs_shell.folding.geometry.wb_cell.wb_cell_4p import WBCell4Param


class WBCell4Param2As(WBCell4Param):
    name = 'WBCell4Param2As'

    a_top = bu.Float(1000, GEO=True)
    a_bot = bu.Float(1000, GEO=True)

    ipw_view = bu.View(
        bu.Item('gamma', latex=r'\gamma', editor=bu.FloatSliderEditor(
            low=1e-6, high=np.pi / 2 - 0.0001, n_steps=401, continuous_update=True, readout_format='.3f')),
        bu.Item('a_top', latex=r'a_\mathrm{top}', editor=bu.FloatSliderEditor(
            low=1e-6, high_name='a_high', n_steps=400, continuous_update=True, readout_format='.1f')),
        bu.Item('a_bot', latex=r'a_\mathrm{bot}', editor=bu.FloatSliderEditor(
            low=1e-6, high_name='a_high', n_steps=400, continuous_update=True, readout_format='.1f')),
        bu.Item('b', latex='b', editor=bu.FloatSliderEditor(
            low=1e-6, high_name='b_high', n_steps=400, continuous_update=True, readout_format='.1f')),
        bu.Item('c', latex='c', editor=bu.FloatSliderEditor(
            low=1e-6, high_name='c_high', n_steps=400, continuous_update=True, readout_format='.1f')),
        *WBCell.ipw_view.content,
    )

    X_Ia = tr.Property(depends_on='+GEO')
    '''Array with nodal coordinates I - node, a - dimension
    '''

    @tr.cached_property
    def _get_X_Ia(self):
        gamma = self.gamma

        self.a = self.a_top
        u_2 = self.symb.get_u_2_()
        u_3 = self.symb.get_u_3_()
        U_ur = [self.a_top, u_2, u_3]
        U_ul = [-self.a_top, u_2, u_3]

        self.a = self.a_bot
        u_2 = self.symb.get_u_2_()
        u_3 = self.symb.get_u_3_()
        U_br = [self.a_bot, -u_2, u_3]
        U_bl = [-self.a_bot, -u_2, u_3]
        return np.array([
            [0, 0, 0],
            U_ur,
            U_ul,
            U_br,
            U_bl,
            [self.c * np.sin(gamma), 0, self.c * np.cos(gamma)],  # W0+
            [-self.c * np.sin(gamma), 0, self.c * np.cos(gamma)]  # W0-
        ], dtype=np.float_
        )
