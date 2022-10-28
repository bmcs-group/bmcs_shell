import bmcs_utils.api as bu
import numpy as np
import traits.api as tr

from bmcs_shell.folding.geometry.wb_cell.wb_cell import WBCell
from bmcs_shell.folding.geometry.wb_cell.wb_cell_4p_v2 import WBCell4ParamV2
from bmcs_shell.folding.geometry.wb_tessellation.wb_tessellation_4p import WBTessellation4P


class WBTessellation4PV2Flat(WBTessellation4P):
    name = 'WB Tessellation 4P Flat'

    wb_cell = bu.Instance(WBCell4ParamV2)
    def _wb_cell_default(self):
        wb_cell = WBCell4ParamV2()
        self.update_wb_cell_params(wb_cell)
        return wb_cell

    fix_c = bu.Bool(depends_on='+GEO')
    last_c = bu.Float

    e_x = bu.Float(200, GEO=True) # where a_0 must be < c
    e_x_high = bu.Float(2000)
    c = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_c(self):
        if self.fix_c:
            return self.last_c
        else:
            c = self.a * (1 - np.sin(self.gamma)) / np.cos(self.gamma) ** 2
            # TODO: this round is a workaround because of the spe
            c = 5 * round(c/5)
            self.last_c = c
            return c

    ipw_view = bu.View(
        bu.Item('gamma', latex=r'\gamma', editor=bu.FloatRangeEditor(
            low=1e-6, high=np.pi / 2, n_steps=401, continuous_update=True)),
        bu.Item('a', latex='a^*', editor=bu.FloatRangeEditor(
            low=1e-6, high_name='a_high', n_steps=401, continuous_update=True)),
        bu.Item('b', latex='b', editor=bu.FloatRangeEditor(
            low=1e-6, high_name='b_high', n_steps=401, continuous_update=True)),
        bu.Item('e_x', latex='e_x', editor=bu.FloatRangeEditor(
            low=1e-6, high_name='e_x_high', n_steps=401, continuous_update=True)),
        bu.Item('c', latex='c', editor=bu.FloatEditor(), readonly=True),
        bu.Item('fix_c'),
        *WBCell.ipw_view.content,
        bu.Item('n_phi_plus', latex = r'n_\phi'),
        bu.Item('n_x_plus', latex = r'n_x'),
        bu.Item('show_nodes'),
        bu.Item('trim_half_cells_along_y'),
        bu.Item('trim_half_cells_along_x'),
        bu.Item('align_outer_nodes_along_x'),
    )

    def update_wb_cell_params(self, wb_cell):
        wb_cell.trait_set(
            gamma=self.gamma,
            a=self.a,
            a_high=self.a_high,
            b=self.b,
            b_high=self.b_high,
            c=self.c,
            c_high=self.c_high,
            e_x=self.e_x,
            e_x_high=self.e_x_high,
        )
