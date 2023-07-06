import bmcs_utils.api as bu
import numpy as np
import traits.api as tr

from bmcs_shell.folding.geometry.wb_cell.wb_cell import WBCell
from bmcs_shell.folding.geometry.wb_tessellation.wb_tessellation_4p import WBTessellation4P


class WBTessellation4PSS(WBTessellation4P):
    name = 'WB Tessellation 4P SS'

    fix_c = bu.Bool(depends_on='+GEO')
    last_c = bu.Float

    c = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_c(self):
        if self.fix_c:
            return self.last_c
        else:
            c = self.a / np.sin(self.gamma)
            # TODO: this round is a workaround because the wb_cell will accept only 5-multiplication c values
            #  (c_max = 2000 and it has 400 steps)
            c = 5 * round(c/5)
            self.last_c = c
            return c

    ipw_view = bu.View(
        bu.Item('gamma', latex=r'\gamma', editor=bu.FloatRangeEditor(
            low=1e-6, high=np.pi / 2, n_steps=401, continuous_update=True)),
        bu.Item('a', latex='a', editor=bu.FloatRangeEditor(
            low=1e-6, high_name='a_high', n_steps=401, continuous_update=True)),
        bu.Item('b', latex='b', editor=bu.FloatRangeEditor(
            low=1e-6, high_name='b_high', n_steps=401, continuous_update=True)),
        bu.Item('c', latex='c', editor=bu.FloatEditor(), readonly=True),
        bu.Item('fix_c'),
        *WBCell.ipw_view.content,
        bu.Item('n_phi_plus', latex = r'n_\phi'),
        bu.Item('n_x_plus', latex = r'n_x'),
        bu.Item('show_nodes'),
        bu.Item('trim_half_cells_along_y'),
        bu.Item('trim_half_cells_along_x'),
    )