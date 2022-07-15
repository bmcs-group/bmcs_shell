import bmcs_utils.api as bu
import numpy as np
import traits.api as tr

from bmcs_shell.folding.geometry.wb_cell.wb_cell import WBCell
from bmcs_shell.folding.geometry.wb_cell.wb_cell_4p import WBCell4Param


class WBCell4ParamFlat(WBCell4Param):
    name = 'Waterbomb cell 4p flat'

    c = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_c(self):
        # The following is obtained from the condition that all z coords of cell
        # vertices should be equal (except center)
        return self.a * (1 - np.sin(self.gamma)) / np.cos(self.gamma) ** 2

    ipw_view = bu.View(
        bu.Item('gamma', latex=r'\gamma', editor=bu.FloatRangeEditor(
            low=1e-6, high=np.pi / 2, n_steps=401, continuous_update=True)),
        bu.Item('a', latex='a', editor=bu.FloatRangeEditor(
            low=1e-6, high_name='a_high', n_steps=401, continuous_update=True)),
        bu.Item('b', latex='b', editor=bu.FloatRangeEditor(
            low=1e-6, high_name='b_high', n_steps=401, continuous_update=True)),
        bu.Item('c', latex='c', readonly=True, editor=bu.FloatEditor()),
        *WBCell.ipw_view.content,
    )
