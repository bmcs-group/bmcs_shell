import bmcs_utils.api as bu
from bmcs_shell.folding.geometry.wb_cell.wb_cell_4p_v2 import WBCell4ParamV2
from bmcs_shell.folding.geometry.wb_tessellation.wb_tessellation_4p import WBTessellation4P


class WBTessellation4PV2(WBTessellation4P):
    name = 'WB Tessellation 4P V2'

    wb_cell = bu.Instance(WBCell4ParamV2)

    def _wb_cell_default(self):
        wb_cell = WBCell4ParamV2()
        self.update_wb_cell_params(wb_cell)
        return wb_cell

    e_x = bu.Float(200, GEO=True) # where a_0 must be < c
    e_x_high = bu.Float(2000)

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

    ipw_view = bu.View(
        # bu.Item('wb_cell'),
        *WBCell4ParamV2.ipw_view.content,
        bu.Item('n_phi_plus', latex = r'n_\phi'),
        bu.Item('n_x_plus', latex = r'n_x'),
        bu.Item('show_nodes'),
        bu.Item('trim_half_cells_along_y'),
        bu.Item('trim_half_cells_along_x'),
        bu.Item('align_outer_nodes_along_x'),
    )
