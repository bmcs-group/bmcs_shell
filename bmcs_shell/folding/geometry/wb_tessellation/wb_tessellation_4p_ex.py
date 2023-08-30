import bmcs_utils.api as bu
from bmcs_shell.folding.geometry.wb_cell.wb_cell_4p_ex import WBCell4ParamEx
from bmcs_shell.folding.geometry.wb_tessellation.wb_tessellation_4p import WBTessellation4P
import traits.api as tr
import numpy as np


class WBTessellation4PEx(WBTessellation4P):
    name = 'WBTessellation4PEx'

    wb_cell = bu.Instance(WBCell4ParamEx)

    def _wb_cell_default(self):
        wb_cell = WBCell4ParamEx()
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
        *WBCell4ParamEx.ipw_view.content,
        *WBTessellation4P.ipw_view_items
    )

    def _get_idx_of_facets_to_trim(self):
        along_y_first_cell = (0, 1, 5, 7)
        along_y_last_cell = (2, 3, 4, 6)
        along_x_first_cell = (6, 7)
        along_x_last_cell = (4, 5)
        return along_x_first_cell, along_x_last_cell, along_y_first_cell, along_y_last_cell


    X_Ia_no_constraint = tr.Property(depends_on='+GEO')
    '''Array with nodal coordinates I - node, a - dimension
    '''
    @tr.cached_property
    def _get_X_Ia_no_constraint(self):
        idx_unique, _ = self.unique_node_map
        X_Ia = self.X_cells_Ia[idx_unique]
        if self.trim_half_cells_along_x:
            _, cells_out_xyj = self.cells_in_out_xyj
            X_Ia[cells_out_xyj[-1, :, 0]] = (X_Ia[cells_out_xyj[-1, :, 0]] + X_Ia[cells_out_xyj[-1, :, 1]]) / 2
            X_Ia[cells_out_xyj[-1, :, 4]] = (X_Ia[cells_out_xyj[-1, :, 4]] + X_Ia[cells_out_xyj[-1, :, 5]]) / 2
            X_Ia[cells_out_xyj[0, :, 1]] = (X_Ia[cells_out_xyj[0, :, 0]] + X_Ia[cells_out_xyj[0, :, 1]]) / 2
            X_Ia[cells_out_xyj[0, :, 5]] = (X_Ia[cells_out_xyj[0, :, 4]] + X_Ia[cells_out_xyj[0, :, 5]]) / 2
        return X_Ia