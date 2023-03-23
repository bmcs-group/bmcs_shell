import bmcs_utils.api as bu
import numpy as np
import traits.api as tr

from bmcs_shell.folding.geometry.wb_tessellation.wb_num_tessellation_grad_base import WBNumTessellationGradBase

"""
A class for a waterbomb cell tessellation that uses a list of different but compatible wb cells arranged gradually
along the x axis to form the tessellation.
"""
class WBNumTessellationGrad(WBNumTessellationGradBase):
    name = 'WBNumTessellationGrad'

    n_y = bu.Int(2, GEO=True)
    # n_x = bu.Int(2, GEO=True)

    n_x = tr.Property(depends_on='wb_cells')
    @tr.cached_property
    def _get_n_x(self):
        return len(self.wb_cells)

    ipw_view = bu.View(
        # *WBTessellationBase.ipw_view.content,
        bu.Item('wb_cells'),
        # bu.Item('n_x', latex=r'n_x', readonly=True),
        bu.Item('n_y', latex=r'n_y'),
    )

    def calc_mesh_for_tessellated_cells(self):
        # TODO: the resulting mesh_X_nmIa, mesh_I_Fi are just summing up all cells, repeation deletion is needed to use
        #  it in analysis
        I_Fi = self.wb_cells[0].I_Fi
        X_Ia = self.wb_cells[0].X_Ia
        y_base_cell_X_Ia = X_Ia
        base_cell_X_Ia = X_Ia
        n_y, n_x = self.n_y, self.n_x
        mesh_X_nmIa = np.zeros((n_y, n_x, 7, 3))

        for i in range(n_y):
            print('i = ', i) if self.debug else None
            add_br = True  # to switch between adding br and ur
            for j in range(n_x):
                print(' j = ', j) if self.debug else None
                if j == 0:
                    print('  add_base') if self.debug else None
                    mesh_X_nmIa[i, j, ...] = base_cell_X_Ia
                    continue
                X1_Ia = base_cell_X_Ia
                X2_Ia = self.wb_cells[j].X_Ia
                if add_br:
                    print('  add_br') if self.debug else None
                    cell_to_add = self._get_br_X_Ia(X1_Ia, X2_Ia=X2_Ia)
                    mesh_X_nmIa[i, j, ...] = cell_to_add
                else:
                    print('  add_ur') if self.debug else None
                    cell_to_add = self._get_ur_X_Ia(X1_Ia, X2_Ia=X2_Ia)
                    mesh_X_nmIa[i, j, ...] = cell_to_add
                add_br = not add_br
                base_cell_X_Ia = cell_to_add

            X1_Ia = y_base_cell_X_Ia
            X2_Ia = self.wb_cells[1].X_Ia
            X0_Ia = self.wb_cells[0].X_Ia
            ur_cell = self._get_ur_X_Ia(X1_Ia, X2_Ia=X2_Ia)
            y_base_cell_X_Ia = self._get_ul_X_Ia(ur_cell, X2_Ia=X0_Ia)
            base_cell_X_Ia = y_base_cell_X_Ia

        indices_of_cells_to_skip = self._get_indices_of_cells_to_skip()

        # TODO: here the coodinates of indicies of cells to skip are set simply to zero and not eliminated!
        #  when exporting geometry for analysis they need to be completly eliminated

        #     mesh_X_nmIa[-1, indices_of_cells_to_skip, :, :] = 0
        mesh_X_Oa = mesh_X_nmIa.reshape((n_y * n_x * 7, 3))

        # Calculating mesh_I_Fi
        seven_mult_i = 7 * np.arange(n_x * n_y)
        seven_mult_Iab = seven_mult_i[:, np.newaxis, np.newaxis] + np.zeros((n_x * n_y, 6, 3), dtype=np.int32)
        mesh_I_Fi = np.full((n_x * n_y, 6, 3), I_Fi)
        mesh_I_Fi = mesh_I_Fi + seven_mult_Iab
        mesh_I_Fi.reshape((n_x * n_y * 6, 3))

        return mesh_X_Oa, mesh_I_Fi

    def _get_indices_of_cells_to_skip(self):
        """ This function will return indices of the cells in the last added row (along y) which needs to be eliminated
        in order to have a symmetric shell """
        n_x = self.n_x
        indices = np.concatenate((np.arange(1, n_x, 4), np.arange(2, n_x, 4)))
        indices = np.sort(indices)

        if self.n_y % 2 == 0:
            a = np.arange(n_x)
            indices = np.delete(a, indices)
        return indices

    # Plotting ##########################################################################

    def setup_plot(self, pb):
        self.pb = pb
        pb.clear_fig()
        X_Ia, I_Fi = self.calc_mesh_for_tessellated_cells()
        self.add_cell_to_pb(pb, X_Ia, I_Fi, 'wb_tess_mesh')

    def update_plot(self, pb):
        if self.k3d_mesh:
            X_Ia, I_Fi  = self.calc_mesh_for_tessellated_cells()

            # TODO: make this cleaner
            self.X_Ia_shell = X_Ia
            self.I_Fi_shell = I_Fi

            X_Ia = X_Ia.astype(np.float32)
            I_Fi = I_Fi.astype(np.uint32)

            self.k3d_mesh['wb_tess_mesh'].vertices = X_Ia
            self.k3d_mesh['wb_tess_mesh'].indices = I_Fi
            self.k3d_wireframe['wb_tess_mesh'].vertices = X_Ia
            self.k3d_wireframe['wb_tess_mesh'].indices = I_Fi
            self.k3d_wireframe['wb_tess_mesh'].width = self.wireframe_width
        else:
            self.setup_plot(pb)

