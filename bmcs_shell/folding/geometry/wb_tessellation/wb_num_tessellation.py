import bmcs_utils.api as bu

from bmcs_shell.folding.geometry.wb_tessellation.wb_num_tessellation_base import WBNumTessellationBase
import numpy as np

class WBNumTessellation(WBNumTessellationBase):
    name = 'WBNumTessellation'

    n_y = bu.Int(3, GEO=True)
    n_x = bu.Int(3, GEO=True)

    ipw_view = bu.View(
        *WBNumTessellationBase.ipw_view.content,
        bu.Item('n_x', latex=r'n_x'),
        bu.Item('n_y', latex=r'n_y'),
    )

    def calc_mesh_for_tessellated_cells(self):
        # TODO: the resulting mesh_X_nmIa, mesh_I_Fi are just summing up all cells, repeation deletion is needed to use
        #  it in analysis
        I_Fi = self.I_Fi
        X_Ia = self.X_Ia

        y_base_cell_X_Ia = X_Ia
        next_y_base_cell_X_Ia = X_Ia
        base_cell_X_Ia = X_Ia
        next_base_cell_X_Ia = X_Ia

        n_y, n_x = self.n_y, self.n_x

        mesh_X_nmIa = np.zeros((n_y, n_x, 7, 3))

        for i in range(n_y):
            i_row_is_even = (i + 1) % 2 == 0

            add_br = True  # to switch between adding br and ur
            add_bl = True  # to switch between adding bl and ul

            for j in range(n_x):
                j_is_even = (j + 1) % 2 == 0

                if j == 0:
                    mesh_X_nmIa[i, j, ...] = base_cell_X_Ia
                    continue
                if j_is_even:
                    # Number of cell_to_add is even (add right from base cell)
                    if add_br:
                        cell_to_add = self._get_br_X_Ia(base_cell_X_Ia)
                        mesh_X_nmIa[i, j, ...] = cell_to_add
                    else:
                        cell_to_add = self._get_ur_X_Ia(base_cell_X_Ia)
                        mesh_X_nmIa[i, j, ...] = cell_to_add
                    add_br = not add_br
                    base_cell_X_Ia = next_base_cell_X_Ia
                    next_base_cell_X_Ia = cell_to_add
                else:
                    # Number of cell_to_add is odd (add left from base cell)
                    if add_bl:
                        cell_to_add = self._get_bl_X_Ia(base_cell_X_Ia)
                        mesh_X_nmIa[i, j, ...] = cell_to_add
                    else:
                        cell_to_add = self._get_ul_X_Ia(base_cell_X_Ia)
                        mesh_X_nmIa[i, j, ...] = cell_to_add
                    add_bl = not add_bl
                    base_cell_X_Ia = next_base_cell_X_Ia
                    next_base_cell_X_Ia = cell_to_add

            if i_row_is_even:
                # Next row is odd (change y_base_cell_X_Ia to a cell below base cell)
                base_cell_X_Ia = self._get_bl_X_Ia(self._get_br_X_Ia(next_y_base_cell_X_Ia))
                next_base_cell_X_Ia = base_cell_X_Ia
                next_y_base_cell_X_Ia = y_base_cell_X_Ia
                y_base_cell_X_Ia = base_cell_X_Ia
            else:
                # Next row is even (change y_base_cell_X_Ia to a cell above base cell)
                base_cell_X_Ia = self._get_ul_X_Ia(self._get_ur_X_Ia(next_y_base_cell_X_Ia))
                next_base_cell_X_Ia = base_cell_X_Ia
                next_y_base_cell_X_Ia = y_base_cell_X_Ia
                y_base_cell_X_Ia = base_cell_X_Ia

        indices_of_cells_to_skip = self._get_indices_of_cells_to_skip()

        # TODO: here the coodinates of indicies of cells to skip are set simply to zero and not eliminated!
        #  when exporting geometry for analysis they need to be completly eliminated
        mesh_X_nmIa[-1, indices_of_cells_to_skip, :, :] = 0
        mesh_X_Oa = mesh_X_nmIa.reshape((n_y * n_x * 7, 3))

        # Calculating mesh_I_Fi
        seven_mult_i = 7 * np.arange(n_x * n_y)
        seven_mult_Iab = seven_mult_i[:, np.newaxis, np.newaxis] + np.zeros((n_x * n_y, 6, 3), dtype=np.int32)
        mesh_I_Fi = np.full((n_x * n_y, 6, 3), I_Fi)
        mesh_I_Fi = mesh_I_Fi + seven_mult_Iab
        mesh_I_Fi.reshape((n_x * n_y * 6, 3))

        return mesh_X_Oa, mesh_I_Fi

    def _get_indices_of_cells_to_skip(self):
        """ This function will return indices of the cells in the last added row which needs to be eliminated
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
            self.I_Fi_shell = I_Fi.reshape((I_Fi.shape[0] * I_Fi.shape[1], 3))

            X_Ia = X_Ia.astype(np.float32)
            I_Fi = I_Fi.astype(np.uint32)

            self.k3d_mesh['wb_tess_mesh'].vertices = X_Ia
            self.k3d_mesh['wb_tess_mesh'].indices = I_Fi
            self.k3d_wireframe['wb_tess_mesh'].vertices = X_Ia
            self.k3d_wireframe['wb_tess_mesh'].indices = I_Fi
            self.k3d_wireframe['wb_tess_mesh'].width = self.wireframe_width
        else:
            self.setup_plot(pb)
