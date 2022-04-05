import bmcs_utils.api as bu

from bmcs_shell.folding.geometry.wb_tessellation.wb_num_tessellation_base import WBNumTessellationBase


class WBNumTessellation(WBNumTessellationBase):
    name = 'WBNumTessellation'

    n_y = bu.Int(3, GEO=True)
    n_x = bu.Int(3, GEO=True)

    ipw_view = bu.View(
        *WBNumTessellationBase.ipw_view.content,
        bu.Item('n_x', latex=r'n_x'),
        bu.Item('n_y', latex=r'n_y'),
    )

    # Plotting ##########################################################################

    def setup_plot(self, pb):
        self.pb = pb
        # TODO, save all cells to be plotted in an array, adapt indicies and plot it as one 3D mesh
        #  then make it also available for updates in the update_plot function
        pb.clear_fig()
        I_Fi = self.I_Fi
        X_Ia = self.X_Ia

        y_base_cell_X_Ia = X_Ia
        next_y_base_cell_X_Ia = X_Ia
        base_cell_X_Ia = X_Ia
        next_base_cell_X_Ia = X_Ia

        mesh_X_Ia = []
        mesh_I_Fi = []
        for i in range(self.n_y):
            i_row_is_even = (i + 1) % 2 == 0

            add_br = True  # to switch between adding br and ur
            add_bl = True  # to switch between adding bl and ul

            for j in range(self.n_x):
                if j == 0:
                    self.add_cell_to_pb(pb, base_cell_X_Ia, I_Fi, '')
                    continue
                if (j + 1) % 2 == 0:
                    # Number of cell_to_add is even (add right from base cell)
                    if add_br:
                        cell_to_add = self._get_br_X_Ia(base_cell_X_Ia)
                        self.add_cell_to_pb(pb, cell_to_add, I_Fi, '')
                    else:
                        cell_to_add = self._get_ur_X_Ia(base_cell_X_Ia)
                        self.add_cell_to_pb(pb, cell_to_add, I_Fi, '')
                    add_br = not add_br
                    base_cell_X_Ia = next_base_cell_X_Ia
                    next_base_cell_X_Ia = cell_to_add
                else:
                    # Number of cell_to_add is odd (add left from base cell)
                    if add_bl:
                        cell_to_add = self._get_bl_X_Ia(base_cell_X_Ia)
                        self.add_cell_to_pb(pb, cell_to_add, I_Fi, '')
                    else:
                        cell_to_add = self._get_ul_X_Ia(base_cell_X_Ia)
                        self.add_cell_to_pb(pb, cell_to_add, I_Fi, '')
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

    def update_plot(self, pb):
        self.setup_plot(pb)
        # if self.k3d_mesh:
        #     pass
        #     # self.k3d_mesh['X_Ia'].vertices = self.X_Ia.astype(np.float32)
        #     # self.k3d_mesh['br_X_Ia'].vertices = self._get_br_X_Ia(self.X_Ia,
        #     #                                                      self.rot_br if self.investigate_rot else -sol[
        #     #                                                          0]).astype(np.float32)
        #     # self.k3d_mesh['ur_X_Ia'].vertices = self._get_ur_X_Ia(self.X_Ia,
        #     #                                                      self.rot_ur if self.investigate_rot else -sol[
        #     #                                                          1]).astype(np.float32)
        #     # self.k3d_wireframe['X_Ia'].vertices = self.X_Ia.astype(np.float32)
        #     # self.k3d_wireframe['br_X_Ia'].vertices = self._get_br_X_Ia(self.X_Ia,
        #     #                                                           self.rot_br if self.investigate_rot else -sol[
        #     #                                                               0]).astype(np.float32)
        #     # self.k3d_wireframe['ur_X_Ia'].vertices = self._get_ur_X_Ia(self.X_Ia,
        #     #                                                           self.rot_ur if self.investigate_rot else -sol[
        #     #                                                               1]).astype(np.float32)
        # else:
        #     self.setup_plot(pb)
