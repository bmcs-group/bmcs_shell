import json
import time

import bmcs_utils.api as bu
import k3d
import numpy as np
import traits.api as tr
import matplotlib.pyplot as plt

from bmcs_shell.folding.geometry.wb_cell.wb_cell_4p import \
    WBCell4Param, axis_angle_to_q, qv_mult
from bmcs_shell.folding.geometry.wb_geo_utils import WBGeoUtils

from bmcs_shell.folding.utils.dihedral_angles import get_dih_angles

class WBTessellation4P(bu.Model):
    name = 'WB Tessellation 4P'

    wb_cell = bu.Instance(WBCell4Param)

    def _wb_cell_default(self):
        #print('_wb_cell_default')   
        wb_cell = WBCell4Param()
        self.update_wb_cell_params(wb_cell)
        return wb_cell

    tree = ['wb_cell']

    plot_backend = 'k3d'

    n_phi_plus = bu.Int(5, GEO=True)
    n_x_plus = bu.Int(3, GEO=True)
    gamma = bu.Float(1.25, GEO=True)
    a = bu.Float(1000, GEO=True)
    a_high = bu.Float(2000)
    b = bu.Float(1000, GEO=True)
    b_high = bu.Float(2000)
    c = bu.Float(1000, GEO=True)
    c_high = bu.Float(2000)

    wireframe_width = bu.Float(15)
    show_wireframe = bu.Bool(True, GEO=True)
    show_nodes = bu.Bool(False, GEO=True)
    show_node_labels = bu.Bool(False, GEO=True)
    WIREFRAME = 'k3d_mesh_wireframe'
    NODES = 'k3d_nodes'
    NODES_LABELS = 'k3d_nodes_labels'

    constraint_node_idx = bu.Int(0, GEO=True)
    constraint_coord_idx = bu.Int(0, GEO=True) # 0 for x, 1 for y, 2 for z, -1 for complete fix

    trim_half_cells_along_y = bu.Bool(False, GEO=True)
    trim_half_cells_along_x = bu.Bool(False, GEO=True)

    plot_points_diff_btn = bu.Bool()
    show_folding_path_btn = bu.Bool()
    export_obj_file_btn = bu.Bool()
    export_fold_file_btn = bu.Bool()
    # show_folding_path_btn = bu.Button()
    # export_obj_file_btn = bu.Button()
    # export_fold_file_btn = bu.Button()

    @tr.observe('show_folding_path_btn')
    def show_folding_path_btn_click(self, event=None):
        ##print('show_folding_path_btn_click - WB Tessellation 4P')
        # Plotting
        k3d_name = 'folding_path'
        if k3d_name in self.pb.objects:
            self.pb.clear_object(k3d_name)

        if not self.show_folding_path_btn:
            return

        gamma_tmp = self.gamma
        n_gamma = 30
        X_gIa = np.zeros((n_gamma, *self.X_Ia_trimmed.shape))
        for i, gamma in enumerate(np.linspace(np.pi / 2 - 0.001, self.gamma, n_gamma)):
            self.gamma = gamma
            X_gIa[i, ...] = self.X_Ia_trimmed
        self.gamma = gamma_tmp

        self.pb.objects[k3d_name] = []

        Xg_Ia = X_gIa.reshape((X_gIa.shape[0] * X_gIa.shape[1], X_gIa.shape[2]))
        plt_points = k3d.points(positions=Xg_Ia,
                                point_size=70,
                                shader='3d',
                                color=0x3f6bc5)
        self.pb.objects[k3d_name].append(plt_points)
        self.pb.plot_fig += plt_points

        base_lines = k3d.lines(X_gIa[0, :, :], self.I_Fi_trimmed,
                               shader='mesh', width=30,
                               color=0xff0000)
        self.pb.objects[k3d_name].append(base_lines)
        self.pb.plot_fig += base_lines


    @tr.observe('+GEO', post_init=True)
    def update_wb_cell(self, event):
        ##print('update_wb_cell - WB Tessellation 4P')
        self.update_wb_cell_params(self.wb_cell)

    def update_wb_cell_params(self, wb_cell):
        ##print('update_wb_cell_params - WB Tessellation 4P')
        wb_cell.trait_set(
            gamma=self.gamma,
            a=self.a,
            a_high=self.a_high,
            b=self.b,
            b_high=self.b_high,
            c=self.c,
            c_high=self.c_high,
        )

    ipw_view_items = [
        bu.Item('n_phi_plus', latex = r'n_\phi'),
        bu.Item('n_x_plus', latex = r'n_x'),
        bu.Item('show_nodes'),
        bu.Item('trim_half_cells_along_y'),
        bu.Item('trim_half_cells_along_x'),
        bu.Item('constraint_node_idx', latex=r'n_{i\mathrm{, node}}'),
        bu.Item('constraint_coord_idx', latex=r'c_{i\mathrm{, constraint}}'),
        # The following buttons are not showing due to a problem in bmcs_utils left panel: only button are
        # not pushing the panel to grow
        # bu.Item('show_folding_path_btn', editor=bu.ButtonEditor(icon='eye', label='Show folding path')),
        # bu.Item('export_obj_file_btn', editor=bu.ButtonEditor(icon='file', label='Export .obj')),
        # bu.Item('export_fold_file_btn', editor=bu.ButtonEditor(icon='file', label='Export .fold')),
        bu.Item('show_folding_path_btn'),
        bu.Item('export_obj_file_btn'),
        bu.Item('export_fold_file_btn'),
        bu.Item('plot_points_diff_btn'),
    ]

    ipw_view = bu.View(
        *WBCell4Param.ipw_view.content,
        *ipw_view_items,
    )

    def get_phi_range(self, delta_phi):
        ##print('get_phi_range - WB Tessellation 4P')
        return np.arange(-(self.n_phi_plus - 1), self.n_phi_plus) * delta_phi

    def get_X_phi_range(self,delta_phi, R_0):
        ##print('get_X_phi_range - WB Tessellation 4P')
        """Given an array of angles and radius return an array of coordinates
        """
        phi_range = self.get_phi_range((delta_phi))
        return np.array([np.fabs(R_0) * np.sin(phi_range),
                         np.fabs(R_0) * np.cos(phi_range) + R_0]).T

    def get_X_x_range(self,delta_x):
        ##print('get_X_x_range - WB Tessellation 4P')
        return np.arange(-(self.n_x_plus - 1), self.n_x_plus) * delta_x

    cell_map = tr.Property
    def _get_cell_map(self):
        ##print('cell_map - WB Tessellation 4P')
        delta_x = self.wb_cell.delta_x
        delta_phi = self.wb_cell.delta_phi
        R_0 = self.wb_cell.R_0

        X_x_range = self.get_X_x_range(delta_x)
        X_phi_range = self.get_X_phi_range(delta_phi, R_0)
        n_idx_x = len(X_x_range)
        n_idx_phi = len(X_phi_range)
        idx_x = np.arange(n_idx_x)
        idx_phi = np.arange(n_idx_phi)

        idx_x_ic = idx_x[(n_idx_x) % 2::2]
        idx_x_id = idx_x[(n_idx_x + 1) % 2::2]
        idx_phi_ic = idx_phi[(n_idx_phi) % 2::2]
        idx_phi_id = idx_phi[(n_idx_phi + 1) % 2::2]

        n_ic = len(idx_x_ic) * len(idx_phi_ic)
        n_id = len(idx_x_id) * len(idx_phi_id)

        n_cells = n_ic + n_id
        return n_cells, n_ic, n_id, idx_x_ic, idx_x_id, idx_phi_ic, idx_phi_id

    n_cells = tr.Property
    def _get_n_cells(self):
        #print('n_cells - WB Tessellation 4P')
        n_cells, _, _, _, _, _, _ = self.cell_map
        return n_cells

    X_Ia_const_change = tr.Property(depends_on='constraint_node_idx, constraint_coord_idx')
    @tr.cached_property
    def _get_X_Ia_const_change(self):
        #print('X_Ia_const_change - WB Tessellation 4P')
        return np.copy(self.X_Ia_no_constraint)

    X_cells_Ia = tr.Property(depends_on='+GEO')
    '''Array with nodal coordinates of uncoupled cells
    I - node, a - dimension
    '''
    @tr.cached_property
    def _get_X_cells_Ia(self):
        #print('X_cells_Ia - WB Tessellation 4P')

        delta_x = self.wb_cell.delta_x
        delta_phi = self.wb_cell.delta_phi
        R_0 = self.wb_cell.R_0

        X_Ia_wb_rot = np.copy(self.wb_cell.X_Ia)
        X_Ia_wb_rot[...,2] -= R_0
        X_cIa = np.array([X_Ia_wb_rot], dtype=np.float_)
        rotation_axes = np.array([[1, 0, 0]], dtype=np.float_)
        rotation_angles = self.get_phi_range(delta_phi)
        q = axis_angle_to_q(rotation_axes, rotation_angles)
        X_dIa = qv_mult(q, X_cIa)
        X_dIa[...,2] += R_0

        X_x_range = self.get_X_x_range(delta_x)
        X_phi_range = self.get_X_phi_range(delta_phi, R_0)
        n_idx_x = len(X_x_range)
        n_idx_phi = len(X_phi_range)
        idx_x = np.arange(n_idx_x)
        idx_phi = np.arange(n_idx_phi)

        idx_x_ic = idx_x[(n_idx_x) % 2::2]
        idx_x_id = idx_x[(n_idx_x + 1) % 2::2]
        idx_phi_ic = idx_phi[(n_idx_phi) % 2::2]
        idx_phi_id = idx_phi[(n_idx_phi + 1) % 2::2]

        X_E = X_x_range[idx_x_ic]
        X_F = X_x_range[idx_x_id]

        X_CIa = X_dIa[idx_phi_ic]
        X_DIa = X_dIa[idx_phi_id]

        expand = np.array([1,0,0])
        X_E_a = np.einsum('i,j->ij', X_E, expand)
        X_ECIa = X_CIa[np.newaxis,:,:,:] + X_E_a[:,np.newaxis,np.newaxis,:]
        X_F_a = np.einsum('i,j->ij', X_F, expand)
        X_FDIa = X_DIa[np.newaxis,:,:,:] + X_F_a[:,np.newaxis,np.newaxis,:]

        X_Ia = np.vstack([X_ECIa.flatten().reshape(-1,3), X_FDIa.flatten().reshape(-1,3)])
        return X_Ia

    I_cells_Fi = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_I_cells_Fi(self):
        #print('I_cells_Fi - WB Tessellation 4P')
        I_Fi_cell = self.wb_cell.I_Fi
        n_I_cell = self.wb_cell.n_I
        n_cells = self.n_cells
        i_range = np.arange(n_cells) * n_I_cell
        I_Fi = (I_Fi_cell[np.newaxis,:,:] + i_range[:, np.newaxis, np.newaxis]).reshape(-1, 3)
        return I_Fi

    I_cells_Li = tr.Property(depends_on='+GEO')
    ''' Lines-node mapping (uncombined)'''
    @tr.cached_property
    def _get_I_cells_Li(self):
        #print('I_cells_Li - WB Tessellation 4P')
        return self._get_shell_lines_uncombined_I_Li(self.wb_cell.I_Li)

    I_cells_V_Li = tr.Property(depends_on='+GEO')
    ''' Valley lines-node mapping (uncombined)'''
    @tr.cached_property
    def _get_I_cells_V_Li(self):
        #print('I_cells_V_Li - WB Tessellation 4P')
        return self._get_shell_lines_uncombined_I_Li(self.wb_cell.I_V_Li)

    I_V_Li = tr.Property(depends_on='+GEO')
    ''' Valley lines-node mapping '''
    def _get_I_V_Li(self):
        #print('I_V_Li - WB Tessellation 4P')
        _, idx_remap = self.unique_node_map
        I_Li = np.unique(np.sort(idx_remap[self.I_cells_V_Li], axis=1), axis=0)
        return I_Li

    I_cells_M_Li = tr.Property(depends_on='+GEO')
    ''' Mountain lines-node mapping (uncombined)'''
    @tr.cached_property
    def _get_I_cells_M_Li(self):
        #print('I_cells_M_Li - WB Tessellation 4P')
        return self._get_shell_lines_uncombined_I_Li(self.wb_cell.I_M_Li)

    I_M_Li = tr.Property(depends_on='+GEO')
    ''' Mountain lines-node mapping '''
    def _get_I_M_Li(self):
        #print('I_M_Li - WB Tessellation 4P')
        _, idx_remap = self.unique_node_map
        I_Li = np.unique(np.sort(idx_remap[self.I_cells_M_Li], axis=1), axis=0)
        return I_Li

    def _get_shell_lines_uncombined_I_Li(self, I_Li_cell):
        #print('_get_shell_lines_uncombined_I_Li - WB Tessellation 4P')
        n_I_cell = self.wb_cell.n_I
        n_cells = self.n_cells
        i_range = np.arange(n_cells) * n_I_cell
        I_Li = (I_Li_cell[np.newaxis, :, :] + i_range[:, np.newaxis, np.newaxis]).reshape(-1, 2)
        return I_Li


    X_Ia_no_constraint = tr.Property(depends_on='+GEO')
    '''Array with nodal coordinates I - node, a - dimension
    '''
    @tr.cached_property
    def _get_X_Ia_no_constraint(self):
        #print('X_Ia_no_constraint - WB Tessellation 4P')
        idx_unique, _ = self.unique_node_map
        X_Ia = self.X_cells_Ia[idx_unique]
        if self.trim_half_cells_along_x:
            _, cells_out_xyj = self.cells_in_out_xyj
            X_Ia[cells_out_xyj[-1, :, 3]] = (X_Ia[cells_out_xyj[-1, :, 3]] + X_Ia[cells_out_xyj[-1, :, 4]]) / 2
            X_Ia[cells_out_xyj[0, :, 4]] = (X_Ia[cells_out_xyj[0, :, 3]] + X_Ia[cells_out_xyj[0, :, 4]]) / 2
        return X_Ia

    X_Ia = tr.Property(depends_on='+GEO')
    '''Array with nodal coordinates I - node, a - dimension
    '''
    @tr.cached_property
    def _get_X_Ia(self):
        #print('X_Ia - WB Tessellation 4P')
        X_Ia = self.X_Ia_no_constraint
        if self.constraint_coord_idx == 0 and self.constraint_node_idx == 0:
            return X_Ia
        else:
            X_Ia_const_change = self.X_Ia_const_change
            coord_idx = self.constraint_coord_idx
            node_idx = self.constraint_node_idx
            const_X_Ia = np.copy(X_Ia)
            if coord_idx == -1:
                diff = X_Ia[node_idx, :] - X_Ia_const_change[node_idx, :]
                const_X_Ia[:, :] = X_Ia[:, :] - diff[np.newaxis]
            else:
                diff = X_Ia[node_idx, coord_idx] - X_Ia_const_change[node_idx, coord_idx]
                const_X_Ia[:, coord_idx] = X_Ia[:, coord_idx] - diff[np.newaxis]
            return const_X_Ia

    X_Ia_trimmed = tr.Property(depends_on='+GEO')
    '''Array with nodal coordinates I - node, a - dimension
    '''
    @tr.cached_property
    def _get_X_Ia_trimmed(self):
        #print('X_Ia_trimmed - WB Tessellation 4P')
        return self.X_Ia[np.sort(np.unique(self.I_Fi.flatten()))] if self.is_trimmed else self.X_Ia

    I_Li = tr.Property(depends_on='+GEO')
    '''Lines-node mapping
    '''
    def _get_I_Li(self):
        #print('I_Li - WB Tessellation 4P')
        _, idx_remap = self.unique_node_map
        I_Li = np.unique(np.sort(idx_remap[self.I_cells_Li], axis=1), axis=0)
        return I_Li

    I_Fi_ = tr.Property(depends_on='+GEO')
    '''Facet - node mapping
    '''
    def _get_I_Fi_(self):
        #print('I_Fi_ - WB Tessellation 4P')
        _, idx_remap = self.unique_node_map
        return idx_remap[self.I_cells_Fi]

    I_Fi_trimmed = tr.Property(depends_on='+GEO')
    '''Facet - node mapping
    '''
    @tr.cached_property
    def _get_I_Fi_trimmed(self):
        #print('I_Fi_trimmed - WB Tessellation 4P')
        if self.is_trimmed:
            I_Fi = self.I_Fi
            # Reindexing I_Fi to match the new X_Ia (after trimming)
            I_Fi_shape = I_Fi.shape
            I_Fi_flat = I_Fi.flatten()

            old_indices = np.sort(np.unique(I_Fi.flatten()))
            new_indices = np.arange(old_indices.size)

            I_Fi_flat_reindexed = np.copy(I_Fi_flat)
            for old_index, new_index in zip(old_indices, new_indices):
                I_Fi_flat_reindexed[I_Fi_flat == old_index] = new_index
            return I_Fi_flat_reindexed.reshape(I_Fi_shape)
        else:
            return self.I_Fi

    I_Fi = tr.Property(depends_on='+GEO')
    '''Facet - node mapping
    '''
    @tr.cached_property
    def _get_I_Fi(self):
        #print('I_Fi - WB Tessellation 4P')
        _, idx_remap = self.unique_node_map
        I_Fi = idx_remap[self.I_cells_Fi]

        if self.trim_half_cells_along_y or self.trim_half_cells_along_x:
            cells_in_xyfi_, cells_out_xyfi_ = self.cells_in_out_xyfi
            cells_in_xyfi = np.copy(cells_in_xyfi_)
            cells_out_xyfi = np.copy(cells_out_xyfi_)
            along_x_first_cell, along_x_last_cell, along_y_first_cell, along_y_last_cell = self._get_idx_of_facets_to_trim()
            if self.trim_half_cells_along_y:
                # Set extended facets to -1 indicies (to mark it for later removal)
                cells_out_xyfi[:, 0, along_y_first_cell, :] = -1
                cells_out_xyfi[:, -1, along_y_last_cell, :] = -1
            if self.trim_half_cells_along_x:
                # Remove half cells along x
                cells_out_xyfi[0, :, along_x_first_cell, :] = -1
                cells_out_xyfi[-1, :, along_x_last_cell, :] = -1

            # Delete extended facets
            I_Fi = np.vstack((cells_in_xyfi.reshape((-1, 3)), cells_out_xyfi.reshape((-1, 3))))
            F_m = I_Fi.flatten()
            I_Fi = np.delete(F_m, np.where(F_m == -1)).reshape((-1, 3))
        return I_Fi

    def _get_idx_of_facets_to_trim(self):
        #print('_get_idx_of_facets_to_trim - WB Tessellation 4P')
        along_y_first_cell = (0, 2, 4)
        along_y_last_cell = (1, 3, 5)
        along_x_first_cell = (4, 5)
        along_x_last_cell = (2, 3)
        return along_x_first_cell, along_x_last_cell, along_y_first_cell, along_y_last_cell

    cells_in_out_xyfi = tr.Property(depends_on='+GEO')
    ''' Convenience indexing for inner and outer cells in the tessellation where (x, y cell index along x and y; 
    f for cell facet; i facet nodes indices)'''
    @tr.cached_property
    def _get_cells_in_out_xyfi(self):
        #print('cells_in_out_xyfi - WB Tessellation 4P')
        mesh_elem_num = self.cell_mesh_surf_elem_num
        F_cfi = self.F_cfi
        n_x_in, n_x_out, n_y_in, n_y_out, cells_in_indices, cells_out_indices = self.cells_in_out_info
        cells_out_cfi = F_cfi[cells_out_indices]
        cells_out_xyfi = cells_out_cfi.reshape((n_x_out, n_y_out, mesh_elem_num, 3))
        cells_in_cfi = F_cfi[cells_in_indices]
        cells_in_xyfi = cells_in_cfi.reshape((n_x_in, n_y_in, mesh_elem_num, 3))
        return cells_in_xyfi, cells_out_xyfi

    cell_mesh_surf_elem_num = tr.Property()
    def _get_cell_mesh_surf_elem_num(self):
        #print('cell_mesh_surf_elem_num - WB Tessellation 4P')
        return self.wb_cell.I_Fi.shape[0]

    cells_in_out_xyj = tr.Property(depends_on='+GEO')
    ''' Convenience indexing for inner and outer cells in the tessellation where (x, y cell index along x and y; 
    j is cell nodes indices in order)'''
    @tr.cached_property
    def _get_cells_in_out_xyj(self):
        #print('cells_in_out_xyj - WB Tessellation 4P')
        F_cfi = self.F_cfi
        n_x_in, n_x_out, n_y_in, n_y_out, cells_in_indices, cells_out_indices = self.cells_in_out_info
        I_cj = np.unique(F_cfi.reshape((F_cfi.shape[0], -1)), axis=1)
        cells_out_cj = I_cj[cells_out_indices]
        cells_in_cj = I_cj[cells_in_indices]
        cells_out_xyj = cells_out_cj.reshape((n_x_out, n_y_out, self.cell_node_num))
        cells_in_xyj = cells_in_cj.reshape((n_x_in, n_y_in, self.cell_node_num))
        return cells_in_xyj, cells_out_xyj

    cell_node_num = tr.Property()
    def _get_cell_node_num(self):
        #print('cell_node_num - WB Tessellation 4P')
        return self.wb_cell.X_Ia.shape[0]

    cells_in_out_info = tr.Property(depends_on='+GEO')
    ''' n_x_in, n_x_out, n_y_in, n_y_out are number of inner and outer cells along x and y'''
    @tr.cached_property
    def _get_cells_in_out_info(self):
        #print('cells_in_out_info - WB Tessellation 4P')
        n_x_real = 2 * self.n_x_plus - 1
        n_y_real = self.n_phi_plus
        n_x_in = int(n_x_real / 2)
        n_x_out = n_x_in + 1
        n_y_in = n_y_real - 1
        n_y_out = n_y_real
        cells_num_in = n_x_in * n_y_in
        cells_num_out = n_x_out * n_y_out
        cells_in_indices = np.arange(cells_num_in)
        cells_out_indices = np.arange(cells_num_in, cells_num_in + cells_num_out)
        return n_x_in, n_x_out, n_y_in, n_y_out, cells_in_indices, cells_out_indices

    F_cfi = tr.Property(depends_on='+GEO')
    ''' Convenience indexing where (c cell index, f facet index, i indices of facet's nodes)'''
    @tr.cached_property
    def _get_F_cfi(self):
        #print('F_cfi - WB Tessellation 4P')
        # c cell index, f facet index, i indices of facet's nodes
        I_Fi = self.I_Fi_
        return I_Fi.reshape((self.n_cells, self.cell_mesh_surf_elem_num, 3))

    node_match_threshold = tr.Property(depends_on='+GEO')

    def _get_node_match_threshold(self):
        #print('node_match_threshold - WB Tessellation 4P')
        min_length = np.min([self.a, self.b, self.c])
        return min_length * 1e-3

    unique_node_map = tr.Property(depends_on='+GEO')
    '''Property containing the mapping between the crease pattern nodes
    with duplicate nodes and pattern with compressed nodes array.
    The criterion for removing a node is geometric, the threshold
    is specified in node_match_threshold.
    '''
    def _get_unique_node_map(self):
        #print('unique_node_map - WB Tessellation 4P')
        # reshape the coordinates in array of segments to the shape (n_N, n_D
        x_0 = self.X_cells_Ia
        # construct distance vectors between every pair of nodes
        x_x_0 = x_0[:, np.newaxis, :] - x_0[np.newaxis, :, :]
        # calculate the distance between every pair of nodes
        dist_0 = np.sqrt(np.einsum('...i,...i', x_x_0, x_x_0))
        # identify those at the same location
        zero_dist = dist_0 < self.node_match_threshold
        # get their indices
        i_idx, j_idx = np.where(zero_dist)
        # take only the upper triangle indices
        upper_triangle = i_idx < j_idx
        idx_multi, idx_delete = i_idx[upper_triangle], j_idx[upper_triangle]
        # construct a boolean array with True at valid and False at deleted
        # indices
        idx_unique = np.ones((len(x_0),), dtype='bool')
        idx_unique[idx_delete] = False
        # Boolean array of nodes to keep - includes both those that
        # are unique and redirection nodes to be substituted for duplicates
        idx_keep = np.ones((len(x_0),), dtype=np.bool_)
        idx_keep[idx_delete] = False
        # prepare the enumeration map map
        ij_map = np.ones_like(dist_0, dtype=np.int_) + len(x_0)
        i_ = np.arange(len(x_0))
        # indexes of nodes that are being kept
        idx_row = i_[idx_keep]
        # enumerate the kept nodes by putting their number onto the diagonal
        ij_map[idx_keep, idx_keep] = np.arange(len(idx_row))
        # broadcast the substitution nodes into the interaction positions
        ij_map[i_idx, j_idx] = ij_map[i_idx, i_idx]
        # get the substitution node by picking up the minimum index within ac column
        idx_remap = np.min(ij_map, axis=0)

        return idx_unique, idx_remap

    I_CDij = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_I_CDij(self):
        #print('I_CDij - WB Tessellation 4P')
        n_cells, n_ic, n_id, _, x_cell_idx, _, y_cell_idx = self.cell_map
        x_idx, y_idx = x_cell_idx / 2, y_cell_idx / 2
        n_x_, n_y_ = len(x_idx), len(y_idx)
        I_cell_offset = (n_ic + np.arange(n_x_ * n_y_).reshape(n_x_, n_y_)) * self.wb_cell.n_I
        I_CDij_map = (I_cell_offset.T[:, :, np.newaxis, np.newaxis] +
                      self.wb_cell.I_boundary[np.newaxis, np.newaxis, :, :])
        return I_CDij_map

    is_trimmed = tr.Property(depends_on='+GEO')

    @tr.cached_property
    def _get_is_trimmed(self):
        #print('is_trimmed - WB Tessellation 4P')
        return self.trim_half_cells_along_y or self.trim_half_cells_along_x

    def setup_plot(self, pb):
        #print('setup_plot - WB Tessellation 4P')
        self.pb = pb
        X_Ia = self.X_Ia_trimmed.astype(np.float32)
        I_Fi = self.I_Fi_trimmed.astype(np.uint32)

        I_M = self.I_CDij[(0, -1), :, (0, -1), :]
        _, idx_remap = self.unique_node_map
        J_M = idx_remap[I_M]
        X_Ma = self.X_Ia[J_M.flatten()]

        k3d_mesh = k3d.mesh(X_Ia,
                            I_Fi,
                            color=0x999999,
                            side='double',
                            opacity=1
                            )

        pb.objects['k3d_mesh'] = k3d_mesh
        pb.plot_fig += k3d_mesh

        if self.show_nodes:
            self._add_nodes_to_fig(pb, X_Ma)

        if self.wb_cell.show_node_labels:
            self._add_nodes_labels_to_fig(pb, X_Ia)

        if self.show_wireframe:
            self._add_wireframe_to_fig(pb, X_Ia, I_Fi)

    def update_plot(self, pb):
        #print('update_plot - WB Tessellation 4P')
        X_Ia = self.X_Ia_trimmed.astype(np.float32)
        I_Fi = self.I_Fi_trimmed.astype(np.uint32)

        I_M = self.I_CDij[(0, -1),:,(0, -1),:]
        _, idx_remap = self.unique_node_map
        J_M = idx_remap[I_M]
        X_Ma = self.X_Ia[J_M.flatten()]

        mesh = pb.objects['k3d_mesh']
        mesh.vertices = X_Ia
        mesh.indices = I_Fi

        if self.show_nodes:
            if self.NODES in pb.objects:
                pb.objects[self.NODES].positions = X_Ma
            else:
                self._add_nodes_to_fig(pb, X_Ma)
        else:
            if self.NODES in pb.objects:
                pb.clear_object(self.NODES)

        if self.show_wireframe:
            if self.WIREFRAME in pb.objects:
                wireframe = pb.objects[self.WIREFRAME]
                wireframe.width = self.wireframe_width
                wireframe.vertices = X_Ia
                wireframe.indices = I_Fi
            else:
                self._add_wireframe_to_fig(pb, X_Ia, I_Fi)
        else:
            if self.WIREFRAME in pb.objects:
                pb.clear_object(self.WIREFRAME)

        if self.show_node_labels:
            if self.NODES_LABELS in pb.objects:
                pb.clear_object(self.NODES_LABELS)
            self._add_nodes_labels_to_fig(pb, X_Ia)
        else:
            if self.NODES_LABELS in pb.objects:
                pb.clear_object(self.NODES_LABELS)

    def _add_nodes_labels_to_fig(self, pb, X_Ia):
        #print('_add_nodes_labels_to_fig - WB Tessellation 4P')
        node_indicies_I = np.arange(X_Ia.shape[0])
        node_indicies_str_list = [str(idx) for idx in node_indicies_I]
        k3d_text = k3d.text(node_indicies_str_list, position=X_Ia.flatten(), label_box=False, size=0.8, color=0x00FF00)
        pb.plot_fig += k3d_text
        pb.objects[self.NODES_LABELS] = k3d_text

    def _add_wireframe_to_fig(self, pb, X_Ia, I_Fi):
        #print('_add_wireframe_to_fig - WB Tessellation 4P')
        k3d_mesh_wireframe = k3d.lines(X_Ia,
                                      I_Fi,
                                    shader='mesh',
                                      color=0x000000,
                                      width=self.wireframe_width)
        pb.plot_fig += k3d_mesh_wireframe
        pb.objects[self.WIREFRAME] = k3d_mesh_wireframe

    def _add_nodes_to_fig(self, pb, X_Ma):
        #print('_add_nodes_to_fig - WB Tessellation 4P')
        k3d_points = k3d.points(X_Ma, point_size=300)
        pb.objects[self.NODES] = k3d_points
        pb.plot_fig += k3d_points

    def _show_or_hide_fig_object(self, pb, show_obj, obj_name, obj_add_fun, obj_update_fun):
        #print('_show_or_hide_fig_object - WB Tessellation 4P')
        if show_obj:
            if obj_name in pb.objects:
                obj_update_fun()
            else:
                obj_add_fun()
        else:
            if obj_name in pb.objects:
                pb.clear_object(obj_name)

    # def _get_min_z_disp(self):
    #     gamma_tmp = self.gamma
    #
    #     X_final_Ia = self.X_Ia_trimmed
    #     self.gamma = np.pi / 2 - 0.0001
    #     X_0_Ia = self.X_Ia_trimmed
    #     X_diff_Ia = X_final_Ia - X_0_Ia
    #     z_diff_I = np.abs(X_diff_Ia[:, 2])
    #     sorted_z_diff_indices = np.argsort(z_diff_I)
    #     sorted_z_diff_values = np.sort(z_diff_I)
    #
    #     self.gamma = gamma_tmp
    #
    #     # the point at the first few points should be where supports are needed in the formwork
    #     return sorted_z_diff_indices, sorted_z_diff_values

    def plot_formwork_plan(self, type='folded', ax=None, trimmed=False, gamma=None):
        #print('plot_formwork_plan - WB Tessellation 4P')
        """
        :param type: 'fixed_base' , 'moving_top', 'pattern', 'folded'
        :param ax:
        :param trimmed:
        :param gamma:
        :return:
        """
        if gamma is None:
            gamma = self.gamma

        if type == 'pattern':
            self.gamma = np.pi/2-0.0001
            gamma = self.gamma

        fig = None
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_size_inches(8.5, 8.5)
        ax.axis('equal')

        self.plot_folding_pattern(trimmed=trimmed, ax=ax,
                                  gamma=np.pi/2-0.0001, color='black' if type == 'pattern' else 'silver', view='top')
        if type != 'pattern':
            self.plot_folding_pattern(trimmed=trimmed, ax=ax, gamma=gamma, color='red', view='top')

        gamma_temp = self.gamma
        self.gamma = gamma

        self.plot_formwork_points(ax=ax, type=type)

        self.gamma = gamma_temp

        if fig is not None:
            fig.show()
            return fig, ax

    def plot_formwork_points(self, ax=None, type='fixed_base'):
        #print('plot_formwork_points - WB Tessellation 4P')
        fig = None
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_size_inches(8.5, 8.5)

        x = self.X_Ia_trimmed[:, 0]
        y = self.X_Ia_trimmed[:, 1]
        z_orig = self.X_Ia_trimmed[:, 2]
        z = np.copy(z_orig)
        n = np.arange(x.size)

        # Shift z coords so that the point with minimum z displacement have z=0
        z_min_disp_idx = np.argmin(np.abs(z))
        z = z - z[z_min_disp_idx]

        if type == 'fixed_base':
            #print('Fixed nodes list (fixed on bottom fixed plate):')
            supp_points_indices = np.where(z < 0)[0]
            supp_points_values = z[z < 0]
            supp_points_values = supp_points_values - np.min(supp_points_values)
            n = n[supp_points_indices]
            x = x[supp_points_indices]
            y = y[supp_points_indices]
            z = supp_points_values
        elif type == 'moving_top':
            #print('Moving nodes list (must be raised to fold the shell):')
            moved_points_indices = np.where(z >= 0)[0]
            n = n[moved_points_indices]
            x = x[moved_points_indices]
            y = y[moved_points_indices]
            z = z[moved_points_indices]
        elif type == 'pattern' or type == 'folded':
            #print('Pattern points (in flat unfolded state):')
            z = z_orig
        #print('-----------------------------------------------------------')

        #print('Node num.: Coords. in folded state (x, y, z_diff)')
        #     #print('|Node num. |x|y|')
        #     #print('|:-|:-:|:-:|')

        anno_ratio = int(np.round(0.03 * (np.max(x) - np.min(x))))
        for i, x_i, y_i, z_i in zip(n, x, y, z):
            x_ro = int(np.round(x_i))
            y_ro = int(np.round(y_i))
            z_ro = int(np.round(z_i))
            ax.annotate(str(i), (x_ro, y_ro + anno_ratio), color='red' if z_i > 0 else 'black' if z_i == 0 else 'blue')
            ax.plot(x_i, y_i, 'o', color='red' if z_i > 0 else 'black' if z_i == 0 else 'blue')

            #print(str(i) + ' :\t(' + str(x_ro) + ',\t' + str(y_ro) + ',\t' + str(z_ro) + ')')
        #         #print('|' + str(i) + '|' + str(x_anno) + '|' + str(y_anno) + '|')

        if fig is not None:
            fig.show()
            return fig, ax

    def plot_folding_pattern(self, trimmed=False, ax=None,
                             gamma=np.pi/2-0.0001,
                             color='black',
                             view='top'):
        #print('plot_folding_pattern - WB Tessellation 4P')
        """
        :trimmed (boolean): False: full original tessellation, True: if some cells are trimmed, plot the trimmed tessellation.
        However, the trimmed variant doesn't distinguish valley and mountain folds for now.
        """
        gamma_temp = self.gamma
        self.gamma = gamma
        #print('Plot tessellation with gamma=', round(np.rad2deg(gamma), 2), '°')
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_size_inches(8.5, 8.5)

        if view == 'top':
            coord_1, coord_2 = 0, 1
        elif view == 'side':
            coord_1, coord_2 = 1, 2
        elif view == 'front':
            coord_1, coord_2 = 0, 2
        else:
            raise Exception('No valid value has been provided for argument "view"')

        if trimmed:
            X_I1 = self.X_Ia_trimmed[:, coord_1]
            X_I2 = self.X_Ia_trimmed[:, coord_2]
            triangles = self.I_Fi_trimmed
            ax.triplot(X_I1, X_I2, triangles, lw=1.2, c=color)
        else:
            V_lines = self.X_Ia[self.I_V_Li]
            M_lines = self.X_Ia[self.I_M_Li]
            for i_line in range(V_lines.shape[0]):
                ax.plot(V_lines[i_line, :, coord_1], V_lines[i_line, :, coord_2], '--', c=color)
            for i_line in range(M_lines.shape[0]):
                ax.plot(M_lines[i_line, :, coord_1], M_lines[i_line, :, coord_2], c=color)

        ax.axis('equal')
        self.gamma = gamma_temp
        if 'fig' in locals():
            fig.show()
            return fig, ax

    def get_dih_angles(self):
        #print('get_dih_angles - WB Tessellation 4P')
        return get_dih_angles(self.X_Ia_trimmed, self.I_Fi_trimmed)

    def export_fold_file(self, path=None):
        #print('export_fold_file - WB Tessellation 4P')
        # See https://github.com/edemaine/fold/blob/master/doc/spec.md for fold file specification
        # Viewer: https://edemaine.github.io/fold/examples/foldviewer.html

        output_data = {
            "file_spec": 1,
            "file_creator": "BMCS software suite",
            "file_author": "RWTH Aachen - Institute of Structural Concrete",
            "file_title": "Preliminary Base",
            "file_classes": ["singleModel"],
            "frame_title": "Preliminary Base Crease Pattern",
            "frame_classes": ["creasePattern"],
            "vertices_coords": self.X_Ia.tolist(),
            "faces_vertices": self.I_Fi.tolist(),
            # To be completed
        }

        if path is None:
            path = time.strftime("%Y%m%d-%H%M%S") + '-shell.fold'

        with open(path, 'w') as outfile:
            json.dump(output_data, outfile, sort_keys=True, indent=4)

    @tr.observe('plot_points_diff_btn')
    def plot_points_diff(self, event=None):
        #print('plot_points_diff - WB Tessellation 4P')
        gamma_tmp = self.gamma
        self.gamma = np.pi / 2 - 0.0001
        X_Ia0 = self.X_Ia_trimmed
        self.gamma = gamma_tmp
        X_Ia1 = self.X_Ia_trimmed
        X_Ia_diff = X_Ia1 - X_Ia0
        #print('Node num.: Coords. in folded state (x_diff, y_diff, z_diff)')
        for i, (x_i, y_i, z_i) in zip(np.arange(X_Ia_diff.shape[0]), X_Ia_diff):
            x_ro = int(np.round(x_i))
            y_ro = int(np.round(y_i))
            z_ro = int(np.round(z_i))
            #print(str(i) + ' :\t(' + str(x_ro) + ',\t' + str(y_ro) + ',\t' + str(z_ro) + ')')

    def get_file_name(self):
        #print('get_file_name - WB Tessellation 4P')
        file_name = 'a_' + str(np.round(self.a, 1)) + '_b_' + str(np.round(self.b, 1))
        file_name += '_c_' + str(np.round(self.c, 1)) + '_gamma_' + str(np.round(self.gamma, 2))
        file_name += '_n_x_' + str(self.n_x_plus) + '_n_phi_' + str(self.n_phi_plus)
        return file_name

    @tr.observe('export_obj_file_btn')
    def export_obj_file(self, event=None):
        #print('export_obj_file - WB Tessellation 4P')
        file_name = self.get_file_name() + '.obj'
        WBGeoUtils.export_obj_file(self, file_name)

    @tr.observe('export_fold_file_btn')
    def export_fold_file_btn_click(self, event=None):
        #print('export_fold_file_btn_click - WB Tessellation 4P')
        file_name = self.get_file_name() + '.fold'
        self.export_fold_file(path=file_name)
