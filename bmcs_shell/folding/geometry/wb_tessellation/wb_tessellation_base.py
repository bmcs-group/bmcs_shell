"""Indices: I: node index, a: coordinate, F: facet index, i: node indices of the facet"""

import random

import bmcs_utils.api as bu
import k3d
import numpy as np
import traits.api as tr

from bmcs_shell.folding.geometry.math_utils import get_rot_matrix_around_vector, get_best_rot_and_trans_3d
from bmcs_shell.folding.geometry.wb_cell.wb_cell_4p import WBCell4Param
from bmcs_shell.folding.geometry.wb_cell.wb_cell_5p import WBCell5Param
from bmcs_shell.folding.geometry.wb_cell.wb_cell_5p_v2 import \
    WBCell5ParamV2
from bmcs_shell.folding.geometry.wb_cell.wb_cell_5p_v3 import WBCell5ParamV3


class WBTessellationBase(bu.Model):
    name = 'WB Tessellation Base'

    plot_backend = 'k3d'

    # show_wireframe = bu.Bool(True, GEO=True)
    show_node_labels = bu.Bool(False, GEO=True)
    wb_cell = bu.EitherType(options=[('WBCell4Param', WBCell4Param),
                                     ('WBCell5Param', WBCell5Param),
                                     ('WBCell5ParamV2', WBCell5ParamV2),
                                     ('WBCell5ParamV3', WBCell5ParamV3)
                                     ], GEO=True)
    X_Ia = tr.DelegatesTo('wb_cell_')
    I_Fi = tr.DelegatesTo('wb_cell_')
    tree = ['wb_cell']

    event_geo = bu.Bool(True, GEO=True)

    # Note: Update traits to 6.3.2 in order for the following command to work!!
    @tr.observe('wb_cell_.+GEO', post_init=True)
    def update_after_wb_cell_GEO_changes(self, event):
        self.event_geo = not self.event_geo
        self.update_plot(self.pb)

    ipw_view = bu.View(
        bu.Item('wb_cell'),
        # bu.Item('show_wireframe'),
        bu.Item('show_node_labels'),
    )

    def _get_br_X_Ia(self, X_Ia, rot=None):
        br_X_Ia = self._get_cell_matching_v1_to_v2(X_Ia, np.array([4, 6]), np.array([5, 1]))
        return self.rotate_cell(br_X_Ia, np.array([4, 6]), self.sol[0] if rot is None else rot)

    def _get_ur_X_Ia(self, X_Ia, rot=None):
        ur_X_Ia = self._get_cell_matching_v1_to_v2(X_Ia, np.array([6, 2]), np.array([3, 5]))
        return self.rotate_cell(ur_X_Ia, np.array([6, 2]), self.sol[1] if rot is None else rot)

    def _get_ul_X_Ia(self, X_Ia, rot=None):
        br_X_Ia = self._get_cell_matching_v1_to_v2(X_Ia, np.array([5, 1]), np.array([4, 6]))
        return self.rotate_cell(br_X_Ia, np.array([5, 1]), -self.sol[0] if rot is None else rot)

    def _get_bl_X_Ia(self, X_Ia, rot=None):
        br_X_Ia = self._get_cell_matching_v1_to_v2(X_Ia, np.array([3, 5]), np.array([6, 2]))
        return self.rotate_cell(br_X_Ia, np.array([3, 5]), -self.sol[1] if rot is None else rot)

    def _get_cell_matching_v1_to_v2(self, X_Ia, v1_ids, v2_ids):
        v1_2a = np.array([X_Ia[v1_ids[0]], X_Ia[v1_ids[1]], X_Ia[0]]).T
        v2_2a = np.array([X_Ia[v2_ids[0]], X_Ia[v2_ids[1]], X_Ia[v2_ids[0]] + X_Ia[v2_ids[1]] - X_Ia[0]]).T
        rot, trans = get_best_rot_and_trans_3d(v1_2a, v2_2a)

        translated_X_Ia = trans.flatten() + np.einsum('ba, Ia -> Ib', rot, X_Ia)

        return self.rotate_cell(translated_X_Ia, v1_ids, angle=np.pi)

    def rotate_cell(self, cell_X_Ia, v1_ids, angle=np.pi):
        # Rotating around vector #######
        # 1. Bringing back to origin (because rotating is around a vector originating from origin)
        cell_X_Ia_copy = np.copy(cell_X_Ia)
        cell_X_Ia = cell_X_Ia_copy - cell_X_Ia_copy[v1_ids[1]]

        # 2. Rotating
        rot_around_v1 = get_rot_matrix_around_vector(cell_X_Ia[v1_ids[0]] - cell_X_Ia[v1_ids[1]], angle)
        cell_X_Ia = np.einsum('ba, Ia -> Ib', rot_around_v1, cell_X_Ia)

        # 3. Bringing back in position
        return cell_X_Ia + cell_X_Ia_copy[v1_ids[1]]

    sol = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_sol(self):
        # No solution is provided in base class, a default value is provided for visualization
        return np.array([np.pi, np.pi])

    # Plotting ##########################################################################

    def setup_plot(self, pb):
        self.pb = pb
        pb.clear_fig()
        I_Fi = self.I_Fi
        X_Ia = self.X_Ia
        br_X_Ia = self._get_br_X_Ia(X_Ia)
        ur_X_Ia = self._get_ur_X_Ia(X_Ia)

        self.add_cell_to_pb(pb, X_Ia, I_Fi, 'X_Ia')
        self.add_cell_to_pb(pb, br_X_Ia, I_Fi, 'br_X_Ia')
        self.add_cell_to_pb(pb, ur_X_Ia, I_Fi, 'ur_X_Ia')

    k3d_mesh = {}
    k3d_wireframe = {}
    k3d_labels = {}

    def update_plot(self, pb):
        if self.k3d_mesh:
            X_Ia = self.X_Ia.astype(np.float32)
            br_X_Ia = self._get_br_X_Ia(self.X_Ia).astype(np.float32)
            ur_X_Ia = self._get_ur_X_Ia(self.X_Ia).astype(np.float32)
            self.k3d_mesh['X_Ia'].vertices = X_Ia
            self.k3d_mesh['br_X_Ia'].vertices = br_X_Ia
            self.k3d_mesh['ur_X_Ia'].vertices = ur_X_Ia
            self.k3d_wireframe['X_Ia'].vertices = X_Ia
            self.k3d_wireframe['br_X_Ia'].vertices = br_X_Ia
            self.k3d_wireframe['ur_X_Ia'].vertices = ur_X_Ia
        else:
            self.setup_plot(pb)

    def add_cell_to_pb(self, pb, X_Ia, I_Fi, obj_name):
        plot = pb.plot_fig

        wb_mesh = k3d.mesh(X_Ia.astype(np.float32),
                           I_Fi.astype(np.uint32),
                           # opacity=0.9,
                           color=0x999999,
                           side='double')
        rand_color = random.randint(0, 0xFFFFFF)
        plot += wb_mesh

        self.k3d_mesh[obj_name] = wb_mesh

        # wb_points = k3d.points(X_Ia.astype(np.float32),
        #                          color=0x999999,
        #                        point_size=100)
        # plot +=wb_points

        if self.show_node_labels:
            texts = []
            for I, X_a in enumerate(X_Ia):
                k3d_text = k3d.text('%g' % I, tuple(X_a), label_box=False, size=0.8, color=rand_color)
                plot += k3d_text
                texts.append(k3d_text)
            self.k3d_labels[obj_name] = texts

        wb_mesh_wireframe = k3d.mesh(X_Ia.astype(np.float32),
                                     I_Fi.astype(np.uint32),
                                     color=0x000000,
                                     wireframe=True)
        plot += wb_mesh_wireframe
        self.k3d_wireframe[obj_name] = wb_mesh_wireframe
