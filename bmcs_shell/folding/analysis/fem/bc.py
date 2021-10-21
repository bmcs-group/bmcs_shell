import bmcs_utils.api as bu
import k3d
import numpy as np
import traits.api as tr
from bmcs_shell.folding.geometry.wb_shell_geometry import WBShellGeometry
from ibvpy.bcond import BCDof


class BoundaryConditions(bu.Model):
    name = 'BoundaryConditions'
    plot_backend = 'k3d'

    geo = bu.Instance(WBShellGeometry, ())

    bc_values = bu.Str(r'0, 0, 0')
    f_dir_value = bu.Str(r',, -100')
    add_bc_btn = bu.Button()
    add_bc_btn_editor = bu.ButtonEditor(icon='plus')
    add_force_btn = bu.Button()
    add_force_btn_editor = bu.ButtonEditor(icon='plus')
    save_geo_and_bc = bu.Button()
    active_button = tr.Str

    @tr.observe('add_bc_btn')
    def add_bc_click(self, event=None):
        self._switch_btn('add_bc_btn', self.add_bc_btn_editor, self.add_force_btn_editor)

    @tr.observe('add_force_btn')
    def add_force_click(self, event=None):
        self._switch_btn('add_force_btn', self.add_force_btn_editor, self.add_bc_btn_editor)

    def _switch_btn(self, button_str, btn_editor, other_editor):
        if self.active_button == button_str:
            self.active_button = ''
            self.pb.plot_fig.mode = 'view'
            self.pb.plot_fig.camera_auto_fit = True
            btn_editor.widget.button_style = ''
        else:
            self.active_button = button_str
            self.pb.plot_fig.mode = 'callback'
            self.pb.plot_fig.camera_auto_fit = False
            btn_editor.widget.button_style = 'success'
            other_editor.widget.button_style = ''

    def _closest_node(self, node, nodes):
        nodes = np.asarray(nodes)
        dist_2 = np.sum((nodes - node) ** 2, axis=1)
        return np.argmin(dist_2)

    ipw_view = bu.View(
        bu.Item('bc_values', latex=r'\mathrm{BC~values}'),
        bu.Item('add_bc_btn', editor=add_bc_btn_editor),
        bu.Item('f_dir_value', latex=r'f\mathrm{~dir, value}'),
        bu.Item('add_force_btn', editor=add_force_btn_editor),
        bu.Item('save_geo_and_bc', editor=bu.ButtonEditor(icon='download')),
    )


    bc_loaded = tr.Property(depends_on="state_changed")

    @tr.cached_property
    def _get_bc_loaded(self):
        xdomain = self.xdomain
        ix2 = int((self.n_phi_plus) / 2)
        F_I = xdomain.mesh.I_CDij[ix2, :, 0, :].flatten()
        _, idx_remap = xdomain.mesh.unique_node_map
        loaded_nodes = idx_remap[F_I]  # loaded_nodes = xdomain.bc_J_F
        loaded_dofs = (loaded_nodes[:, np.newaxis] * 3 + 2).flatten()
        bc_loaded = [BCDof(var='f', dof=dof, value=self.F)
                     for dof in loaded_dofs]
        return bc_loaded, loaded_nodes, loaded_dofs

    bc_fixed_nodes = bu.Array() # [[node_idx, bc_x, bc_y, bc_z]]
    bc_fixed = tr.Property(depends_on="state_changed")

    @tr.cached_property
    def _get_bc_fixed(self):
        xdomain = self.xdomain
        # Node indicies for nodes that are fixed in x, y, z directions
        fixed_xyz_nodes = xdomain.bc_J_xyz
        # Node indicies for nodes that are fixed in x direction
        fixed_x_nodes = xdomain.bc_J_x
        fixed_nodes = np.unique(np.hstack([fixed_xyz_nodes, fixed_x_nodes]))
        fixed_xyz_dofs = (fixed_xyz_nodes[:, np.newaxis] * 3 + np.arange(3)[np.newaxis, :]).flatten()
        fixed_x_dofs = (fixed_x_nodes[:, np.newaxis] * 3).flatten()
        fixed_dofs = np.unique(np.hstack([fixed_xyz_dofs, fixed_x_dofs]))
        bc_fixed = [BCDof(var='u', dof=dof, value=0)
                    for dof in fixed_dofs]
        return bc_fixed, fixed_nodes, fixed_dofs


    bc_node_k3d_object_map = {}
    pb = tr.Any

    def setup_plot(self, pb):
        self.pb = pb
        self.geo.setup_plot(pb)

        # Each time one clicks on the tap in the tree, setup_plot() gets called
        for node_idx, k3d_point in self.bc_node_k3d_object_map.items():
            self.pb.plot_fig += k3d_point

        def mesh_click(params):
            """ (example) params = {'msg_type': 'hover_callback',
                                    'position': [x, y, z],
                                    'normal': [0, 0, 1],
                                    'distance': 20,
                                    'face_index': 0,
                                    'face': [0, 1, 2]} """
            vertices = self.geo.X_Ia
            touch_pos = np.array(params['position'])
            idx = self._closest_node(touch_pos, vertices)
            point = vertices[idx]

            dim = 3
            if self.active_button == 'add_bc_btn':
                if idx in self.bc_node_k3d_object_map:
                    self.pb.plot_fig -= self.bc_node_k3d_object_map[idx]
                    self.bc_node_k3d_object_map.pop(idx)
                    self._remove_bc(idx)
                else:
                    k3d_point = k3d.points(point, point_size=100, color=0xFF0000)
                    self.pb.plot_fig += k3d_point
                    self.bc_node_k3d_object_map[idx] = k3d_point
                    self._add_bc(idx)

            elif self.active_button == 'add_force_btn':
                pass

        pb.objects['k3d_mesh'].click_callback = mesh_click
        # wb_mesh_1.hover_callback = foo

    def _parse_text_to_xyz(self, text):
        try:
            x, y, z = np.nan, np.nan, np.nan

            text = text.replace(' ', '')
            commas_idxs = [i for i, char_ in enumerate(text) if char_ == ',']

            x_str = text[0:commas_idxs[0]]
            if x_str:
                x = float(x_str)
            y_str = text[commas_idxs[0] + 1:commas_idxs[1]]
            if y_str:
                y = float(y_str)
            z_str = text[commas_idxs[1] + 1:]
            if z_str:
                z = float(z_str)
            return x, y, z
        except:
            raise ValueError('An invalid value has been provided in bc text box!')

    def _remove_bc(self, idx):
        # Remove the row corresponding to idx
        self.bc_fixed_nodes = self.bc_fixed_nodes[self.bc_fixed_nodes[:, 0] != idx]

    def _add_bc(self, idx):
        x, y, z = self._parse_text_to_xyz(self.bc_values)
        if self.bc_fixed_nodes.size == 0:
            self.bc_fixed_nodes = np.array([[idx, x, y, z]])
        else:
            self.bc_fixed_nodes = np.append(self.bc_fixed_nodes, [[idx, x, y, z]], axis=0)


    def update_plot(self, pb):
        self.geo.update_plot(pb)

