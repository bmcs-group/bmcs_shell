import bmcs_utils.api as bu
import numpy as np
import traits.api as tr
import k3d

class WBCell(bu.Model):
    name = 'Waterbomb cell'

    plot_backend = 'k3d'

    K3D_NODES_LABELS = 'nodes_labels'
    K3D_WIREFRAME = 'wireframe'
    K3D_CELL_MESH = 'cell_mesh'

    wireframe_width = bu.Float(15)

    show_base_cell_ui = bu.Bool(True)
    show_node_labels = bu.Bool(False, GEO=True)
    show_wireframe = bu.Bool(True, GEO=True)
    opacity = bu.Float(0.6, GEO=True)

    ipw_view = bu.View(
        bu.Item('show_node_labels'),
        bu.Item('show_wireframe'),
        bu.Item('wireframe_width'),
    ) if show_base_cell_ui else bu.View()

    X_Ia = tr.Property(depends_on='+GEO')
    '''Array with nodal coordinates I - node, a - dimension
    '''
    @tr.cached_property
    def _get_X_Ia(self):
        return np.array([[0., 0., 0.],
                         [1000., 930.99634691, 365.02849483],
                         [-1000., 930.99634691, 365.02849483],
                         [1000., -930.99634691, 365.02849483],
                         [-1000., -930.99634691, 365.02849483],
                         [764.84218728, 0., 644.21768724],
                         [-764.84218728, 0., 644.21768724]])

    I_Fi = tr.Property
    '''Triangle mapping '''
    @tr.cached_property
    def _get_I_Fi(self):
        return np.array([[0, 1, 2], [0, 3, 4], [0, 1, 5], [0, 5, 3], [0, 2, 6], [0, 6, 4]]).astype(np.int32)

    def setup_plot(self, pb):
        X_Ia = self.X_Ia.astype(np.float32)
        I_Fi = self.I_Fi.astype(np.uint32)
        cell_mesh = k3d.mesh(X_Ia, I_Fi,
                                opacity=self.opacity,
                                color=0x999999,
                                side='double')
        pb.plot_fig += cell_mesh
        pb.objects[self.K3D_CELL_MESH] = cell_mesh

        if self.show_wireframe:
            self._add_wireframe_to_fig(pb, X_Ia, I_Fi)
        if self.show_node_labels:
            self._add_nodes_labels_to_fig(pb, self.X_Ia)

    def update_plot(self, pb):
        # If cell interface was embedded in higher class, this method will be called when user changes parameters
        #  However, cell mesh object will not be there because setup_plot was not called
        if self.K3D_CELL_MESH in pb.objects:
            X_Ia = self.X_Ia.astype(np.float32)
            I_Fi = self.I_Fi.astype(np.uint32)
            cell_mesh = pb.objects[self.K3D_CELL_MESH]
            cell_mesh.vertices = X_Ia
            cell_mesh.indices = I_Fi
            cell_mesh.attributes = X_Ia[:, 2]

            if self.show_wireframe:
                if self.K3D_WIREFRAME in pb.objects:
                    wireframe = pb.objects[self.K3D_WIREFRAME]
                    wireframe.width = self.wireframe_width
                    wireframe.vertices = X_Ia
                    wireframe.indices = I_Fi
                else:
                    self._add_wireframe_to_fig(pb, X_Ia, I_Fi)
            else:
                if self.K3D_WIREFRAME in pb.objects:
                        pb.clear_object(self.K3D_WIREFRAME)

            if self.show_node_labels:
                if self.K3D_NODES_LABELS in pb.objects:
                    pb.clear_object(self.K3D_NODES_LABELS)
                self._add_nodes_labels_to_fig(pb, self.X_Ia)
            else:
                if self.K3D_NODES_LABELS in pb.objects:
                    pb.clear_object(self.K3D_NODES_LABELS)

    def _add_wireframe_to_fig(self, pb, X_Ia, I_Fi):
        k3d_mesh_wireframe = k3d.lines(X_Ia,
                                      I_Fi,
                                      shader='mesh',
                                      width=self.wireframe_width,
                                      color=0x000000)
        pb.plot_fig += k3d_mesh_wireframe
        pb.objects[self.K3D_WIREFRAME] = k3d_mesh_wireframe

    def _add_nodes_labels_to_fig(self, pb, X_Ia):
        text_list = []
        for I, X_a in enumerate(X_Ia):
            k3d_text = k3d.text('%g' % I, tuple(X_a), label_box=False, size=0.8, color=0x00FF00)
            pb.plot_fig += k3d_text
            text_list.append(k3d_text)
        pb.objects[self.K3D_NODES_LABELS] = text_list
