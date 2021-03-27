
import traits.api as tr
import bmcs_utils.api as bu
from .fe_triangular_mesh import FETriangularMesh
from .wb_shell_geometry import WBShellGeometry
import pygmsh
import numpy as np
import k3d

class WBShellFETriangularMesh(FETriangularMesh):
    """Directly mapped mesh with one-to-one mapping
    """
    name = 'FE-Mesh'

    plot_backend = 'k3d'

    geo = bu.Instance(WBShellGeometry)

    I_CDij = tr.DelegatesTo('geo')
    unique_node_map = tr.DelegatesTo('geo')
    n_phi_plus = tr.DelegatesTo('geo')

    direct_mesh = bu.Bool(False, DSC=True)

    subdivision = bu.Float(10, DSC=True)

    show_wireframe = bu.Bool(True,GEO=True)

    ipw_view = bu.View(
        bu.Item('subdivision'),
        bu.Item('direct_mesh'),
        bu.Item('export_vtk'),
        bu.Item('show_wireframe'),
    )

    mesh = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_mesh(self):

        X_Id = self.geo.X_Ia
        I_Fi = self.geo.I_Fi
        mesh_size = np.linalg.norm(X_Id[1]-X_Id[0])/self.subdivision

        X_Fid = X_Id[I_Fi]
        with pygmsh.geo.Geometry() as geom:
            for X_id in X_Fid:
                geom.add_polygon(X_id, mesh_size=mesh_size)
            mesh = geom.generate_mesh()
        return mesh

    X_Id = tr.Property
    def _get_X_Id(self):
        if self.direct_mesh:
            return self.geo.X_Ia
        return np.array(self.mesh.points, dtype=np.float_)

    I_Fi = tr.Property
    def _get_I_Fi(self):
        if self.direct_mesh:
            return self.geo.I_Fi
        return self.mesh.cells[1][1]

    bc_fixed_nodes = tr.Array(np.int_, value = [])
    bc_loaded_nodes = tr.Array(np.int_, value = [])

    export_vtk = bu.Button

    @tr.observe('export_vtk')
    def write(self, event=None):
        self.mesh.write("test_shell_mesh.vtk")

    def setup_plot(self, pb):

        X_Id = self.X_Id.astype(np.float32)
        I_Fi = self.I_Fi.astype(np.uint32)

        fixed_nodes = self.bc_fixed_nodes
        loaded_nodes = self.bc_loaded_nodes

        X_Ma = X_Id[fixed_nodes]
        k3d_fixed_nodes = k3d.points(X_Ma, color=0x22ffff, point_size=100)
        pb.plot_fig += k3d_fixed_nodes
        pb.objects['fixed_nodes'] = k3d_fixed_nodes

        X_Ma = X_Id[loaded_nodes]
        k3d_loaded_nodes = k3d.points(X_Ma, color=0xff22ff, point_size=100)
        pb.plot_fig += k3d_loaded_nodes
        pb.objects['loaded_nodes'] = k3d_loaded_nodes

        wb_fe_mesh = k3d.mesh(X_Id, I_Fi,
                              color=0x999999, opacity=0.5,
                              side='double')
        pb.plot_fig += wb_fe_mesh
        pb.objects['wb_mesh'] = wb_fe_mesh

        if self.show_wireframe:
            k3d_mesh_wireframe = k3d.mesh(X_Id,
                                          I_Fi,
                                          color=0x000000,
                                          wireframe=True)
            pb.plot_fig += k3d_mesh_wireframe
            pb.objects['mesh_wireframe'] = k3d_mesh_wireframe


    def update_plot(self, pb):
        fixed_nodes = self.bc_fixed_nodes
        loaded_nodes = self.bc_loaded_nodes
        X_Id = self.X_Id.astype(np.float32)
        I_Fi = self.I_Fi.astype(np.uint32)
        pb.objects['fixed_nodes'].positions = X_Id[fixed_nodes]
        pb.objects['loaded_nodes'].positions = X_Id[loaded_nodes]
        mesh = pb.objects['wb_mesh']
        mesh.vertices = X_Id
        mesh.indices = I_Fi
        if self.show_wireframe:
            wireframe = pb.objects['mesh_wireframe']
            wireframe.vertices = X_Id
            wireframe.indices = I_Fi
