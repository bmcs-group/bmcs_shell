import traits.api as tr
from bmcs_shell.folding.analysis.wb_fets2d3u1m_fe import FETS2D3U1M
import bmcs_utils.api as bu
import numpy as np
from ibvpy.mesh.i_fe_uniform_domain import IFEUniformDomain
import k3d

@tr.provides(IFEUniformDomain)
class FETriangularMesh(bu.Model):
    name = 'FETriangularMesh'

    X_Id = tr.Array(np.float_, value=[[0, 0, 0], [2, 0, 0], [2, 2, 0], [1, 1, 0]])
    I_Fi = tr.Array(np.int_, value=[[0, 1, 3],
                                    [1, 2, 3],
                                    ])

    fets = tr.Instance(FETS2D3U1M, ())

    n_nodal_dofs = tr.DelegatesTo('fets')
    dof_offset = tr.Int(0)

    n_active_elems = tr.Property
    def _get_n_active_elems(self):
        return len(self.I_Fi)

    #=========================================================================
    # 3d Visualization
    #=========================================================================
    plot_backend = 'k3d'
    show_wireframe = bu.Bool(True)

    def setup_plot(self, pb):
        X_Id = self.X_Id.astype(np.float32)
        I_Fi = self.I_Fi.astype(np.uint32)

        fe_mesh = k3d.mesh(X_Id, I_Fi,
                              color=0x999999, opacity=0.5,
                              side='double')
        pb.plot_fig += fe_mesh
        pb.objects['mesh'] = fe_mesh

        if self.show_wireframe:
            k3d_mesh_wireframe = k3d.mesh(X_Id,
                                          I_Fi,
                                          color=0x000000,
                                          wireframe=True)
            pb.plot_fig += k3d_mesh_wireframe
            pb.objects['mesh_wireframe'] = k3d_mesh_wireframe

    def update_plot(self, pb):
        X_Id = self.X_Id.astype(np.float32)
        I_Fi = self.I_Fi.astype(np.uint32)

        mesh = pb.objects['mesh']
        mesh.vertices = X_Id
        mesh.indices = I_Fi
        if self.show_wireframe:
            wireframe = pb.objects['mesh_wireframe']
            wireframe.vertices = X_Id
            wireframe.indices = I_Fi