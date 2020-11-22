import k3d
from bmcs_shell.folding.wbfe_xdomain import \
    FEWBShellMesh,FETriangularMesh, XWBDomain
import numpy as np
from bmcs_shell.folding.vmats2D_elastic import MATS2DElastic
from ibvpy.sim.tstep_bc import TStepBC
from ibvpy.tmodel.viz3d_scalar_field import \
    Vis3DStateField, Viz3DScalarField
from bmcs_shell.folding.wb_cell import WBElem
import bmcs_utils.api as bu
from ibvpy.tmodel.viz3d_tensor_field import \
    Vis3DTensorField, Viz3DTensorField
import traits.api as tr
from ibvpy.bcond import BCDof
from bmcs_shell.folding.wbfe_xdomain import XWBDomain
itags_str = '+GEO,+MAT,+BC'

import matplotlib.cm
class WBModel(TStepBC,bu.InteractiveModel):

    name = 'Deflection'

    F = bu.Float(-1000, BC=True)
    h = bu.Float(-1000, GEO=True)

    ipw_view = bu.View(
        bu.Item('F',editor=bu.FloatRangeEditor(low=-20000,high=20000,n_steps=100),
                continuous_update=False),
        bu.Item('h',
                editor=bu.FloatRangeEditor(low=1, high=100, n_steps=100),
                continuous_update=False),
    )

    n_phi_plus = tr.Property()
    def _get_n_phi_plus(self):
        return self.xdomain.mesh.n_phi_plus

    tmodel = tr.Instance(MATS2DElastic,())

    wb_mesh = tr.Instance(FEWBShellMesh,())

    xdomain = tr.Property(tr.Instance(XWBDomain),
                         depends_on=itags_str)
    '''Discretization object.
    '''
    @tr.cached_property
    def _get_xdomain(self):
        return XWBDomain(
            mesh=self.wb_mesh,
            integ_factor=self.h
        )

    domains = tr.Property(depends_on=itags_str)

    @tr.cached_property
    def _get_domains(self):
        return [(self.xdomain, self.tmodel)]

    bc_loaded = tr.Property(depends_on=itags_str)

    @tr.cached_property
    def _get_bc_loaded(self):
        xdomain, _ = self.domains[0]
        ix2 = int((self.n_phi_plus) / 2)
        F_I = xdomain.mesh.I_CDij[ix2, :, 0, :].flatten()
        _, idx_remap = xdomain.mesh.unique_node_map
        loaded_nodes = idx_remap[F_I]  # loaded_nodes = xdomain.bc_J_F
        loaded_dofs = (loaded_nodes[:, np.newaxis] * 3 + 2).flatten()
        bc_loaded = [BCDof(var='f', dof=dof, value=self.F)
                     for dof in loaded_dofs]
        return bc_loaded, loaded_nodes, loaded_dofs

    bc_fixed = tr.Property(depends_on=itags_str)

    @tr.cached_property
    def _get_bc_fixed(self):
        xdomain, _ = self.domains[0]
        fixed_xyz_nodes = xdomain.bc_J_xyz
        fixed_x_nodes = xdomain.bc_J_x
        fixed_nodes = np.unique(np.hstack([fixed_xyz_nodes, fixed_x_nodes]))
        fixed_xyz_dofs = (fixed_xyz_nodes[:, np.newaxis] * 3 + np.arange(3)[np.newaxis, :]).flatten()
        fixed_x_dofs = (fixed_x_nodes[:, np.newaxis] * 3).flatten()
        fixed_dofs = np.unique(np.hstack([fixed_xyz_dofs, fixed_x_dofs]))
        bc_fixed = [BCDof(var= 'u', dof=dof, value=0 )
                   for dof in fixed_dofs]
        return bc_fixed, fixed_nodes, fixed_dofs

    bc = tr.Property(depends_on=itags_str)
    @tr.cached_property
    def _get_bc(self):
        bc_fixed, _, _ = self.bc_fixed
        bc_loaded, _, _ = self.bc_loaded
        return bc_fixed + bc_loaded

    plot_backend = 'k3d'

    k3d_fixed_nodes = tr.Any
    k3d_loaded_nodes = tr.Any
    wb_mesh_0 = tr.Any
    wb_mesh_1 = tr.Any

    def run(self):
        s = self.sim
        s.tloop.k_max = 10
        s.tline.step = 1
        s.tloop.verbose = False
        s.run()

    def get_max_vals(self):
        self.run()
        U_1 = self.hist.U_t[-1]
        U_max = np.max(np.fabs(U_1))
        return U_max

    def plot_k3d(self, k3d_plot):

        self.run()
        U_1 = self.hist.U_t[-1]
        X1_Id = self.xdomain.mesh.X_Id + (U_1.reshape(-1, 3) * 1)
        X1_Id = X1_Id.astype(np.float32)
        I_Ei = self.xdomain.I_Ei.astype(np.uint32)

        _, fixed_nodes, _ = self.bc_fixed
        _, loaded_nodes, _ = self.bc_loaded

        X_Ma = X1_Id[fixed_nodes]
        self.k3d_fixed_nodes = k3d.points(X_Ma, color=0x22ffff, point_size=100)
        k3d_plot += self.k3d_fixed_nodes
        X_Ma = X1_Id[loaded_nodes]
        self.k3d_loaded_nodes = k3d.points(X_Ma, color=0xff22ff, point_size=100)
        k3d_plot += self.k3d_loaded_nodes

        self.wb_mesh_0 = k3d.mesh(self.xdomain.X_Id.astype(np.float32),
                                         I_Ei,
                                         color=0x999999, opacity=0.5,
                                         side='double')
        k3d_plot += self.wb_mesh_0
        self.wb_mesh_1 = k3d.mesh(X1_Id,
                                          I_Ei,
                                          color_map=k3d.colormaps.basic_color_maps.Jet,
                                          attribute=U_1.reshape(-1, 3)[:, 2],
                                          color_range=[np.min(U_1), np.max(U_1)],
                                          side='double')
        k3d_plot += self.wb_mesh_1

    def update_plot(self, k3d_plot):
        X1_Id = self.xdomain.mesh.X_Id
        s = self.sim
        s.reset()
        s.tloop.k_max = 10
        s.tline.step = 1
        s.tloop.verbose = False
        s.run()

        U_1 = self.hist.U_t[-1]
        X1_Id = self.xdomain.mesh.X_Id + (U_1.reshape(-1, 3) * 1)
        X1_Id = X1_Id.astype(np.float32)
        I_Ei = self.xdomain.I_Ei.astype(np.uint32)

        _, fixed_nodes, _ = self.bc_fixed
        _, loaded_nodes, _ = self.bc_loaded

        self.k3d_fixed_nodes.positions = X1_Id[fixed_nodes]
        self.k3d_loaded_nodes.positions = X1_Id[loaded_nodes]

        mesh = self.wb_mesh_1
        mesh.vertices = X1_Id
        mesh.indices = I_Ei
        mesh.attributes = U_1.reshape(-1, 3)[:, 2]
        mesh.color_range=[np.min(U_1), np.max(U_1)]

    def get_Pw(self):
        import numpy as np
        F_to = self.hist.F_t
        U_to = self.hist.U_t
        _, _, loaded_dofs = self.bc_loaded
        F_loaded = np.sum(F_to[:, loaded_dofs], axis=-1)
        U_loaded = np.average(U_to[:, loaded_dofs], axis=-1)
        return U_loaded, F_loaded