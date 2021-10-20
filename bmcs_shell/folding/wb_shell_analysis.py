import k3d
import numpy as np
from bmcs_shell.folding.vmats2D_elastic import MATS2DElastic
from ibvpy.sim.tstep_bc import TStepBC
import bmcs_utils.api as bu
import traits.api as tr
from ibvpy.bcond import BCDof
from .tri_xdomain_fe import TriXDomainFE
from .wb_shell_geometry import WBShellGeometry
from .wb_fe_triangular_mesh import WBShellFETriangularMesh

itags_str = '+GEO,+MAT,+BC'


class WBShellAnalysis(TStepBC, bu.InteractiveModel):
    name = 'WBShellAnalysis'

    F = bu.Float(-1000, BC=True)
    h = bu.Float(-1000, GEO=True)
    show_wireframe = bu.Bool(True, GEO=True)

    ipw_view = bu.View(
        bu.Item('F', editor=bu.FloatRangeEditor(low=-20000, high=20000, n_steps=100),
                continuous_update=False),
        bu.Item('h',
                editor=bu.FloatRangeEditor(low=1, high=100, n_steps=100),
                continuous_update=False),
        bu.Item('show_wireframe'),
        time_editor=bu.ProgressEditor(run_method='run',
                                      reset_method='reset',
                                      interrupt_var='interrupt',
                                      time_var='t',
                                      time_max='t_max')
    )

    n_phi_plus = tr.Property()

    def _get_n_phi_plus(self):
        return self.xdomain.mesh.n_phi_plus

    geo = bu.Instance(WBShellGeometry, ())

    tmodel = bu.Instance(MATS2DElastic, ())

    tree = ['geo', 'tmodel', 'xdomain']

    xdomain = tr.Property(tr.Instance(TriXDomainFE),
                          depends_on="state_changed")
    '''Discretization object.
    '''

    @tr.cached_property
    def _get_xdomain(self):
        # prepare the mesh generator
        mesh = WBShellFETriangularMesh(geo=self.geo, direct_mesh=True)
        # construct the domain with the kinematic strain mapper and stress integrator
        return TriXDomainFE(
            mesh=mesh,
            integ_factor=self.h,
        )

    domains = tr.Property(depends_on="state_changed")

    @tr.cached_property
    def _get_domains(self):
        return [(self.xdomain, self.tmodel)]

    def reset(self):
        self.sim.reset()

    t = tr.Property()

    def _get_t(self):
        return self.sim.t

    def _set_t(self, value):
        self.sim.t = value

    t_max = tr.Property()

    def _get_t_max(self):
        return self.sim.t_max

    def _set_t_max(self, value):
        self.sim.t_max = value

    interrupt = tr.Property()

    def _get_interrupt(self):
        return self.sim.interrupt

    def _set_interrupt(self, value):
        self.sim.interrupt = value

    bc_loaded = tr.Property(depends_on="state_changed")

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

    bc_fixed = tr.Property(depends_on="state_changed")

    @tr.cached_property
    def _get_bc_fixed(self):
        xdomain, _ = self.domains[0]
        fixed_xyz_nodes = xdomain.bc_J_xyz
        fixed_x_nodes = xdomain.bc_J_x
        fixed_nodes = np.unique(np.hstack([fixed_xyz_nodes, fixed_x_nodes]))
        fixed_xyz_dofs = (fixed_xyz_nodes[:, np.newaxis] * 3 + np.arange(3)[np.newaxis, :]).flatten()
        fixed_x_dofs = (fixed_x_nodes[:, np.newaxis] * 3).flatten()
        fixed_dofs = np.unique(np.hstack([fixed_xyz_dofs, fixed_x_dofs]))
        bc_fixed = [BCDof(var='u', dof=dof, value=0)
                    for dof in fixed_dofs]
        return bc_fixed, fixed_nodes, fixed_dofs

    bc = tr.Property(depends_on="state_changed")

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

    def setup_plot(self, pb):
        print('analysis: setup_plot')
        X_Id = self.xdomain.mesh.X_Id
        if len(self.hist.U_t) == 0:
            U_1 = np.zeros_like(X_Id)
        else:
            U_1 = self.hist.U_t[-1]
        X1_Id = X_Id + (U_1.reshape(-1, 3) * 1)
        X1_Id = X1_Id.astype(np.float32)
        I_Ei = self.xdomain.I_Ei.astype(np.uint32)

        _, fixed_nodes, _ = self.bc_fixed
        _, loaded_nodes, _ = self.bc_loaded

        X_Ma = X1_Id[fixed_nodes]

        k3d_fixed_nodes = k3d.points(X_Ma, color=0x22ffff, point_size=100)
        pb.plot_fig += k3d_fixed_nodes
        pb.objects['fixed_nodes'] = k3d_fixed_nodes

        X_Ma = X1_Id[loaded_nodes]

        k3d_loaded_nodes = k3d.points(X_Ma, color=0xff22ff, point_size=100)
        pb.plot_fig += k3d_loaded_nodes
        pb.objects['loaded_nodes'] = k3d_loaded_nodes

        wb_mesh_0 = k3d.mesh(self.xdomain.X_Id.astype(np.float32),
                             I_Ei,
                             color=0x999999, opacity=0.5,
                             side='double')
        pb.plot_fig += wb_mesh_0
        pb.objects['wb_mesh_0'] = wb_mesh_0

        wb_mesh_1 = k3d.mesh(X1_Id,
                             I_Ei,
                             color_map=k3d.colormaps.basic_color_maps.Jet,
                             attribute=U_1.reshape(-1, 3)[:, 2],
                             color_range=[np.min(U_1), np.max(U_1)],
                             side='double')
        pb.plot_fig += wb_mesh_1
        pb.objects['wb_mesh_1'] = wb_mesh_1

        if self.show_wireframe:
            k3d_mesh_wireframe = k3d.mesh(X1_Id,
                                          I_Ei,
                                          color=0x000000,
                                          wireframe=True)
            pb.plot_fig += k3d_mesh_wireframe
            pb.objects['mesh_wireframe'] = k3d_mesh_wireframe

    def update_plot(self, pb):

        X_Id = self.xdomain.mesh.X_Id
        print('analysis: update_plot', len(X_Id))
        if len(self.hist.U_t) == 0:
            U_1 = np.zeros_like(X_Id)
            print('analysis: U_I', )
        else:
            U_1 = self.hist.U_t[-1]
        X1_Id = X_Id + (U_1.reshape(-1, 3) * 1)
        X1_Id = X1_Id.astype(np.float32)

        I_Ei = self.xdomain.I_Ei.astype(np.uint32)

        _, fixed_nodes, _ = self.bc_fixed
        _, loaded_nodes, _ = self.bc_loaded

        pb.objects['fixed_nodes'].positions = X1_Id[fixed_nodes]
        pb.objects['loaded_nodes'].positions = X1_Id[loaded_nodes]

        mesh = pb.objects['wb_mesh_1']
        mesh.vertices = X1_Id
        mesh.indices = I_Ei
        mesh.attributes = U_1.reshape(-1, 3)[:, 2]
        mesh.color_range = [np.min(U_1), np.max(U_1)]
        if self.show_wireframe:
            wireframe = pb.objects['mesh_wireframe']
            wireframe.vertices = X1_Id
            wireframe.indices = I_Ei

    def get_Pw(self):
        import numpy as np
        F_to = self.hist.F_t
        U_to = self.hist.U_t
        _, _, loaded_dofs = self.bc_loaded
        F_loaded = np.sum(F_to[:, loaded_dofs], axis=-1)
        U_loaded = np.average(U_to[:, loaded_dofs], axis=-1)
        return U_loaded, F_loaded
