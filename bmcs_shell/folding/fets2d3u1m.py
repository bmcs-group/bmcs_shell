
import traits.api as tr
from ibvpy.fets import FETSEval
from ibvpy.mathkit.tensor import DELTA23_ab
from bmcs_shell.folding.waterbomb_shell import WBShell
import numpy as np
from oricreate.util import \
    get_theta, get_theta_du


INPUT = '+cp_input'


# Kronecker delta
DELTA = np.zeros((3, 3,), dtype='f')
DELTA[(0, 1, 2), (0, 1, 2)] = 1

# Levi Civita symbol
EPS = np.zeros((3, 3, 3), dtype='f')
EPS[(0, 1, 2), (1, 2, 0), (2, 0, 1)] = 1
EPS[(2, 1, 0), (1, 0, 2), (0, 2, 1)] = -1


from ibvpy.sim.tstep_bc import TStepBC

class FETS2D3U1M(FETSEval):
    r'''Triangular, three-node element.
    '''

    vtk_r = tr.Array(np.float_, value=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vtk_cells = [[0, 1, 2]]
    vtk_cell_types = 'Triangle'
    vtk_cell = [0, 1, 2]
    vtk_cell_type = 'Triangle'

    vtk_expand_operator = tr.Array(np.float_, value=DELTA23_ab)

    # =========================================================================
    # Surface integrals using numerical integration
    # =========================================================================
    eta_ip = tr.Array('float_')
    r'''Integration points within a triangle.
    '''

    def _eta_ip_default(self):
        return np.array([[1. / 3., 1. / 3., 1. / 3.]], dtype='f')

    w_m = tr.Array('float_')
    r'''Weight factors for numerical integration.
    '''

    def _w_m_default(self):
        return np.array([1. / 2.], dtype='f')

    n_m = tr.Int(1)
    r'''Number of integration points.
    '''
    @tr.cached_property
    def _get_w_m(self):
        return len(self.w_m)

    n_nodal_dofs = tr.Int(3)

    N_im = tr.Property(depends_on='eta_ip')
    r'''Shape function values in integration points.
    '''
    @tr.cached_property
    def _get_N_im(self):
        eta = self.eta_ip
        return np.array([eta[:, 0], eta[:, 1], 1 - eta[:, 0] - eta[:, 1]],
                        dtype='f')

    dN_imr = tr.Property(depends_on='eta_ip')
    r'''Derivatives of the shape functions in the integration points.
    '''
    @tr.cached_property
    def _get_dN_imr(self):
        dN_mri = np.array([[[1, 0, -1],
                          [0, 1, -1]],
                          ], dtype=np.float_)
        return np.einsum('mri->imr', dN_mri)

    dN_inr = tr.Property(depends_on='eta_ip')
    r'''Derivatives of the shape functions in the integration points.
    '''
    @tr.cached_property
    def _get_dN_inr(self):
        return self.dN_imr

    vtk_expand_operator = tr.Array(value=[1,1,0])
    vtk_node_cell_data = tr.Array
    vtk_ip_cell_data = tr.Array

from ibvpy.mesh.i_fe_uniform_domain import IFEUniformDomain
from ibvpy.xdomain.xdomain_fe_grid import XDomainFE



@tr.provides(IFEUniformDomain)
class FECustomMesh(tr.HasStrictTraits):

    X_Id = tr.Array(np.float_, value=[[0,0, 0], [2,0, 0], [2,2,0], [2,0,0], [1,1,0]])
    I_Fi = tr.Array(np.int_, value=[[0,1,4],
                                    [1,2,4],
                                    [2,3,4],
                                    [3,0,4]])

    fets = tr.Instance(FETS2D3U1M, ())

    n_nodal_dofs = tr.DelegatesTo('fets')
    dof_offset = tr.Int(0)

    n_active_elems = tr.Property
    def _get_n_active_elems(self):
        return len(self.I_Fi)


@tr.provides(IFEUniformDomain)
class FEWBShellMesh(WBShell):

    X_Id = tr.Property
    def _get_X_Id(self):
        return self.X_Ia

    fets = tr.Instance(FETS2D3U1M, ())

    n_nodal_dofs = tr.DelegatesTo('fets')
    dof_offset = tr.Int(0)

    n_active_elems = tr.Property
    def _get_n_active_elems(self):
        return len(self.I_Ei)

class XWBDomain(XDomainFE):
    '''
    Finite element discretization with dofs and mappings derived from the FE definition
    '''
    mesh = tr.Instance(FECustomMesh, ())
    fets = tr.DelegatesTo('mesh')

    n_dofs = tr.Property

    def _get_n_dofs(self):
        return len(self.mesh.X_Id) * self.mesh.n_nodal_dofs

    eta_w = tr.Property
    r'''Weight factors for numerical integration.
    '''

    def _get_eta_w(self):
        return self.fets.w_m

    Na_deta = tr.Property()
    r'''Derivatives of the shape functions in the integration points.
    '''
    @tr.cached_property
    def _get_Na_deta(self):
        return np.einsum('imr->mri', self.fets.dN_imr)

    x_0 = tr.Property()
    r'''Derivatives of the shape functions in the integration points.
    '''
    @tr.cached_property
    def _get_x_0(self):
        return self.mesh.X_Id

    x = tr.Property()
    r'''Derivatives of the shape functions in the integration points.
    '''
    @tr.cached_property
    def _get_x(self):
        return self.x_0

    F = tr.Property()
    r'''Derivatives of the shape functions in the integration points.
    '''
    @tr.cached_property
    def _get_F(self):
        return self.mesh.I_Fi

    T_Fab = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_T_Fab(self):
        return self.F_L_bases[:, 0, :]

    I_Ei = tr.Property
    def _get_I_Ei(self):
        return self.F_N

    x_Eia = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_x_Eia(self):
        X_Eia = self.X_Id[self.I_Ei, :]
        X_E0a = X_Eia[:, 0, :]
        X_Eia -= X_E0a[:, np.newaxis, :]
        X_Eic = np.einsum('Eac,Eic->Eia', self.T_Fab, X_Eia)
        return X_Eic[...,:-1]

    def U2u(self, U_Eia):
        u0_Eia = U_Eia[...,:-1]
        return u0_Eia

    def xU2u(self, U_Eia):
        u1_Eia = np.einsum('Eab,Eib->Eia', self.T_Fab, U_Eia)
        u2_Eie =  np.einsum('ea,Eia->Eie', DELTA23_ab, u1_Eia)
        return u2_Eie

    def f2F(self, f_Eid):
        F0_Eia = np.concatenate( [f_Eid, np.zeros_like(f_Eid[...,:1])], axis=-1)
        return F0_Eia

    def xf2F(self, f_Eid):
        F1_Eia = np.einsum('da,Eid->Eia', DELTA23_ab, f_Eid)
        F2_Eia = np.einsum('Eab,Eia->Eib', self.T_Fab, F1_Eia)
        return F2_Eia

    def k2K(self, K_Eiejf):
        K0_Eicjf = np.concatenate( [K_Eiejf, np.zeros_like(K_Eiejf[:,:,:1,:,:])], axis=2)
        K0_Eicjd = np.concatenate( [K0_Eicjf, np.zeros_like(K0_Eicjf[:,:,:,:,:1])], axis=4)
        return K0_Eicjd

    def xk2K(self, K_Eiejf):
        K1_Eiejf = np.einsum('ea,fb,Eiejf->Eiajb', DELTA23_ab, DELTA23_ab, K_Eiejf) # correct
        T_Eeafb = np.einsum('Eea,Efb->Eeafb', self.T_Fab, self.T_Fab)
        #K_Eab = np.einsum('Eeafb,ef->Eab', T_Eeafb, k_ef)
        K2_Eiajb = np.einsum('Eeafb,Eiejf->Eiajb', T_Eeafb, K1_Eiejf)
        #K2_Eicjd = np.einsum('Eca,Ebd,Eiajb->Eicjd', self.T_Fab, self.T_Fab, K1_Eicjd)
        #K2_Eicjd = np.einsum('Eac,Edb,Eiajb->Eicjd', self.T_Fab, self.T_Fab, K1_Eicjd)
        return K2_Eiajb

    # =========================================================================
    # Property operators for initial configuration
    # =========================================================================
    F0_normals = tr.Property(tr.Array, depends_on='X, L, F')
    r'''Normal facet vectors.
    '''
    @tr.cached_property
    def _get_F0_normals(self):
        x_F = self.x_0[self.F]
        N_deta_ip = self.Na_deta
        r_deta = np.einsum('ajK,IKi->Iaij', N_deta_ip, x_F)
        Fa_normals = np.einsum('Iai,Iaj,ijk->Iak',
                               r_deta[..., 0], r_deta[..., 1], EPS)
        return np.sum(Fa_normals, axis=1)

    sign_normals = tr.Property(tr.Array, depends_on='X,L,F')
    r'''Orientation of the normal in the initial state.
    This array is used to switch the normal vectors of the faces
    to be oriented in the positive sense of the z-axis.
    '''
    @tr.cached_property
    def _get_sign_normals(self):
        return np.sign(self.F0_normals[:, 2])

    F_N = tr.Property(tr.Array, depends_on='X,L,F')
    r'''Counter-clockwise enumeration.
    '''
    @tr.cached_property
    def _get_F_N(self):
        turn_facets = np.where(self.sign_normals < 0)
        F_N = np.copy(self.F)
        F_N[turn_facets, :] = self.F[turn_facets, ::-1]
        return F_N

    F_normals = tr.Property(tr.Array, depends_on=INPUT)
    r'''Get the normals of the facets.
    '''
    @tr.cached_property
    def _get_F_normals(self):
        n = self.Fa_normals
        return np.sum(n, axis=1)

    F_normals_0 = tr.Property(tr.Array, depends_on=INPUT)
    r'''Get the normals of the facets.
    '''
    @tr.cached_property
    def _get_F_normals_0(self):
        n = self.Fa_normals_0
        return np.sum(n, axis=1)

    norm_F_normals = tr.Property(tr.Array, depends_on=INPUT)
    r'''Get the normed normals of the facets.
    '''
    @tr.cached_property
    def _get_norm_F_normals(self):
        n = self.F_normals
        mag_n = np.sqrt(np.einsum('...i,...i', n, n))
        return n / mag_n[:, np.newaxis]

    norm_F_normals_0 = tr.Property(tr.Array, depends_on=INPUT)
    r'''Get the normed normals of the facets.
    '''
    @tr.cached_property
    def _get_norm_F_normals_0(self):
        n = self.F_normals_0
        mag_n = np.sqrt(np.einsum('...i,...i', n, n))
        return n / mag_n[:, np.newaxis]

    F_normals_du = tr.Property(tr.Array, depends_on=INPUT)
    r'''Get the normals of the facets.
    '''
    @tr.cached_property
    def _get_F_normals_du(self):
        n_du = self.Fa_normals_du
        return np.sum(n_du, axis=1)

    F_area = tr.Property(tr.Array, depends_on=INPUT)
    r'''Get the surface area of the facets.
    '''
    @tr.cached_property
    def _get_F_area(self):
        a = self.Fa_area
        A = np.einsum('a,Ia->I', self.eta_w, a)
        return A

    # =========================================================================
    # Potential energy
    # =========================================================================
    F_V = tr.Property(tr.Array, depends_on=INPUT)
    r'''Get the total potential energy of gravity for each facet
    '''
    @tr.cached_property
    def _get_F_V(self):
        eta_w = self.eta_w
        a = self.Fa_area
        ra = self.Fa_r
        F_V = np.einsum('a,Ia,Ia->I', eta_w, ra[..., 2], a)
        return F_V

    F_V_du = tr.Property(tr.Array, depends_on=INPUT)
    r'''Get the derivative of total potential energy of gravity for each facet
    with respect to each node and displacement component [FIi]
    '''
    @tr.cached_property
    def _get_F_V_du(self):
        r = self.Fa_r
        a = self.Fa_area
        a_dx = self.Fa_area_du
        r3_a_dx = np.einsum('Ia,IaJj->IaJj', r[..., 2], a_dx)
        N_eta_ip = self.Na
        r3_dx = np.einsum('aK,KJ,j->aJj', N_eta_ip, DELTA, DELTA[2, :])
        a_r3_dx = np.einsum('Ia,aJj->IaJj', a, r3_dx)
        F_V_du = np.einsum('a,IaJj->IJj', self.eta_w, (a_r3_dx + r3_a_dx))
        return F_V_du

    # =========================================================================
    # Line vectors
    # =========================================================================

    F_L_vectors_0 = tr.Property(tr.Array, depends_on=INPUT)
    r'''Get the cycled line vectors around the facet
    The cycle is closed - the first and last vector are identical.

    .. math::
        v_{pld} \;\mathrm{where} \; p\in\mathcal{F}, l\in (0,1,2), d\in (0,1,2)

    with the indices :math:`p,l,d` representing the facet, line vector around
    the facet and and vector component, respectively.
    '''
    @tr.cached_property
    def _get_F_L_vectors_0(self):
        F_N = self.F_N  # F_N is cycled counter clockwise
        return self.x_0[F_N[:, (1, 2, 0)]] - self.x_0[F_N[:, (0, 1, 2)]]

    F_L_vectors = tr.Property(tr.Array, depends_on=INPUT)
    r'''Get the cycled line vectors around the facet
    The cycle is closed - the first and last vector are identical.

    .. math::
        v_{pld} \;\mathrm{where} \; p\in\mathcal{F}, l\in (0,1,2), d\in (0,1,2)

    with the indices :math:`p,l,d` representing the facet, line vector around
    the facet and and vector component, respectively.
    '''
    @tr.cached_property
    def _get_F_L_vectors(self):
        F_N = self.F_N  # F_N is cycled counter clockwise
        return self.x[F_N[:, (1, 2, 0)]] - self.x[F_N[:, (0, 1, 2)]]

    F_L_vectors_du = tr.Property(tr.Array, depends_on=INPUT)
    r'''Get the derivatives of the line vectors around the facets.

    .. math::
        \pard{v_{pld}}{x_{Ie}} \; \mathrm{where} \;
        p \in \mathcal{F},  \in (0,1,2), d\in (0,1,2), I\in \mathcal{N},
        e \in (0,1,3)

    with the indices :math:`p,l,d,I,e` representing the facet,
    line vector around the facet and and vector component,
    node vector and and its component index,
    respectively.

    This array works essentially as an index function delivering -1
    for the components of the first node in each dimension and +1
    for the components of the second node
    in each dimension.

    For a facet :math:`p` with lines :math:`l` and component :math:`d` return
    the derivatives with respect to the displacement of the node :math:`I`
    in the direction :math:`e`.

    .. math::
        \bm{a}_1 = \bm{x}_2 - \bm{x}_1 \\
        \bm{a}_2 = \bm{x}_3 - \bm{x}_2 \\
        \bm{a}_3 = \bm{x}_1 - \bm{x}_3

    The corresponding derivatives are then

    .. math::
        \pard{\bm{a}_1}{\bm{u}_1} = -1, \;\;\;
        \pard{\bm{a}_1}{\bm{u}_2} = 1 \\
        \pard{\bm{a}_2}{\bm{u}_2} = -1, \;\;\;
        \pard{\bm{a}_2}{\bm{u}_3} = 1 \\
        \pard{\bm{a}_3}{\bm{u}_3} = -1, \;\;\;
        \pard{\bm{a}_3}{\bm{u}_1} = 1 \\

    '''

    def _get_F_L_vectors_du(self):
        return self.L_vectors_du[self.F_L]

    F_L_vectors_dul = tr.Property(tr.Array, depends_on=INPUT)

    def _get_F_L_vectors_dul(self):
        return self.L_vectors_dul[self.F_L]

    norm_F_L_vectors = tr.Property(tr.Array, depends_on=INPUT)
    r'''Get the cycled line vectors around the facet
    The cycle is closed - the first and last vector are identical.
    '''
    @tr.cached_property
    def _get_norm_F_L_vectors(self):
        v = self.F_L_vectors
        mag_v = np.sqrt(np.einsum('...i,...i', v, v))
        return v / mag_v[..., np.newaxis]

    norm_F_L_vectors_du = tr.Property(tr.Array, depends_on=INPUT)
    '''Get the derivatives of cycled line vectors around the facet
    '''
    @tr.cached_property
    def _get_norm_F_L_vectors_du(self):
        v = self.F_L_vectors
        v_du = self.F_L_vectors_du  # @UnusedVariable
        mag_v = np.einsum('...i,...i', v, v)  # @UnusedVariable
        # @todo: finish the chain rule
        raise NotImplemented

    # =========================================================================
    # Orthonormal basis of each facet.
    # =========================================================================
    F_L_bases = tr.Property(tr.Array, depends_on=INPUT)
    r'''Line bases around a facet.
    '''
    @tr.cached_property
    def _get_F_L_bases(self):
        l = self.norm_F_L_vectors
        n = self.norm_F_normals
        lxn = np.einsum('...li,...j,...kij->...lk', l, n, EPS)
        n_ = n[:, np.newaxis, :] * np.ones((1, 3, 1), dtype='float_')
        T = np.concatenate([l[:, :, np.newaxis, :],
                            -lxn[:, :, np.newaxis, :],
                            n_[:, :, np.newaxis, :]], axis=2)
        return T

    F_L_bases_du = tr.Property(tr.Array, depends_on=INPUT)
    r'''Derivatives of the line bases around a facet.
    '''
    @tr.cached_property
    def _get_F_L_bases_du(self):
        '''Derivatives of line bases'''
        raise NotImplemented

    # =========================================================================
    # Sector angles
    # =========================================================================
    F_theta = tr.Property(tr.Array, depends_on=INPUT)
    '''Get the sector angles :math:`\theta`  within a facet.
    '''
    @tr.cached_property
    def _get_F_theta(self):
        v = self.F_L_vectors
        a = -v[:, (2, 0, 1), :]
        b = v[:, (0, 1, 2), :]
        return get_theta(a, b)

    F_theta_du = tr.Property(tr.Array, depends_on=INPUT)
    r'''Get the derivatives of sector angles :math:`\theta` within a facet.
    '''
    @tr.cached_property
    def _get_F_theta_du(self):
        v = self.F_L_vectors
        v_du = self.F_L_vectors_du

        a = -v[:, (2, 0, 1), :]
        b = v[:, (0, 1, 2), :]
        a_du = -v_du[:, (2, 0, 1), ...]
        b_du = v_du[:, (0, 1, 2), ...]

        return get_theta_du(a, a_du, b, b_du)

    Fa_normals_du = tr.Property
    '''Get the derivatives of the normals with respect
    to the node displacements.
    '''

    def _get_Fa_normals_du(self):
        x_F = self.x[self.F_N]
        N_deta_ip = self.Na_deta
        NN_delta_eps_x1 = np.einsum('aK,aL,KJ,dli,ILl->IaiJd',
                                    N_deta_ip[:, 0, :], N_deta_ip[:, 1, :],
                                    DELTA, EPS, x_F)
        NN_delta_eps_x2 = np.einsum('aK,aL,LJ,kdi,IKk->IaiJd',
                                    N_deta_ip[:, 0, :], N_deta_ip[:, 1, :],
                                    DELTA, EPS, x_F)
        n_du = NN_delta_eps_x1 + NN_delta_eps_x2
        return n_du

    Fa_area_du = tr.Property
    '''Get the derivatives of the facet area with respect
    to node displacements.
    '''

    def _get_Fa_area_du(self):
        a = self.Fa_area
        n = self.Fa_normals
        n_du = self.Fa_normals_du
        a_du = np.einsum('Ia,Iak,IakJd->IaJd', 1 / a, n, n_du)
        return a_du

    Fa_normals = tr.Property
    '''Get normals of the facets.
    '''

    def _get_Fa_normals(self):
        x_F = self.x[self.F_N]
        N_deta_ip = self.Na_deta
        r_deta = np.einsum('ajK,IKi->Iaij', N_deta_ip, x_F)
        return np.einsum('Iai,Iaj,ijk->Iak',
                         r_deta[..., 0], r_deta[..., 1], EPS)

    Fa_normals_0 = tr.Property
    '''Get normals of the facets.
    '''

    def _get_Fa_normals_0(self):
        x_F = self.x_0[self.F_N]
        N_deta_ip = self.Na_deta
        r_deta = np.einsum('ajK,IKi->Iaij', N_deta_ip, x_F)
        return np.einsum('Iai,Iaj,ijk->Iak',
                         r_deta[..., 0], r_deta[..., 1], EPS)

    Fa_area = tr.Property
    '''Get the surface area of the facets.
    '''

    def _get_Fa_area(self):
        n = self.Fa_normals
        a = np.sqrt(np.einsum('Iai,Iai->Ia', n, n))
        return a

    Fa_r = tr.Property
    '''Get the reference vector to integrations point in each facet.
    '''

    def _get_Fa_r(self):
        x_F = self.x[self.F_N]
        N_eta_ip = self.Na
        r = np.einsum('aK,IKi->Iai', N_eta_ip, x_F)
        return r

    # =========================================================================
    # Interier as level set
    # =========================================================================
