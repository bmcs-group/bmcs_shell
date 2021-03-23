"""

"""
import bmcs_utils.api as bu
import sympy as sp
from sympy.algebras.quaternion import Quaternion
import k3d
import traits.api as tr
import numpy as np
import math


class WBElemSymb5ParamXL(bu.SymbExpr):

    a, b, c = sp.symbols('a, b, c', positive=True)
    gamma = sp.symbols('gamma')

    U_ur_0 = sp.Matrix([a, b, 0])
    U_ul_0 = sp.Matrix([-a, b, 0])
    V_r_0 = sp.Matrix([c, 0, 0])
    V_l_0 = sp.Matrix([-c, 0, 0])

    x_ur, y_ur, z_ur = sp.symbols(r'x_ur, y_ur, z_ur')
    x_ul, y_ul, z_ul = sp.symbols(r'x_ul, y_ul, z_ul')

    U_ur_1 = sp.Matrix([x_ur, y_ur, z_ur])
    U_ul_1 = sp.Matrix([x_ul, y_ul, z_ul])
    V_r_1 = sp.Matrix([c * sp.sin(gamma), 0, c * sp.cos(gamma)])
    V_l_1 = sp.Matrix([-c * sp.sin(gamma), 0, c * sp.cos(gamma)])
    X_UOV_r_0 = U_ur_0.T * V_r_0
    X_VOU_l_0 = U_ul_0.T * V_l_0
    X_UOV_r_1 = U_ur_1.T * V_r_1
    X_VOU_l_1 = U_ul_1.T * V_l_1
    Eq_UOV_r = sp.Eq(X_UOV_r_0[0], X_UOV_r_1[0])
    Eq_UOV_l = sp.Eq(X_VOU_l_0[0], X_VOU_l_1[0])
    X_VUO_r_0 = (V_r_0 - U_ur_0).T * (-U_ur_0)
    X_VUO_l_0 = (V_l_0 - U_ul_0).T * (-U_ul_0)
    X_VUO_r_1 = (V_r_1 - U_ur_1).T * (-U_ur_1)
    X_VUO_l_1 = (V_l_1 - U_ul_1).T * (-U_ul_1)
    Eq_VUO_r = sp.Eq(-X_VUO_r_0[0], -X_VUO_r_1[0])
    Eq_VUO_l = sp.Eq(-X_VUO_l_0[0], -X_VUO_l_1[0])

    X_UOU_0 = (U_ul_0).T * (U_ur_0)
    X_UOU_1 = (U_ul_1).T * (U_ur_1)
    Eq_UOU = sp.Eq(X_UOU_0[0], X_UOU_1[0])

    yz_ur_sol1, yz_ur_sol2 = sp.solve({Eq_UOV_r, Eq_VUO_r}, [y_ur, z_ur])
    yz_ul_sol1, yz_ul_sol2 = sp.solve({Eq_UOV_l, Eq_VUO_l}, [y_ul, z_ul])

    y_ur_sol, z_ur_sol = yz_ur_sol1
    y_ul_sol, z_ul_sol = yz_ul_sol1

    subs_yz = {y_ur: y_ur_sol, z_ur: z_ur_sol,
               y_ul: y_ul_sol, z_ul: z_ul_sol}

    Eq_UOU_x = Eq_UOU.subs(subs_yz)
    Eq_UOU_x_rearr = sp.Eq(-Eq_UOU_x.args[1].args[1],
                           -Eq_UOU_x.args[0] + Eq_UOU_x.args[1].args[0] + Eq_UOU_x.args[1].args[2])
    Eq_UOU_x_rhs = Eq_UOU_x_rearr.args[1] ** 2 - Eq_UOU_x_rearr.args[0] ** 2
    Eq_UOU_x_rhs_collect = sp.collect(sp.expand(Eq_UOU_x_rhs), x_ul)

    A_ = Eq_UOU_x_rhs_collect.coeff(x_ul, 2)
    B_ = Eq_UOU_x_rhs_collect.coeff(x_ul, 1)
    C_ = Eq_UOU_x_rhs_collect.coeff(x_ul, 0)
    A, B, C = sp.symbols('A, B, C')

    x_ul_sol1, x_ul_sol2 = sp.solve(A * x_ul ** 2 + B * x_ul + C, x_ul)

    x_ul_ = x_ul_sol2
    y_ur_ = y_ur_sol
    y_ul_ = y_ul_sol
    z_ur_ = z_ur_sol
    z_ul_ = z_ul_sol

    symb_model_params = ['gamma', 'x_ur', 'a', 'b', 'c', ]
    symb_expressions = [
        ('x_ul_', ('A', 'B', 'C')),
        ('y_ur_', ('x_ul',)),
        ('y_ul_', ('x_ul',)),
        ('z_ur_', ('x_ul',)),
        ('z_ul_', ('x_ul',)),
        ('A_', ()),
        ('B_', ()),
        ('C_', ()),
        ('V_r_1', ()),
        ('V_l_1', ()),
    ]

class WBElem5Param(bu.InteractiveModel,bu.InjectSymbExpr):
    name = 'waterbomb cell 5p'
    symb_class = WBElemSymb5ParamXL

    plot_backend = 'k3d'

    gamma = bu.Float(np.pi/2+1e-5, GEO=True)
    x_ur = bu.Float(1000, GEO=True)
    a = bu.Float(1000, GEO=True)
    b = bu.Float(1000, GEO=True)
    c = bu.Float(1000, GEO=True)
    a_high = bu.Float(2000)
    b_high = bu.Float(2000)
    c_high = bu.Float(2000)

    show_wireframe = tr.Bool

    ipw_view = bu.View(
        bu.Item('gamma', latex=r'\gamma', editor=bu.FloatRangeEditor(
            low=1e-6, high=np.pi / 2, n_steps=100, continuous_update=True)),
        bu.Item('x_ur', latex=r'x^\urcorner', editor=bu.FloatRangeEditor(
            low=1e-6, high=2000, n_steps=100, continuous_update=True)),
        bu.Item('a', latex='a', editor=bu.FloatRangeEditor(low=1e-6, high_name='a_high', n_steps=100, continuous_update=True)),
        bu.Item('b', latex='b', editor=bu.FloatRangeEditor(low=1e-6, high_name='b_high', n_steps=100, continuous_update=True)),
        bu.Item('c', latex='c', editor=bu.FloatRangeEditor(low=1e-6, high_name='c_high', n_steps=100, continuous_update=True)),
    )

    n_I = tr.Property
    def _get_n_I(self):
        return len(self.X_Ia)

    X_Ia = tr.Property(depends_on='+GEO')
    '''Array with nodal coordinates I - node, a - dimension
    '''
    @tr.cached_property
    def _get_X_Ia(self):
        gamma = self.gamma
        alpha = np.pi/2 - gamma

        x_ur = self.x_ur
        A = self.symb.get_A_()
        B = self.symb.get_B_()
        C = self.symb.get_C_()

        x_ul = self.symb.get_x_ul_(A, B, C)
        y_ur = self.symb.get_y_ur_(x_ul)
        y_ul = self.symb.get_y_ul_(x_ul)
        z_ur = self.symb.get_z_ur_(x_ul)
        z_ul = self.symb.get_z_ul_(x_ul)

        x_ll = x_ul
        x_lr = x_ur
        y_ll = - y_ul
        y_lr = - y_ur
        z_ll = z_ul
        z_lr = z_ur

        V_r_1 = self.symb.get_V_r_1().flatten()
        V_l_1 = self.symb.get_V_l_1().flatten()

        return np.array([
            [0,0,0], # 0 point
            [x_ur, y_ur, z_ur], #U++
            [x_ul, y_ul, z_ul], #U-+  ul
            [x_lr, y_lr, z_lr], #U+-
            [x_ll, y_ll, z_ll], #U--
            V_r_1, # [self.c * np.sin(gamma), 0, self.c * np.cos(gamma)],
            V_l_1, # [-self.c * np.sin(gamma), 0, self.c * np.cos(gamma)],
            ], dtype=np.float_
        )

    I_boundary = tr.Array(np.int_, value=[[2,1],
                                          [6,5],
                                          [4,3],])
    '''Boundary nodes in 2D array to allow for generation of shell boundary nodes'''

    X_theta_Ia = tr.Property(depends_on='+GEO')
    '''Array with nodal coordinates I - node, a - dimension
    '''
    @tr.cached_property
    def _get_X_theta_Ia(self):
        D_a = self.symb.get_D_(self.alpha).T
        theta = self.symb.get_theta_sol(self.alpha)
        XD_Ia = D_a + self.X_Ia
        X_center = XD_Ia[1,:]

        rotation_axes = np.array([[1, 0, 0]], dtype=np.float_)
        rotation_angles = np.array([-theta], dtype=np.float_)
        rotation_centers = np.array([X_center], dtype=np.float_)

        x_single = np.array([XD_Ia], dtype='f')
        x_pulled_back = x_single - rotation_centers[:, np.newaxis, :]
        q = axis_angle_to_q(rotation_axes, rotation_angles)
        x_rotated = qv_mult(q, x_pulled_back)
        x_pushed_forward = x_rotated + rotation_centers[:, np.newaxis, :]
        x_translated = x_pushed_forward #  + self.translations[:, np.newaxis, :]
        return x_translated[0,...]

    I_Fi = tr.Property
    '''Triangle mapping '''
    @tr.cached_property
    def _get_I_Fi(self):
        return np.array([[0,1,2],
                         [0,3,4],
                         [0,1,5],
                         [0,2,6],
                         [0,3,5],
                         [0,4,6],
                         ])

    delta_x = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_delta_x(self):
        return self.symb.get_delta_x()

    delta_phi = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_delta_phi(self):
        return self.symb.get_delta_phi()

    R_0 = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_R_0(self):
        return self.symb.get_R_0()

    def setup_plot(self, pb):
        wb_mesh = k3d.mesh(self.X_Ia.astype(np.float32),
                                 self.I_Fi.astype(np.uint32),
                                 color=0x999999,
                                 side='double')
        pb.plot_fig += wb_mesh
        pb.objects['wb_mesh'] = wb_mesh
        if self.show_wireframe:
            wb_mesh_wireframe = k3d.mesh(self.X_Ia.astype(np.float32),
                                            self.I_Fi.astype(np.uint32),
                                            color=0x000000,
                                            wireframe=True)

            pb.plot_fig += wb_mesh_wireframe
            pb.objects['wb_mesh_wireframe'] = wb_mesh_wireframe

    def update_plot(self, pb):
        wb_mesh = pb.objects['wb_mesh']
        wb_mesh_wireframe = pb.objects['wb_mesh_wireframe']
        self._assign_mesh_data(wb_mesh)
        if self.show_wireframe:
            self._assign_mesh_data(wb_mesh_wireframe)

    def _assign_mesh_data(self, mesh):
        mesh.vertices = self.X_Ia.astype(np.float32)
        mesh.indices = self.I_Fi.astype(np.uint32)
        mesh.attributes = self.X_Ia[:, 2].astype(np.float32)

def q_normalize(q, axis=1):
    sq = np.sqrt(np.sum(q * q, axis=axis))
    sq[np.where(sq == 0)] = 1.e-19
    return q / sq[:, np.newaxis]


def v_normalize(q, axis=1):
    sq = np.einsum('...a,...a->...', q, q)
    sq[np.where(sq == 0)] = 1.e-19
    return q / sq[..., np.newaxis]


def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z], dtype='f')


def q_conjugate(q):
    qn = q_normalize(q.T).T
    w, x, y, z = qn
    return np.array([w, -x, -y, -z], dtype='f')


def qv_mult(q1, u):
    #print('q1', q1.shape, 'u', u.shape)
    zero_re = np.zeros((u.shape[0], u.shape[1]), dtype='f')
    #print('zero_re', zero_re.shape)
    q2 = np.concatenate([zero_re[:, :, np.newaxis], u], axis=2)
    #print('q2', q2.shape)
    q2 = np.rollaxis(q2, 2)
    #print('q2', q2.shape)
    q12 = q_mult(q1[:, :, np.newaxis], q2[:, :, :])
    #print('q12', q12.shape)
    q_con = q_conjugate(q1)
    #print('q_con', q_con.shape)
    q = q_mult(q12, q_con[:, :, np.newaxis])
    #print('q', q.shape)
    q = np.rollaxis(np.rollaxis(q, 2), 2)
    #print('q', q.shape)
    return q[:, :, 1:]


def axis_angle_to_q(v, theta):
    v_ = v_normalize(v, axis=1)
    x, y, z = v_.T
#    print('x,y,z', x, y, z)
    theta = theta / 2
#    print('theta', theta)
    w = np.cos(theta)
    x = x * np.sin(theta)
    y = y * np.sin(theta)
    z = z * np.sin(theta)
#    print('x,y,z', x, y, z)
    return np.array([w, x, y, z], dtype='f')


def q_to_axis_angle(q):
    w, v = q[0, :], q[1:, :]
    theta = np.arccos(w) * 2.0
    return theta, v_normalize(v)

