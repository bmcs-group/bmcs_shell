"""

"""
import bmcs_utils.api as bu
import sympy as sp
from sympy.algebras.quaternion import Quaternion
import k3d
import traits.api as tr
import numpy as np
import math


class WBElemSymb5ParamVersion4(bu.SymbExpr):

    a, b, c = sp.symbols('a, b, c', positive=True)
    alpha = sp.symbols('alpha')
    beta = sp.symbols('beta')
    x_ur, x_ul, x_ll, x_lr = sp.symbols('x_ur, x_ul, x_ll, x_lr')
    y_l, y_r = sp.symbols('y_r, y_r')

    # y_r * y_l > 0
    x_ul1_ = (
        - sp.cos(beta) * a +
        (
            (sp.sin(beta) * sp.sin(alpha+beta))
            /
            (sp.cos(alpha+beta) + 1)
        ) * a
    )

    x_ul2_ = (
        - sp.cos(beta) * a +
        (
            (sp.sin(beta) * sp.sqrt(
                sp.sin(alpha+beta)**2 * a**2 - 2*(sp.cos(alpha+beta) - 1) * (b**2-a**2)
            ))
            /
            (sp.cos(alpha+beta) - 1)
        )
    )

    y_r_ = (sp.sqrt(sp.sin(alpha)**2 * b**2
                    - (a * sp.cos(alpha) - x_ur )**2 )
            /
            (sp.sin(alpha))
            )

    y_l_ = (sp.sqrt(sp.sin(beta)**2 * b**2
                    - (a * sp.cos(beta) + x_ur )**2 )
            /
            (sp.sin(beta))
            )

    x_ur_ = (
        ( sp.sin(alpha) * x_ul + sp.sin(alpha + beta) * a )
        /
        ( sp.sin(beta) )
    )

    x_lr_ = 2*a * sp.cos(alpha) - x_ur
    x_ll_ = -2*a * sp.cos(beta) - x_ul

    z_ul_ = (a + sp.cos(beta)*x_ul)/(sp.sin(beta))
    z_ur_ = (a - sp.cos(alpha)*x_ur)/(sp.sin(alpha))
    z_ll_ = (a + sp.cos(beta)*x_ll)/(sp.sin(beta))
    z_lr_ = (a - sp.cos(alpha)*x_lr)/(sp.sin(alpha))

    symb_model_params = ['alpha', 'beta', 'a', 'b', 'c', ]
    symb_expressions = [
        ('x_ul1_', ()),
        ('x_ul2_', ()),
        ('y_l_', ('x_ur',)),
        ('y_r_', ('x_ur',)),
        ('x_ur_', ('x_ul',)),
        ('x_ll_', ('x_ul',)),
        ('x_lr_', ('x_ur',)),
        ('z_ul_', ('x_ul',)),
        ('z_ur_', ('x_ur',)),
        ('z_ll_', ('x_ll',)),
        ('z_lr_', ('x_lr',)),
    ]


class WBElemSymb5ParamVersionX(bu.SymbExpr):

    a, b, c = sp.symbols('a, b, c', positive=True)
    alpha = sp.symbols('alpha')
    beta = sp.symbols('beta')

    sin = sp.sin
    cos = sp.cos
    x_ul_ = (
        (
                (((sin(alpha)-sin(beta)) * sin(alpha+beta) * sin(alpha) * cos(beta) * a)
                /
                (cos(alpha) - cos(beta)))
                + sin(alpha) * sin(beta) * sin(alpha +beta) * a
        )
        /
        (
                sin(alpha) * (cos(alpha+beta) + 1)
        )
    )

    x_ul = sp.symbols('x_ul')
    x_ur = sp.symbols('x_ur')

    sqrt = sp.sqrt
    y_r_ = (
        (
            sqrt(
            sin(alpha)**2 * b**2 - (a * cos(alpha) - x_ur) ** 2
            )
        )
        /
        (sin(alpha))
    )

    y_l_ = (
        (
            sqrt(
            sin(beta)**2 * b**2 - (a * cos(beta) + x_ul) ** 2
            )
        )
        /
        (sin(beta))
    )

    x_ur_ = (
        ( sin(alpha) * x_ul +sin(alpha+beta) * a)
        /
        ( sin(beta) )
    )

    x_ur = sp.symbols('x_ur')

    x_lr_ = 2 * a * cos(alpha) - x_ur
    x_ll_ = -2 * a * cos(beta) - x_ul

    x_lr = sp.symbols('x_lr')
    x_ll = sp.symbols('x_ll')

    z_ul_ = (a + sp.cos(beta)*x_ul)/(sp.sin(beta))
    z_ur_ = (a - sp.cos(alpha)*x_ur)/(sp.sin(alpha))
    z_ll_ = (a + sp.cos(beta)*x_ll)/(sp.sin(beta))
    z_lr_ = (a - sp.cos(alpha)*x_lr)/(sp.sin(alpha))

    symb_model_params = ['alpha', 'beta', 'a', 'b', 'c', ]
    symb_expressions = [
        ('y_l_', ('x_ul',)),
        ('y_r_', ('x_ur',)),
        ('x_ul_', () ),
        ('x_ur_', ('x_ul',)),
        ('x_ll_', ('x_ul',)),
        ('x_lr_', ('x_ur',)),
        ('z_ul_', ('x_ul',)),
        ('z_ur_', ('x_ur',)),
        ('z_ll_', ('x_ll',)),
        ('z_lr_', ('x_lr',)),
    ]

class WBElemSymb5ParamVersionEta(bu.SymbExpr):

    def get_U_ur(x_ur, y_ur, z_ur, a, b, c, V_r_l):
        U_ur_0 = sp.Matrix([a, b, 0])
        V_r_0 = sp.Matrix([c, 0, 0])
        UV_r_0 = V_r_0 - U_ur_0
        L2_U_ur_0 = (U_ur_0.T * U_ur_0)[0]
        L2_UV_r_0 = (UV_r_0.T * UV_r_0)[0]
        U_ur_1 = sp.Matrix([x_ur, y_ur, z_ur])
        UV_r_1 = U_ur_1 - V_r_l
        L2_U_ur_1 = (U_ur_1.T * U_ur_1)[0]
        L2_UV_r_1 = (UV_r_1.T * UV_r_1)[0]
        Eq_L2_U_ur = sp.simplify(sp.Eq(L2_U_ur_1 - L2_U_ur_0, 0))
        y_ur_sol = sp.solve(Eq_L2_U_ur, y_ur)[0]
        Eq_L2_UV_r = sp.simplify(sp.Eq(L2_UV_r_1 - L2_UV_r_0, 0))
        Eq_L2_UV_r_z_ur = Eq_L2_UV_r.subs(y_ur, y_ur_sol)
        z_ur_sol = sp.solve(Eq_L2_UV_r_z_ur, z_ur)[0]
        return sp.Matrix([x_ur, y_ur_sol.subs(z_ur, z_ur_sol), z_ur_sol])

    a, b, c = sp.symbols('a, b, c', positive=True)
    alpha = sp.symbols('alpha', positive=True)
    V_r_l = sp.Matrix([c * sp.sin(alpha), 0, c * sp.cos(alpha)])

    x_ur, y_ur, z_ur = sp.symbols('x_ur, y_ur, z_ur', positive=True)

    U_ur = get_U_ur(x_ur, y_ur, z_ur, a, b, c, V_r_l)

    x_ul = sp.symbols('x_ul', negative=True)

    U_ul = U_ur.subs(x_ur, -x_ul)
    U_ul[0] *= -1

    UU_u_1 = U_ur - U_ul
    L2_UU_u_1 = (UU_u_1.T * UU_u_1)[0]

    eta = sp.symbols('eta')
    x_ul_ = - eta * x_ur
    L2_UU_u_1_eta = L2_UU_u_1.subs(x_ul, x_ul_)

    Eq_L2_UU_u_1 = sp.Eq(L2_UU_u_1_eta - (2 * a) ** 2, 0)
    x_ur_sol_1, x_ur_sol_2 = sp.solve(Eq_L2_UU_u_1.subs(alpha, 0), x_ur)

    x_ur_ = x_ur_sol_1
    y_r_ = U_ur[1]
    z_ur_ = U_ur[2]
    y_l_ = U_ul[1]
    z_ul_ = U_ul[2]

    symb_model_params = ['alpha', 'eta', 'a', 'b', 'c', ]
    symb_expressions = [
        ('y_l_', ('x_ul',)),
        ('y_r_', ('x_ur',)),
        ('x_ul_', ('x_ur',) ),
        ('x_ur_', ()),
        ('z_ul_', ('x_ul',)),
        ('z_ur_', ('x_ur',)),
    ]

class WBElem5Param(bu.InteractiveModel,bu.InjectSymbExpr):
    name = 'waterbomb cell 5p'
    symb_class = WBElemSymb5ParamVersion4

    plot_backend = 'k3d'

    alpha = bu.Float(np.pi/2+1e-5, GEO=True)
    eta = bu.Float(1, GEO=True)
    a = bu.Float(1000, GEO=True)
    b = bu.Float(1000, GEO=True)
    c = bu.Float(1000, GEO=True)
    a_high = bu.Float(2000)
    b_high = bu.Float(2000)
    c_high = bu.Float(2000)

    show_wireframe = tr.Bool

    ipw_view = bu.View(
        bu.Item('alpha', latex=r'\alpha', editor=bu.FloatRangeEditor(
            low=1e-6, high=np.pi / 2, n_steps=100, continuous_update=True)),
        bu.Item('eta', latex=r'\eta', editor=bu.FloatRangeEditor(
            low=1e-6, high=10, n_steps=100, continuous_update=True)),
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
        alpha = self.alpha

        x_ur = self.symb.get_x_ur_()

        y_r = self.symb.get_y_r_(x_ur)
        y_l = self.symb.get_y_l_(x_ur)

        prod_y_rl = y_r * y_l
        if prod_y_rl > 0:
            x_ul = self.symb_get_x_ul1()
        else:
            x_ul = self.symb_get_x_ul2()

        x_lr = self.symb.get_x_lr_(x_ur)
        x_ll = self.symb.get_x_ll_(x_ul)

        z_ul = self.symb.get_z_ul_(x_ul)
        z_ur = self.symb.get_z_ur_(x_ur)
        z_ll = self.symb.get_z_ll_(x_ll)
        z_lr = self.symb.get_z_lr_(x_lr)

        return np.array([
            [0,0,0], # 0 point
            [x_ur, y_r, z_ur], #U++
            [x_ul, y_l, z_ul], #U-+  ul
            [x_lr,-y_r, z_lr], #U+-
            [x_ll,-y_l, z_ll], #U--
            [self.c * sp.sin(alpha), 0, self.c * sp.cos(alpha)],
            [-self.c * sp.sin(alpha), 0, self.c * sp.cos(alpha)],
            ], dtype=np.float_
        )


    def _get_X_Ia_Eta(self):
        alpha = self.alpha

        x_ur = self.symb.get_x_ur_()
        x_ul = self.symb.get_x_ul_(x_ur)
        y_r = self.symb.get_y_r_(x_ur)
        y_l = self.symb.get_y_l_(x_ul)
        z_ul = self.symb.get_z_ul_(x_ul)
        z_ur = self.symb.get_z_ur_(x_ur)

        x_lr = x_ur
        x_ll = x_ul
        z_lr = z_ur
        z_ll = z_ul
        return np.array([
            [0,0,0], # 0 point
            [x_ur, y_r, z_ur], #U++
            [x_ul, y_l, z_ul], #U-+  ul
            [x_lr,-y_r, z_lr], #U+-
            [x_ll,-y_l, z_ll], #U--
            [self.c * sp.sin(alpha), 0, self.c * sp.cos(alpha)],
            [-self.c * sp.sin(alpha), 0, self.c * sp.cos(alpha)],
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

