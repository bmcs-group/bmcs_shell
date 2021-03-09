"""

"""
import bmcs_utils.api as bu
import sympy as sp
from sympy.algebras.quaternion import Quaternion
import k3d
import traits.api as tr
import numpy as np
import math


class WBElemSymb5Param(bu.SymbExpr):

    a, b, c = sp.symbols('a, b, c', positive=True)
    alpha = sp.symbols('alpha')
    beta = sp.symbols('beta')

    y_r_ = (sp.sqrt(sp.sin(alpha)**2 * b**2
                    - (sp.cos(alpha) - (sp.cos(beta))/(sp.cos(alpha+beta)))**2 * a**2)
            /
            (sp.sin(alpha))
            )

    y_l_ = (
            (sp.sin(alpha) * ((1 - sp.cos(alpha+beta))*a**2 + sp.cos(alpha+beta)*b**2))
            /
            (sp.cos(alpha+beta)*
             sp.sqrt(sp.sin(alpha)**2 * b**2 -
                     (sp.cos(beta)-(sp.cos(alpha))/(sp.cos(alpha+beta)) )**2 * a**2)
             )
    )

    xy_l_ = (
            (sp.sin(alpha) * (sp.cos(alpha + beta) * (b ** 2 - a ** 2) + a**2) )
            /
            ((sp.cos(alpha + beta) *
              sp.sqrt(b ** 2 * sp.sin(alpha) ** 2 - a ** 2 * (sp.cos(alpha) * sp.cos(alpha +beta)
                                                - sp.cos(beta)) ** 2 / sp.cos(alpha + beta) ** 2))
            )
    )

    y_l, y_r = sp.symbols('y_r, y_r')

    x_ul_ = - a * sp.cos(beta) + sp.sin(beta) * sp.sqrt(b**2 - y_l**2)
    x_ll_ = - a * sp.cos(beta) + sp.sin(beta) * sp.sqrt(b**2 - y_l**2)
    x_r_ = (a*sp.cos(beta))/(sp.cos(alpha+beta))
    x_ur_ = x_r_
    x_lr_ = x_r_

    x_ur, x_ul, x_ll, x_lr = sp.symbols('x_ur, x_ul, x_ll, x_lr')

    z_ul_ = (a + sp.cos(beta)*x_ul)/(sp.sin(beta))
    z_ur_ = (a - sp.cos(alpha)*x_ur)/(sp.sin(alpha))
    z_ll_ = (a + sp.cos(beta)*x_ll)/(sp.sin(beta))
    z_lr_ = (a - sp.cos(alpha)*x_lr)/(sp.sin(alpha))

    symb_model_params = ['alpha', 'beta', 'a', 'b', 'c', ]
    symb_expressions = [
        ('y_l_', ()),
        ('y_r_', ()),
        ('x_ul_', ('y_l',)),
        ('x_ur_', ()),
        ('x_ll_', ('y_l',)),
        ('x_lr_', ()),
        ('z_ul_', ('x_ul',)),
        ('z_ur_', ('x_ur',)),
        ('z_ll_', ('x_ll',)),
        ('z_lr_', ('x_lr',)),
    ]


class WBElem5Param(bu.InteractiveModel,bu.InjectSymbExpr):
    name = 'waterbomb cell 5p'
    symb_class = WBElemSymb5Param

    plot_backend = 'k3d'

    alpha = bu.Float(1e-5, GEO=True)
    beta = bu.Float(1e-5, GEO=True)
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
        bu.Item('beta', latex=r'\beta', editor=bu.FloatRangeEditor(
            low=1e-6, high=np.pi / 2, n_steps=100, continuous_update=True)),
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
        beta = self.beta
        a, b, c = self.a, self.b, self.c

        y_l = self.symb.get_y_l_()
        y_r = self.symb.get_y_r_()
        x_ul = self.symb.get_x_ul_( y_l )
        x_ur = self.symb.get_x_ur_()
        x_ll = self.symb.get_x_ll_( y_l )
        x_lr = self.symb.get_x_lr_()
        z_ul = self.symb.get_z_ul_(x_ul)
        z_ur = self.symb.get_z_ur_(x_ur)
        z_ll = self.symb.get_z_ll_(x_ll)
        z_lr = self.symb.get_z_lr_(x_lr)

        print('upper left')
        print(x_ul, y_l, z_ul)
        return np.array([
            [0,0,0], # 0 point
            [x_ur, y_r, z_ur], #U++
#            [a, b, 0], #U++
            [x_ul, y_l, z_ul], #U-+  ul
#            [-a, b, 0], #U-+  ul
            [x_lr,-y_r, z_lr], #U+-
#            [a,-b, 0], #U+-
            [x_ll,-y_l, z_ll], #U--
#            [-a,-b, 0], #U--
            [self.c * np.cos(alpha), 0, self.c * np.sin(alpha)], # W0+
            [-self.c * np.cos(beta), 0, self.c * np.sin(beta)] # W0-
            ], dtype=np.float_
        )

        return np.array([
            [0,0,0], # 0 point
            [x_ur, y_r, z_ur], #U++
            [x_ul, y_l, z_ul], #U-+  ul
            [x_lr,-y_r, z_lr], #U+-
            [x_ll,-y_l, z_ll], #U--
            [self.c * np.cos(alpha), 0, self.c * np.sin(alpha)], # W0+
            [-self.c * np.cos(beta), 0, self.c * np.sin(beta)] # W0-
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

    def plot_k3d(self, k3d_plot):
        self.wb_mesh = k3d.mesh(self.X_Ia.astype(np.float32),
                                 self.I_Fi.astype(np.uint32),
                                 color=0x999999,
                                 side='double')
        k3d_plot += self.wb_mesh

        if self.show_wireframe:
            self.wb_mesh_wireframe = k3d.mesh(self.X_Ia.astype(np.float32),
                                            self.I_Fi.astype(np.uint32),
                                            color=0x000000,
                                            wireframe=True)

            k3d_plot += self.wb_mesh_wireframe

    def update_plot(self, k3d_plot):
        self._assign_mesh_data(self.wb_mesh)
        if self.show_wireframe:
            self._assign_mesh_data(self.wb_mesh_wireframe)

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

