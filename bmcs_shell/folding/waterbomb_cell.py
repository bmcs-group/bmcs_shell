"""

"""
import bmcs_utils.api as bu
import sympy as sp
from sympy.algebras.quaternion import Quaternion

class WBElemSymb(bu.SymbExpr):

    a, b, c = sp.symbols('a, b, c', positive=True)
    u_2, u_3 = sp.symbols('u_2, u_3', positive=True)
    alpha = sp.symbols('alpha', positive=True)

    U_0 = sp.Matrix([a, b, 0])
    W_0 = sp.Matrix([c, 0, 0])

    U = sp.Matrix([a, u_2, u_3])
    W = sp.Matrix([c * sp.cos(alpha), 0, c * sp.sin(alpha)])

    UW_0 = W_0 - U_0

    L2_U0 = (U_0.T * U_0)[0]
    L2_UW0 = (UW_0.T * UW_0)[0]

    UW = U - W

    L2_U = (U.T * U)[0]
    L2_UW = (UW.T * UW)[0]

    u2_test = sp.solve(L2_U - L2_U0, u_2)[0]
    u3_test = sp.solve((L2_UW - L2_UW0).subs(u_2, u2_test), u_3)[0]
    u_3_ = u3_test
    u_2_ = u2_test.subs(u_3, u3_test)

    U_sol = U.subs({u_2: u_2_, u_3: u_3_})
    W_sol = W.subs({u_2: u_2_, u_3: u_3_})
    V_UW_sol = U_sol - W_sol
    L_UW_sol = sp.sqrt(V_UW_sol[1] ** 2 + V_UW_sol[2] ** 2)
    theta_sol = sp.simplify(2 * sp.asin( V_UW_sol[2] / L_UW_sol))

    U0 = U.subs({u_2: u_2_, u_3: u_3_})
    W0 = W.subs({u_2: u_2_, u_3: u_3_})
    WC = sp.Matrix([-W0[0], W0[1], W0[2]])

    d_1, d_2, d_3 = sp.symbols('d_1, d_2, d_3')
    D = sp.Matrix([d_1, d_2, d_3])
    UD = D + U0
    d_subs = sp.solve(UD - WC, [d_1, d_2, d_3])
    D_ = D.subs(d_subs)

    theta = sp.symbols('theta')
    x_1, x_2, x_3 = sp.symbols('x_1, x_2, x_3')

    q_theta = Quaternion.from_axis_angle([1, 0, 0], theta)
    X_rot = q_theta.rotate_point((x_1, x_2, x_3), q_theta)
    X_theta_a = sp.simplify(sp.Matrix(X_rot))

    symb_model_params = ['a', 'b', 'c', ]
    symb_expressions = [
        ('u_2_', ('alpha',)),
        ('u_3_', ('alpha',)),
        ('U0', ('alpha',)),
        ('UD', ('alpha',)),
        ('W0', ('alpha',)),
        ('WC', ('alpha',)),
        ('X_theta_a', ('alpha', 'theta', 'x_1', 'x_2', 'x_3')),
        ('D_', ('alpha',)),
        ('theta_sol', ('alpha',))
    ]

from ipyvolume import figure

# In[22]:

sp.algebras.quaternion.Quaternion

import traits.api as tr
import numpy as np
import matplotlib.pylab as plt

class WBElem(bu.InteractiveModel,bu.InjectSymbExpr):
    name = 'Waterbomb cell'
    symb_class = WBElemSymb

    a = bu.Float(1, GEO=True)
    b = bu.Float(1, GEO=True)
    c = bu.Float(1, GEO=True)

    alpha = bu.Float(1e-5, GEO=True)

    d_1 = bu.Float(0, GEO=True)
    d_2 = bu.Float(0, GEO=True)
    d_3 = bu.Float(0, GEO=True)

    theta = bu.Float(0, GEO=True)

    ipw_view = bu.View(
        bu.Item('alpha', latex = r'\alpha', editor=bu.FloatRangeEditor(low=1e-10,high=np.pi/2)),
        bu.Item('theta', latex = r'\theta', editor=bu.FloatRangeEditor(low=-np.pi/2,high=np.pi/2)),
        bu.Item('a'),
        bu.Item('b'),
        bu.Item('c'),
        bu.Item('d_1'),
        bu.Item('d_2'),
        bu.Item('d_3'),
    )

    X_Ia = tr.Property(depends_on='+GEO')
    '''Array with nodal coordinates I - node, a - dimension
    '''
    @tr.cached_property
    def _get_X_Ia(self):
        alpha = self.alpha
        u_2 = self.symb.get_u_2_(alpha)
        u_3 = self.symb.get_u_3_(alpha)
        return np.array([
            [0,0,0], # 0 point
            [self.a, u_2, u_3], #U++
            [-self.a, u_2, u_3], #U-+
            [self.a,-u_2, u_3], #U+-
            [-self.a,-u_2, u_3], #U--
            [self.c * np.cos(alpha), 0, self.c * np.sin(alpha)], # W0+
            [-self.c * np.cos(alpha), 0, self.c * np.sin(alpha)] # W0-
            ], dtype=np.float_
        )

    X_theta_Ia = tr.Property(depends_on='+GEO')
    '''Array with nodal coordinates I - node, a - dimension
    '''
    @tr.cached_property
    def _get_X_theta_Ia(self):
        D_a = self.symb.get_D_(self.alpha).T
        theta = self.symb.get_theta_sol(self.alpha)
        XD_Ia = D_a + self.X_Ia
        X_center = XD_Ia[1,:]

        # X_pulled_back =  (XD_Ia - X_center[np.newaxis,:])
        # X_rot_Ia = np.array([
        #     self.symb.get_X_theta_a(self.alpha, -theta,
        #                             X_a[0], X_a[1], X_a[2]).flatten()
        #     for X_a in (XD_Ia - X_center[np.newaxis,:])
        # ], dtype=np.float_)
        # #return X_rot_Ia + X_center[np.newaxis,:]

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

    def subplots(self,fig):
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        return ax

    def update_plot(self, axes):
        ax = axes
        x, y, z = self.X_Ia.T
        triangles = self.I_Fi
        ax.plot_trisurf(x, y, z, triangles=triangles, cmap=plt.cm.Spectral)
        ax.set_zlim(0.01, np.max([self.a, self.b, self.c]))
        x, y, z = self.X_theta_Ia.T
        ax.plot_trisurf(x, y, z, triangles=triangles, cmap=plt.cm.Spectral)
        # r = 1.5
        # x_ = np.max([r * self.a, r * self.c])
        # ax.set_xlim(-x_, +x_ )
        # y_ = r * self.b
        # ax.set_ylim(-y_, +y_)


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
    print('shapes')
    print('q1', q1.shape, 'u', u.shape)
    zero_re = np.zeros((u.shape[0], u.shape[1]), dtype='f')
    print('zero_re', zero_re.shape)
    q2 = np.concatenate([zero_re[:, :, np.newaxis], u], axis=2)
    print('q2', q2.shape)
    q2 = np.rollaxis(q2, 2)
    print('q2', q2.shape)
    q12 = q_mult(q1[:, :, np.newaxis], q2[:, :, :])
    print('q12', q12.shape)
    q_con = q_conjugate(q1)
    print('q_con', q_con.shape)
    q = q_mult(q12, q_con[:, :, np.newaxis])
    print('q', q.shape)
    q = np.rollaxis(np.rollaxis(q, 2), 2)
    print('q', q.shape)
    return q[:, :, 1:]


def axis_angle_to_q(v, theta):
    v_ = v_normalize(v, axis=1)
    x, y, z = v_.T
    print('x,y,z', x, y, z)
    theta = theta / 2
    print('theta', theta)
    w = np.cos(theta)
    x = x * np.sin(theta)
    y = y * np.sin(theta)
    z = z * np.sin(theta)
    print('x,y,z', x, y, z)
    return np.array([w, x, y, z], dtype='f')


def q_to_axis_angle(q):
    w, v = q[0, :], q[1:, :]
    theta = np.arccos(w) * 2.0
    return theta, v_normalize(v)

