"""

"""
import bmcs_utils.api as bu
import sympy as sp

import k3d

class WBElemSymb(bu.SymbExpr):

    a, b, c = sp.symbols('a, b, c', positive=True)
    u_2, u_3 = sp.symbols('u_2, u_3', positive=True)
    alpha = sp.symbols('alpha', nonnegative=True)

    U_0 = sp.Matrix([a, b, 0])
    W_0 = sp.Matrix([c, 0, 0])

    U = sp.Matrix([a, u_2, u_3])
    W = sp.Matrix([c * sp.cos(alpha), 0, c * sp.sin(alpha)])

    UW_0 = U_0 - W_0

    L2_U0 = U_0.T * U_0
    L2_UW0 = UW_0.T * UW_0

    UW = U - W

    L2_U = U.T * U
    L2_UW = UW.T * UW

    eq_U = sp.Eq(L2_U[0], L2_U0[0])
    eq_UW = sp.Eq(L2_UW[0], L2_UW0[0])

    u_2_, u_3_ = sp.solve({eq_U, eq_UW}, [u_2, u_3])

    # u_2 = u_2_[1]
    u_3 = u_3_[1]
    u_2 = u_2_[0]
    # u_3 = u_3_[0]

    symb_model_params = ['a', 'b', 'c']
    symb_expressions = [
        ('u_2', ('alpha',)),
        ('u_3', ('alpha',))
    ]


# In[22]:


import traits.api as tr
import numpy as np
import matplotlib.pylab as plt

class WBElem(bu.InteractiveModel,bu.InjectSymbExpr):
    name = 'Waterbomb cell'
    symb_class = WBElemSymb

    plot_backend = 'k3d'

    a = bu.Float(1, GEO=True)
    b = bu.Float(1, GEO=True)
    c = bu.Float(1, GEO=True)

    alpha = bu.Float(1e-5, GEO=True)

    ipw_view = bu.View(
        bu.Item('alpha', latex=r'\alpha', editor=bu.FloatRangeEditor(low=1e-10, high=np.pi/2)),
        bu.Item('a'),
        bu.Item('b'),
        bu.Item('c'),
    )

    X_Ia = tr.Property(depends_on='+GEO')
    '''Array with nodal coordinates I - node, a - dimension
    '''
    @tr.cached_property
    def _get_X_Ia(self):
        alpha = self.alpha
        u_2 = self.symb.get_u_2(alpha)
        u_3 = self.symb.get_u_3(alpha)
        return np.array([
            [0,0,0], # 0 point
            [self.a, u_2, u_3], #U++
            [-self.a,u_2, u_3], #U-+
            [self.a,-u_2, u_3], #U+-
            [-self.a,-u_2, u_3], #U--
            [self.c * np.cos(alpha), 0, self.c * np.sin(alpha)], # W0+
            [-self.c * np.cos(alpha), 0, self.c * np.sin(alpha)] # W0-
            ], dtype=np.float_
        )

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

    # def subplots(self,fig):
    #     ax = fig.add_subplot(1, 1, 1, projection='3d')
    #     return ax
    #
    # def update_plot(self, axes):
    #     ax = axes
    #     x, y, z = self.X_Ia.T
    #     triangles = self.I_Fi
    #     ax.plot_trisurf(x, y, z, triangles=triangles, cmap=plt.cm.Spectral)
    #     ax.set_zlim(0.01, np.max([self.a, self.b, self.c]))
    #     # r = 1.5
    #     # x_ = np.max([r * self.a, r * self.c])
    #     # ax.set_xlim(-x_, +x_ )
    #     # y_ = r * self.b
    #     # ax.set_ylim(-y_, +y_)

    def update_plot(self, k3d_plot):
        wb_cell_mesh_surfaces = k3d.mesh(self.X_Ia.astype(np.float32), self.I_Fi.astype(np.uint32),
                            color_map=k3d.colormaps.basic_color_maps.Jet,
                            attribute=self.X_Ia[:, 2],
                            color_range=[-1.1, 2.01], side='double')
        wb_cell_mesh_lines = k3d.mesh(self.X_Ia.astype(np.float32), self.I_Fi.astype(np.uint32),
                             color=0x000000, wireframe=True)
        k3d_plot += wb_cell_mesh_surfaces
        k3d_plot += wb_cell_mesh_lines

