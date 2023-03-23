import bmcs_utils.api as bu
import k3d
import traits.api as tr
import numpy as np
from bmcs_shell.folding.geometry.wb_cell.wb_cell import WBCell
from numpy import sin, cos, sqrt


class WBCell5Param2Betas(WBCell):
    name = 'WBCell5Param2Betas'

    plot_backend = 'k3d'

    gamma = bu.Float(np.pi / 6, GEO=True)

    eta = bu.Float(1.5, GEO=True) # b/a
    zeta = bu.Float(0.8, GEO=True) # c/a

    a = bu.Float(500, GEO=True)

    b = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_b(self):
        return self.eta * self.a if self.eta * self.a != 0 else 0.0001

    c = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_c(self):
        return self.zeta * self.a if self.zeta * self.a != 0 else 0.0001

    beta_u = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_beta_u(self):
        return round(self.beta_u_sym + self.delta_beta_u, 8)

    delta_beta_u = bu.Float(0, GEO=True) # delta_beta_u
    beta_u_sym = tr.Property(depends_on='+GEO') # beta_u_sym
    @tr.cached_property
    def _get_beta_u_sym(self):
        """ This is the value of beta_u that makes the cell symmetric, derived in wb_cell_4p_deriving_beta_u_sym.ipynb"""
        return np.arccos(self.a * (1 - 2 * sin(self.gamma)) / sqrt(self.a ** 2 + self.b ** 2))


    beta_l = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_beta_l(self):
        return round(self.beta_l_sym + self.delta_beta_l, 8)

    delta_beta_l = bu.Float(0, GEO=True)
    beta_l_sym = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_beta_l_sym(self):
        """ This is the value of beta_l that makes the cell symmetric, derived in wb_cell_4p_deriving_beta_l_sym.ipynb"""
        return np.arccos(self.a * (1 - 2 * sin(self.gamma)) / sqrt(self.a ** 2 + self.b ** 2))

    continuous_update = True

    ipw_view = bu.View(
        bu.Item('gamma', latex=r'\gamma', editor=bu.FloatRangeEditor(
            low=1e-6, high=np.pi / 2, n_steps=501, continuous_update=continuous_update)),
        # bu.Item('beta_u', latex=r'\beta_u', editor=bu.FloatRangeEditor(
        #     low=1e-6, high=np.pi - 1e-6, n_steps=501, continuous_update=continuous_update)),
        bu.Item('delta_beta_u', latex=r'\Delta\beta_u', editor=bu.FloatRangeEditor(
            low=-3, high=3, n_steps=601, continuous_update=continuous_update)),
        bu.Item('delta_beta_l', latex=r'\Delta\beta_l', editor=bu.FloatRangeEditor(
            low=-3, high=3, n_steps=601, continuous_update=continuous_update)),
        # bu.Item('a', latex='a', editor=bu.FloatRangeEditor(
        #     low=1e-6, high=2000, n_steps=201, continuous_update=continuous_update)),
        # bu.Item('b', latex='b', editor=bu.FloatRangeEditor(
        #     low=1e-6, high=2000, n_steps=201, continuous_update=continuous_update)),
        # bu.Item('c', latex='c', editor=bu.FloatRangeEditor(
        #     low=1e-6, high=2000, n_steps=201, continuous_update=continuous_update)),
        bu.Item('eta', latex='\eta', editor=bu.FloatRangeEditor(
            low=0, high=50, n_steps=1001, continuous_update=continuous_update)),
        bu.Item('zeta', latex='\zeta', editor=bu.FloatRangeEditor(
            low=0, high=50, n_steps=1001, continuous_update=continuous_update)),
        *WBCell.ipw_view.content,
    )

    X_Ia = tr.Property(depends_on='+GEO')
    '''Array with nodal coordinates I - node, a - dimension
    '''

    @tr.cached_property
    def _get_X_Ia(self):
        return self.get_cell_vertices()

    def get_cell_vertices(self):
        c = self.c
        gamma = self.gamma
        U_ur, U_ul = self.get_sol_for_beta(beta_name='beta_u')
        U_lr, U_ll = self.get_sol_for_beta(beta_name='beta_l')
        V_r = np.array([c * sin(gamma), 0, c * cos(gamma)])
        V_l = np.array([-c * sin(gamma), 0, c * cos(gamma)])

        X_Ia = np.vstack((np.zeros(3), U_lr, U_ll, U_ur, U_ul, V_r, V_l)).astype(np.float32)
        return X_Ia

    def get_sol_for_beta(self, beta_name='beta_u'):
        a = self.a
        b = self.b
        gamma = self.gamma
        beta = self.beta_u if beta_name == 'beta_u' else self.beta_l

        # phi1 is angle between OU_ur line and z axis
        cos_psi1 = ((b ** 2 - a ** 2) - a * sqrt(a ** 2 + b ** 2) * cos(beta)) / (b * sqrt(a ** 2 + b ** 2) * sin(beta))
        sin_psi1 = sqrt(
            a ** 2 * (3 * b ** 2 - a ** 2) + 2 * a * (b ** 2 - a ** 2) * sqrt(a ** 2 + b ** 2) * cos(beta) - (
                    a ** 2 + b ** 2) ** 2 * cos(beta) ** 2) / (b * sqrt(a ** 2 + b ** 2) * sin(beta))
        cos_psi5 = (sqrt(a ** 2 + b ** 2) * cos(beta) - a * cos(2 * gamma)) / (b * sin(2 * gamma))
        sin_psi5 = sqrt(b ** 2 + 2 * a * sqrt(a ** 2 + b ** 2) * cos(beta) * cos(2 * gamma) - (a ** 2 + b ** 2) * (
                cos(beta) ** 2 + cos(2 * gamma) ** 2)) / (b * sin(2 * gamma))
        cos_psi6 = (a - sqrt(a ** 2 + b ** 2) * cos(beta) * cos(2 * gamma)) / (
                sqrt(a ** 2 + b ** 2) * sin(beta) * sin(2 * gamma))
        sin_psi6 = sqrt(b ** 2 + 2 * a * sqrt(a ** 2 + b ** 2) * cos(beta) * cos(2 * gamma) - (a ** 2 + b ** 2) * (
                cos(beta) ** 2 + cos(2 * gamma) ** 2)) / (sqrt(a ** 2 + b ** 2) * sin(beta) * sin(2 * gamma))
        cos_psi1plus6 = cos_psi1 * cos_psi6 - sin_psi1 * sin_psi6
        sin_psi1plus6 = sin_psi1 * cos_psi6 + cos_psi1 * sin_psi6

        cos_phi1 = cos_psi1plus6
        cos_phi2 = cos_psi5
        cos_phi3 = cos_psi5
        cos_phi4 = cos_psi1plus6
        sin_phi1 = sin_psi1plus6
        sin_phi2 = sin_psi5
        sin_phi3 = sin_psi5
        sin_phi4 = sin_psi1plus6

        if beta_name == 'beta_u':
            U_ur = np.array(
                [a * sin(gamma) - b * cos_phi1 * cos(gamma), b * sin_phi1, a * cos(gamma) + b * cos_phi1 * sin(gamma)])
            U_ul = np.array(
                [-a * sin(gamma) + b * cos_phi2 * cos(gamma), b * sin_phi2, a * cos(gamma) + b * cos_phi2 * sin(gamma)])
            return U_ur, U_ul
        else:
            U_lr = np.array(
                [a * sin(gamma) - b * cos_phi3 * cos(gamma), -b * sin_phi3, a * cos(gamma) + b * cos_phi3 * sin(gamma)])
            U_ll = np.array(
                [-a * sin(gamma) + b * cos_phi4 * cos(gamma), -b * sin_phi4, a * cos(gamma) + b * cos_phi4 * sin(gamma)])
            return U_lr, U_ll