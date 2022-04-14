import bmcs_utils.api as bu

from bmcs_shell.folding.geometry.wb_tessellation.wb_num_tessellation_base import WBNumTessellationBase
from bmcs_shell.folding.geometry.wb_tessellation.wb_num_tessellation import WBNumTessellation
from bmcs_shell.folding.geometry.wb_cell.wb_cell_5p_v2 import \
    WBCell5ParamV2
import traits.api as tr
import numpy as np
from numpy import cos, sin, sqrt
from scipy.optimize import minimize
import k3d
import random

class WBTessellation5PV2(WBNumTessellation):
    name = 'WBTessellation5PV2'

    wb_cell = bu.Instance(WBCell5ParamV2, ())
    tree = ['wb_cell']
    X_Ia = tr.DelegatesTo('wb_cell')
    I_Fi = tr.DelegatesTo('wb_cell')

    # sigma_sol_num = bu.Int(-1, GEO=True)
    # rho_sol_num = bu.Int(-1, GEO=True)
    sol_num = bu.Int(4, GEO=True)

    # Note: Update traits to 6.3.2 in order for the following command to work!!
    @tr.observe('wb_cell.+GEO', post_init=True)
    def update_after_wb_cell_GEO_changes(self, event):
        self.event_geo = not self.event_geo
        self.update_plot(self.pb)

    ipw_view = bu.View(
        *WBNumTessellation.ipw_view.content,
        # bu.Item('sigma_sol_num'),
        # bu.Item('rho_sol_num'),
        bu.Item('sol_num'),
    )

    sol = tr.Property(depends_on='+GEO')
    @tr.cached_property
    def _get_sol(self):
        # rhos, sigmas = self.get_3_cells_angles()
        # print('original sigmas=', sigmas)
        # print('original rhos=', rhos)
        # # rhos = 2 * np.pi - rhos
        # # sigmas = 2 * np.pi - sigmas
        # rhos = -rhos
        # sigmas = -sigmas
        # print('sigmas=', sigmas)
        # print('rhos=', rhos)
        #
        # if self.sigma_sol_num != -1 and self.rho_sol_num != -1:
        #     print('Solution {} was used.'.format(self.sigma_sol_num))
        #     sol = np.array([sigmas[self.sigma_sol_num - 1], rhos[self.rho_sol_num - 1]])
        #     return sol
        #
        # sigma_sol_idx = np.argmin(np.abs(sigmas - sol[0]))
        # rho_sol_idx = np.argmin(np.abs(rhos - sol[1]))
        #
        # if sigma_sol_idx != rho_sol_idx:
        #     print('Warning: sigma_sol_idx != rho_sol_idx, num solution is picked!')
        # else:
        #     diff = np.min(np.abs(sigmas - sol[0])) + np.min(np.abs(rhos - sol[1]))
        #     print('Solution {} was picked (nearst to numerical sol), diff={}'.format(sigma_sol_idx + 1, diff))
        #     sol = np.array([sigmas[sigma_sol_idx], rhos[rho_sol_idx]])

        sol_num = self.sol_num

        # Solving with only 4th solution
        rhos, sigmas = self.get_3_cells_angles(sol_num=sol_num)
        sol = np.array([sigmas[0], rhos[0]])
        print('Ana. solution:', sol)
        return sol

    def get_3_cells_angles(self, sol_num=None):
        a = self.wb_cell.a
        b = self.wb_cell.b
        c = self.wb_cell.c
        gamma = self.wb_cell.gamma
        beta = self.wb_cell.beta

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

        sin_phi1 = sin_psi1plus6
        sin_phi2 = sin_psi5
        sin_phi3 = sin_psi5
        sin_phi4 = sin_psi1plus6

        cos_phi1 = cos_psi1plus6
        cos_phi2 = cos_psi5
        cos_phi3 = cos_psi5
        cos_phi4 = cos_psi1plus6

        # ALWAYS USE THIS TO FIND DEFINITE ANGLE IN [-pi, pi] from sin and cos
        phi1 = np.arctan2(sin_phi1, cos_phi1)
        phi2 = np.arctan2(sin_phi2, cos_phi2)
        phi3 = phi2
        phi4 = phi1

        e2 = (a - c) ** 2 + b ** 2
        e = sqrt(e2)
        f = (a - c) ** 2 + b ** 2 * cos(phi1 + phi3)
        W = 2 * (1 - cos(phi1 + phi3)) * (
                    (a ** 2 - cos(phi2) * cos(phi4) * b ** 2) * c ** 2 * sin(2 * gamma) ** 2 - (
                        cos(phi2) + cos(phi4)) * ((cos(2 * gamma) - 1) * c + 2 * a) * sin(
                2 * gamma) * a * b * c - 2 * a ** 2 * c * (2 * a - c) * (cos(2 * gamma) + 1) + 2 * a ** 2 * b ** 2 * (
                                cos(phi1 + phi3) + 1)) - e2 * (cos(phi2) - cos(phi4)) ** 2 * c ** 2 * sin(2 * gamma) ** 2

        T_rho = (cos(phi1 + phi3) - 1) * (
                    a * (b ** 2 * (cos(phi1 + phi3) + 1) + 2 * a * (a - c)) * (cos(2 * gamma) + 1)
                    + b * ((e2 + f) * cos(phi2) + (
                        a - c) * c * (cos(phi2) + cos(phi4))) * sin(2 * gamma) - 2 * a * b ** 2 * (
                                cos(phi1 + phi3) + 1))

        T_sigma = (cos(phi1 + phi3) - 1) * (
                    a * (b ** 2 * (cos(phi1 + phi3) + 1) + 2 * a * (a - c)) * (cos(2 * gamma) + 1)
                    + b * ((e2 + f) * cos(phi4) + (
                        a - c) * c * (cos(phi2) + cos(phi4))) * sin(2 * gamma) - 2 * a * b ** 2 * (
                                cos(phi1 + phi3) + 1))

        print('W:', W)
        print('T_rho:', T_rho)
        print('T_sigma:', T_sigma)

        rhos = []
        sigmas = []
        get_angle = lambda x: np.arctan(x) * 2

        if sol_num == 1 or sol_num is None:
            sol_P1_t_1 = (b * (a - c) * (cos(phi1 + phi3) - 1) * sqrt(W) - b * (
                    a * b * c * sin(phi1 + phi3) * (cos(phi1 + phi3) - 1) * cos(2 * gamma) + (
                    cos(phi2) * f - cos(phi4) * e2) * sin(phi1 + phi3) * c * sin(
                2 * gamma) + a * b * sin(phi1 + phi3) * (cos(phi1 + phi3) - 1) * (2 * a - c))
                          + (cos(phi1 + phi3) - 1) * (e2 + f) * sqrt(
                        4 * a * b ** 2 * (cos(phi2) * sin(2 * gamma) * b + (cos(2 * gamma) + 1) * a) - (
                                a ** 2 + b ** 2) * (
                                cos(phi2) * sin(2 * gamma) * b + (cos(2 * gamma) + 1) * a) ** 2)) / (
                                     e * (b * sin(phi1 + phi3) * sqrt(W) - T_rho))

            sol_Q1_u_1 = (b * (a - c) * (cos(phi1 + phi3) - 1) * sqrt(W) - b * (
                    a * b * c * sin(phi1 + phi3) * (cos(phi1 + phi3) - 1) * cos(2 * gamma) + (
                    cos(phi4) * f - cos(phi2) * e2) * sin(phi1 + phi3) * c * sin(
                2 * gamma) + a * b * sin(phi1 + phi3) * (cos(phi1 + phi3) - 1) * (2 * a - c))
                          + (cos(phi1 + phi3) - 1) * (e2 + f) * sqrt(
                        4 * a * b ** 2 * (cos(phi4) * sin(2 * gamma) * b + (cos(2 * gamma) + 1) * a) - (
                                a ** 2 + b ** 2) * (
                                cos(phi4) * sin(2 * gamma) * b + (cos(2 * gamma) + 1) * a) ** 2)) / (
                                     e * (b * sin(phi1 + phi3) * sqrt(W) - T_sigma))

            rhos.append(get_angle(sol_P1_t_1))
            sigmas.append(get_angle(sol_Q1_u_1))

            print('sol_P1_t_1:', sol_P1_t_1)
            print('sol_Q1_u_1:', sol_Q1_u_1)

        if sol_num == 2 or sol_num is None:
            sol_P1_t_2 = (b * (a - c) * (cos(phi1 + phi3) - 1) * sqrt(W) - b * (
                    a * b * c * sin(phi1 + phi3) * (cos(phi1 + phi3) - 1) * cos(2 * gamma) + (
                    cos(phi2) * f - cos(phi4) * e2) * sin(phi1 + phi3) * c * sin(
                2 * gamma) + a * b * sin(phi1 + phi3) * (cos(phi1 + phi3) - 1) * (2 * a - c))
                          - (cos(phi1 + phi3) - 1) * (e2 + f) * sqrt(
                        4 * a * b ** 2 * (cos(phi2) * sin(2 * gamma) * b + (cos(2 * gamma) + 1) * a) - (
                                a ** 2 + b ** 2) * (
                                cos(phi2) * sin(2 * gamma) * b + (cos(2 * gamma) + 1) * a) ** 2)) / (
                                     e * (b * sin(phi1 + phi3) * sqrt(W) - T_rho))

            sol_Q1_u_2 = (b * (a - c) * (cos(phi1 + phi3) - 1) * sqrt(W) - b * (
                    a * b * c * sin(phi1 + phi3) * (cos(phi1 + phi3) - 1) * cos(2 * gamma) + (
                    cos(phi4) * f - cos(phi2) * e2) * sin(phi1 + phi3) * c * sin(
                2 * gamma) + a * b * sin(phi1 + phi3) * (cos(phi1 + phi3) - 1) * (2 * a - c))
                          - (cos(phi1 + phi3) - 1) * (e2 + f) * sqrt(
                        4 * a * b ** 2 * (cos(phi4) * sin(2 * gamma) * b + (cos(2 * gamma) + 1) * a) - (
                                a ** 2 + b ** 2) * (
                                cos(phi4) * sin(2 * gamma) * b + (cos(2 * gamma) + 1) * a) ** 2)) / (
                                     e * (b * sin(phi1 + phi3) * sqrt(W) - T_sigma))
            rhos.append(get_angle(sol_P1_t_2))
            sigmas.append(get_angle(sol_Q1_u_2))

            print('sol_P1_t_2:', sol_P1_t_2)
            print('sol_Q1_u_2:', sol_Q1_u_2)

        if sol_num == 3 or sol_num is None:
            sol_P2_t_1 = (-b * (a - c) * (cos(phi1 + phi3) - 1) * sqrt(W) - b * (
                    a * b * c * sin(phi1 + phi3) * (cos(phi1 + phi3) - 1) * cos(2 * gamma) + (
                    cos(phi2) * f - cos(phi4) * e2) * sin(phi1 + phi3) * c * sin(
                2 * gamma) + a * b * sin(phi1 + phi3) * (cos(phi1 + phi3) - 1) * (2 * a - c))
                          + (cos(phi1 + phi3) - 1) * (e2 + f) * sqrt(
                        4 * a * b ** 2 * (cos(phi2) * sin(2 * gamma) * b + (cos(2 * gamma) + 1) * a) - (
                                a ** 2 + b ** 2) * (
                                cos(phi2) * sin(2 * gamma) * b + (cos(2 * gamma) + 1) * a) ** 2)) / (
                                     e * (-b * sin(phi1 + phi3) * sqrt(W) - T_rho))

            sol_Q2_u_1 = (-b * (a - c) * (cos(phi1 + phi3) - 1) * sqrt(W) - b * (
                    a * b * c * sin(phi1 + phi3) * (cos(phi1 + phi3) - 1) * cos(2 * gamma) + (
                    cos(phi4) * f - cos(phi2) * e2) * sin(phi1 + phi3) * c * sin(
                2 * gamma) + a * b * sin(phi1 + phi3) * (cos(phi1 + phi3) - 1) * (2 * a - c))
                          + (cos(phi1 + phi3) - 1) * (e2 + f) * sqrt(
                        4 * a * b ** 2 * (cos(phi4) * sin(2 * gamma) * b + (cos(2 * gamma) + 1) * a) - (
                                a ** 2 + b ** 2) * (
                                cos(phi4) * sin(2 * gamma) * b + (cos(2 * gamma) + 1) * a) ** 2)) / (
                                     e * (-b * sin(phi1 + phi3) * sqrt(W) - T_sigma))

            rhos.append(get_angle(sol_P2_t_1))
            sigmas.append(get_angle(sol_Q2_u_1))

            print('sol_P2_t_1:', sol_P2_t_1)
            print('sol_Q2_u_1:', sol_Q2_u_1)

        if sol_num == 4 or sol_num is None:
            sol_P2_t_2 = (-b * (a - c) * (cos(phi1 + phi3) - 1) * sqrt(W) - b * (
                    a * b * c * sin(phi1 + phi3) * (cos(phi1 + phi3) - 1) * cos(2 * gamma) + (
                    cos(phi2) * f - cos(phi4) * e2) * sin(phi1 + phi3) * c * sin(
                2 * gamma) + a * b * sin(phi1 + phi3) * (cos(phi1 + phi3) - 1) * (2 * a - c))
                          - (cos(phi1 + phi3) - 1) * (e2 + f) * sqrt(
                        4 * a * b ** 2 * (cos(phi2) * sin(2 * gamma) * b + (cos(2 * gamma) + 1) * a) - (
                                a ** 2 + b ** 2) * (
                                cos(phi2) * sin(2 * gamma) * b + (cos(2 * gamma) + 1) * a) ** 2)) / (
                                     e * (-b * sin(phi1 + phi3) * sqrt(W) - T_rho))

            sol_Q2_u_2 = (-b * (a - c) * (cos(phi1 + phi3) - 1) * sqrt(W) - b * (
                    a * b * c * sin(phi1 + phi3) * (cos(phi1 + phi3) - 1) * cos(2 * gamma) + (
                    cos(phi4) * f - cos(phi2) * e2) * sin(phi1 + phi3) * c * sin(
                2 * gamma) + a * b * sin(phi1 + phi3) * (cos(phi1 + phi3) - 1) * (2 * a - c))
                          - (cos(phi1 + phi3) - 1) * (e2 + f) * sqrt(
                        4 * a * b ** 2 * (cos(phi4) * sin(2 * gamma) * b + (cos(2 * gamma) + 1) * a) - (
                                a ** 2 + b ** 2) * (
                                cos(phi4) * sin(2 * gamma) * b + (cos(2 * gamma) + 1) * a) ** 2)) / (
                                     e * (-b * sin(phi1 + phi3) * sqrt(W) - T_sigma))
            print('sol_P2_t_2:', sol_P2_t_2)
            print('sol_Q2_u_2:', sol_Q2_u_2)
            rhos.append(get_angle(sol_P2_t_2))
            sigmas.append(get_angle(sol_Q2_u_2))

        # sol_P is tan(rho/2), sol_Q is tan(sigma/2)

        return -np.array(rhos), -np.array(sigmas)