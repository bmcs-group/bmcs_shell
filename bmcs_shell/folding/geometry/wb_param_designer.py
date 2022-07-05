# from matplotlib import cm
import warnings

import bmcs_utils.api as bu
import k3d
import matplotlib.pyplot as plt
import numpy as np
import traits.api as tr
from bmcs_shell.api import WBTessellation4P
from matplotlib import cm
from scipy.interpolate import interp1d


class WbParamDesigner(bu.Model):
    n = bu.Int(50)
    rand_color = bu.Array
    var1_grid_agnn = bu.Array
    gamma_range = np.linspace(10, 85, 10)
    a_range = np.array([125])
    n_mid_cells = bu.Int(2)
    var1 = {'name': 'span', 'value': 2118.16}
    var2 = {'name': 'height', 'value': 279.54}
    var3 = {'name': 'width', 'value': 501.77}

    eta_of_var1 = bu.List([])
    zeta_of_var1 = bu.List([])
    valid_params = bu.List([])

    etas_zetas_grid = tr.Property(depends_on='n')

    @tr.cached_property
    def _get_etas_zetas_grid(self):
        n = self.n
        etas = np.concatenate((np.linspace(0, 1, int(n / 2))[:-1], np.linspace(1, 10, int(n / 2 + 1))))
        zetas = np.copy(etas)
        etas_grid, zetas_grid = np.meshgrid(etas, zetas)
        return etas_grid, zetas_grid

    def calc_valid_params(self):
        print('Attention: Warnings are suppressed!')
        warnings.filterwarnings('ignore')

        var1, var2, var3 = self.var1, self.var2, self.var3
        a_range = self.a_range
        gamma_range = self.gamma_range
        n_mid_cells = self.n_mid_cells
        etas_grid, zetas_grid = self.etas_zetas_grid
        etas_grid_agnn = np.tile(etas_grid, (len(a_range), len(gamma_range), 1, 1)) # a is a index, g is gamma index
        self.var1_grid_agnn = np.zeros_like(etas_grid_agnn)

        fig_h, ax_h = plt.subplots()
        ax_h.set_title(var1['name'] + '=' + str(var1['value']))
        ax_h.set_ylabel(var2['name'])
        ax_h.set_xlabel('eta/zeta')
        ax_h.set_ylim(-1000, 5000)

        fig, ax = plt.subplots()

        wbt4p = WBTessellation4P(n_phi_plus=n_mid_cells + 1, n_x_plus=2, wireframe_width=5)

        for a_i, a in enumerate(a_range):
            wbt4p.trait_set(a=a)
            valid_var1_2_params = []

            self.eta_of_var1.append([])
            self.zeta_of_var1.append([])

            for gamma_i, gamma in enumerate(gamma_range):

                print('gamma =', np.round(gamma, 1), end='°, ')

                wbt4p.trait_set(gamma=np.deg2rad(gamma))

                # Fill the grid of the variable
                # -------------------------------------------------------
                for i_eta in range(len(etas_grid)):
                    for j_zeta in range(len(zetas_grid)):
                        eta = etas_grid[i_eta, j_zeta]
                        zeta = zetas_grid[i_eta, j_zeta]

                        wbt4p.trait_set(b=eta * a, c=zeta * a)

                        self.var1_grid_agnn[a_i, gamma_i, i_eta, j_zeta] = self.get_var_value(var1, wbt4p, n_mid_cells)

                # Find contour line corresponding to the variable value
                # -------------------------------------------------------
                path, color = self.get_longest_contour_for_var(ax, var1, self.var1_grid_agnn, a_i, gamma_i, gamma)
                self.eta_of_var1[a_i].append(path.vertices[:, 0])
                self.zeta_of_var1[a_i].append(path.vertices[:, 1])

                # Find the possible shell heights considering the fixed var value
                # --------------------------------------------------------------
                var2_array = []
                for eta, zeta in zip(self.eta_of_var1[a_i][gamma_i], self.zeta_of_var1[a_i][gamma_i]):
                    wbt4p.trait_set(b=eta * a, c=zeta * a)
                    var2_array.append(self.get_var_value(var2, wbt4p, n_mid_cells))

                ax_h.plot(self.eta_of_var1[a_i][gamma_i], var2_array, '--', label='eta, $\gamma$=' + str(round(gamma, 1)), color=color)
                ax_h.plot(self.zeta_of_var1[a_i][gamma_i], var2_array, label='zeta, $\gamma$=' + str(round(gamma, 1)), color=color)

                eta_inter, zeta_inter = self.interp(var2['value'], var2_array, self.eta_of_var1[a_i][gamma_i], self.zeta_of_var1[a_i][gamma_i])
                valid_var1_2_params.append([a, gamma, eta_inter, zeta_inter])

                ax_h.plot(eta_inter, var2['value'], 'o', color=color)
                ax_h.plot(zeta_inter, var2['value'], 'x', color=color)

            valid_var1_2_params = np.array(valid_var1_2_params)

            var3_array = []
            for params in valid_var1_2_params:
                a, gamma, eta, zeta = params
                wbt4p.trait_set(a=a, b=eta * a, c=zeta * a, gamma=np.deg2rad(gamma))
                var3_array.append(self.get_var_value(var3, wbt4p, n_mid_cells))

            gamma, eta, zeta = [self.interp1(var3['value'], var3_array, valid_var1_2_params[:, i]) for i in [1, 2, 3]]
            self.valid_params.append(dict(a=a, b=a * eta, c=a * zeta, gamma=np.deg2rad(gamma), n_phi_plus=n_mid_cells + 1))

        ax_h.legend()

        # fig_h.show()
        # fig.show()

        print('valid_params=', self.valid_params)

        return self.valid_params


    def get_span(self, wb_shell, n_mid_cells=2):
        if n_mid_cells == 2:
            span_v = wb_shell.X_Ia[8] - wb_shell.X_Ia[1]
        elif n_mid_cells == 3:
            span_v = wb_shell.X_Ia[13] - wb_shell.X_Ia[1]
        elif n_mid_cells == 4:
            span_v = wb_shell.X_Ia[18] - wb_shell.X_Ia[1]
        elif n_mid_cells == 5:
            span_v = wb_shell.X_Ia[23] - wb_shell.X_Ia[1]
        return np.sqrt(span_v @ span_v)

    def get_shell_height(self, wb_shell, n_mid_cells=2):
        if n_mid_cells == 2:
            return wb_shell.X_Ia[3][2] - wb_shell.X_Ia[8][2]
        elif n_mid_cells == 3:
            return wb_shell.X_Ia[10][2] - wb_shell.X_Ia[13][2]
        elif n_mid_cells == 4:
            return wb_shell.X_Ia[8][2] - wb_shell.X_Ia[18][2]
        elif n_mid_cells == 5:
            return wb_shell.X_Ia[15][2] - wb_shell.X_Ia[23][2]

    def get_shell_width(self, wb_shell, n_mid_cells=2):
        # width of two cells (one cell in mid and two halves to sides)
        span_v = wb_shell.X_Ia[32 + (n_mid_cells - 2) * 8] - wb_shell.X_Ia[20 + (n_mid_cells - 2) * 5]
        return np.sqrt(span_v @ span_v)

    def interp(self, interp_value, values, etas, zetas):
        try:
            f_eta = interp1d(values, etas, kind='linear')
            f_zeta = interp1d(values, zetas, kind='linear')  # maybe try 'cubic' but it doesn't work for few values
            eta_inter = f_eta(interp_value)
            zeta_inter = f_zeta(interp_value)
        except:
            eta_inter, zeta_inter = np.nan, np.nan
        finally:
            return eta_inter, zeta_inter

    def interp1(self, interp_value, values, y):
        try:
            f_y = interp1d(values, y, kind='linear')
            y_inter = f_y(interp_value)
        except:
            y_inter = np.nan
        finally:
            return y_inter

    def get_var_value(self, var, wbt4p, n_mid_cells):
        if var['name'] == 'span':
            return self.get_span(wbt4p, n_mid_cells=n_mid_cells)
        elif var['name'] == 'height':
            return self.get_shell_height(wbt4p, n_mid_cells=n_mid_cells)
        elif var['name'] == 'width':
            return self.get_shell_width(wbt4p, n_mid_cells=n_mid_cells)
        elif var['name'] == 'R_0':
            pass
            # return -wb_cell.R_0
        elif var['name'] == 'curv_angle':
            pass
            # for cell!!
            # return self.get_curv_angle(wb_cell)

    def round_to(self, value, base=5):
        return base * round(value / base)

    def get_curv_angle(self, wb_cell):
        X_Ia = wb_cell.X_Ia
        v_56 = (X_Ia[5] + X_Ia[6]) / 2
        v_12 = (X_Ia[1] + X_Ia[2]) / 2
        v_diff = v_12 - v_56
        oy_n = np.array([0, 1, 0])
        v_diff_n = v_diff / np.linalg.norm(v_diff)
        dot_product = np.dot(oy_n, v_diff_n)
        angle = np.arccos(dot_product)
        return np.rad2deg(angle)

    def plot_eta_zeta_var1(self, a_i, gamma_i):
        # Plot 3d
        # --------
        gamma = self.gamma_range[gamma_i]
        a = self.a_range[a_i]
        etas_grid, zetas_grid = self.etas_zetas_grid
        fig_3d, ax_3d = plt.subplots(subplot_kw={"projection": "3d"})
        ax_3d.set_title('$\gamma$ = ' + str(round(gamma, 1)) + '°, a= ' + str(a))
        ax_3d.set_xlabel(r'$\eta$', fontsize=10)
        ax_3d.set_ylabel(r'$\zeta$', fontsize=10)
        ax_3d.plot_surface(etas_grid, zetas_grid, self.var1_grid_agnn[a_i, gamma_i, ...],
                           linewidth=0, antialiased=False, cmap=cm.coolwarm)
        # fig_3d.show()

    def get_longest_contour_for_var(self, ax, var_name_and_value, var_grid, a_i, gamma_i, gamma):
        ax.set_title(var_name_and_value['name'] + '=' + str(var_name_and_value['value']) + ' contours')
        ax.set_xlabel(r'eta', fontsize=10)
        ax.set_ylabel(r'zeta', fontsize=10)
        # for gamma_i, gamma in enumerate(self.gamma_range):

        self.rand_color = np.random.rand(3, )
        color = self.rand_color
        # TODO: try scipy interp2d or interpn instead of getting data from contour
        #  (however contour enables you to see if there are multiple solutions)
        etas_grid, zetas_grid = self.etas_zetas_grid
        cs = ax.contour(etas_grid, zetas_grid, var_grid[a_i, gamma_i, ...],
                        levels=[var_name_and_value['value']],
                        colors=[color])

        for i, path in enumerate(cs.collections[0].get_paths()):
            length = len(path.vertices)
            if i == 0:
                longest_path = path
            elif length > len(longest_path.vertices):
                longest_path = path

        # Label every other level using strings
        ax.clabel(cs, inline=True, fmt={cs.levels[0]: '$\gamma$=' + str(round(gamma, 1))}, fontsize=10)

        print('path length=', len(path))

        return longest_path, color

if __name__ == '__main__':
    wb_p = WbParamDesigner()
    wb_p.calc_valid_params()
    # print(wb_p.etas_zetas_grid)
