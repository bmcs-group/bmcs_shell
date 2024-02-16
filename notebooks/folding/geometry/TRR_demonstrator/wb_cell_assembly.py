import traits.api as tr
from wb_scaned_cell import WBScannedCell
import numpy as np
import itertools
from matplotlib.patches import Arc

class WBCellAssembly(tr.HasTraits):
    """Assembly of the waterbomb cells
    """

    modules = tr.Dict(tr.Str, WBScannedCell,value={})

    modules_list = tr.Property(depends_on='modules')
    @tr.cached_property
    def _get_modules_list(self):
        return [(k, *v, c) for (k, v), c in 
                zip(sorted(self.modules.items()), self.colors) ]     

    colors = tr.List([0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff, 0x00ffff,
          0xff0099, 0xff9900, 0x99ff00, 0x00ff99, 0x9900ff, 0x0099ff,
          0xff9999, 0x99ff99, 0x9999ff, 0xffff99, 0xff99ff, 0x99ffff])   

    wbs = tr.Property(depends_on='modules')
    """Array of waterbomb cells"""

    n_samples = tr.Int(5000)
    """Number fo samples used to find the best combination with a minimum misfit
    """

    cell_enum = tr.Array(value=np.arange(1,13), dtype=np.int_)
    """Current cell enumeration
    """

    plug_pi = tr.Array(value=[
        [0, 8, 1, 1],
        [0, 2, -1, 1],
        [2, 1, 1, -1],
        [2, 3, 1, 1],
        [3, 9, 1, 1],
        [3, 5, -1, 1],
        [5, 4, 1, -1],
        [4, 10, 1, -1],
        [5, 6, 1, 1],
        [5, 7, -1, 1],
        [7, 11, 1, -1]
    ], dtype=np.int_)

    misfit_pi = tr.Array(value=[
        [8, 3, -1, 1],
        [1, 10, -1, 1],
        [2, 4, -1, 1],
        [9, 6, -1, 1],
        [4, 11, -1, 1]
    ], dtype=np.int_)

    def _get_wbs(self, modules):
        return np.array([WBScannedCell(file_path=file, label=key,
                                            rotate_system=rotate_system) 
            for key, file, rotate_system, _ in self.modules_list ], 
            dtype=np.object_)

    def get_slit_seq(self, slit_Si):
        """Distribute the above neighboring information given by 
        the four indexes (fixed, plugged, diagonal, y_dir) such that 
        they can be used in loops over the plug scenario, or for the
        calculation of the misfit"""
        cell_pair_pC = self.cell_enum[slit_Si[:,:2]] - 1
        from_slit_diag_p = slit_Si[:, 2]
        from_slit_dir_p = slit_Si[:, 3]
        return cell_pair_pC, from_slit_diag_p, from_slit_dir_p

    def plug_modules(self):
        """Based on the plug scenario prescribed by an array of the 
        cell index couples, with the first one specifying the fixed and 
        the second, the plugged cell and the third index specifying the 
        diagonal and the fourth, the vertical direction of the slit, 
        i.e. y-direction assemble the whole shell
        """
        for wb_link, diag, dir in zip(*self.get_slit_seq(self.plug_pi)):
            wb_fixed, wb_plugged = self.wbs[wb_link]
            wb_plugged.plug_into(wb_fixed, diag, dir)

    def reset_modules(self):
        """Reset the reference points of the cells and the rotation around 
        the x-axis 
        """
        for wb in self.wbs:
            wb.alpha = 0
            wb.X_a = [0,0,0]
            wb.O_flip = 1
        self.cell_enum = np.arange(13)

    def get_module_misfits(self):
        """Evaluate the misfit along the inter-module transitions 
        that were not explicitly plugged but their compatibility comes
        as a result of imprecisions
        """
        misfits_ = []
        for wb_link, diag, dir in zip(*self.get_slit_seq(self.misfit_pi)):
            wb_fixed, wb_plugged = self.wbs[wb_link]
            m1 = wb_plugged.get_misfit(wb_fixed, diag, dir)
            misfits_.append(m1)

        misfits = np.array(misfits_)
        return misfits

    def plot_modules_3D(self, plot, module_numbers=True, facet_numbers=False):
        for i, wb in enumerate(self.wbs):
            color = self.colors[i % len(self.colors)]
            wb.plot_points(plot, wb.G_centroids_Fa[[14,15,16,17]], point_size=8, plot_numbers=facet_numbers)
            wb.plot_G_facets(plot, color=color, module_numbers=module_numbers)

    def plot_modules_yz(self, ax):
        for i, wb in enumerate(self.wbs):
            G_crease_lines_X_aLi = np.einsum('Lia->aiL', wb.G_crease_lines_X_Lia)
            ax.plot(*G_crease_lines_X_aLi[1:,...], color='black')
        ax.set_aspect('equal')

    def plot_modules_xz(self, ax):
        for i, wb in enumerate(self, self.wbs):
            G_crease_lines_X_aLi = np.einsum('Lia->aiL', wb.G_crease_lines_X_Lia)
            ax.plot(*G_crease_lines_X_aLi[[0,2],...], color='black')
        ax.set_aspect('equal')

    @staticmethod
    def get_all_permutations():
        """Combine 8 entities of type A and four entities of type B in all possible ordered sequences of 12 entities. While the type A entities go to the first 8 slots, the type B entities go to the last 4 slots. The entities are identified by integers starting from 1 ... 8 for type A and 9 ... 12 for type B. Further, the type A entities are assumed to have either positive or negative sign. However, their combinations are not assumed arbitrary. The signs in the slots [0, 1, 3, 4, 6, 7] are assumed always opposite to the slots [2, 5]. 
        """
        # Create arrays of entities
        entities_A = np.arange(1, 9)  # Entities of type A
        entities_B = np.arange(9, 13)  # Entities of type B

        # Generate all permutations for both types
        perms_A = np.array(list(itertools.permutations(entities_A)))
        perms_B = np.array(list(itertools.permutations(entities_B)))

        # Create sign arrays
        signs_1 = np.array([1, 1, -1, 1, 1, -1, 1, 1])
        signs_2 = -signs_1
        alter_signs = np.array([signs_1, signs_2])

        perms_A_signed = perms_A[:, np.newaxis, :8] * alter_signs[np.newaxis, :, :]
        perms_A = perms_A_signed.reshape(-1, 8)

        # Combine permutations of A and B in all possible sequences
        all_perms = np.array([np.concatenate((pa, pb)) 
                              for pa in perms_A for
                              pb in perms_B])
        return all_perms

    def flip_modules(self, flip_map):
        """Based on the flip_map rotate the specified cells 
        around the z-axes. This is taken care of by the WBCellScan objects
        """
        for wb, O_flip in zip(self.wbs, flip_map):
            wb.O_flip = O_flip

    combination_choice = tr.Property(depends_on='modules, sample_size')
    @tr.cached_property
    def _get_combination_choice(self):
        combination_choice = np.random.choice(
            self.all_perms.shape[0], self.sample_size, replace=False)

    misfit_of_all_combinations = tr.Property(depends_on='modules, sample_size')
    @tr.cached_property
    def _get_misfit_of_all_combinations(self):
        """Out of all possible combinations, randomly select combinations 
        of the specified sample size and evaluate the misfits
        """
        all_perms = self.get_all_permutations()
        misfit_tC_ = []
        for wb_perm in all_perms[self.combination_choice]:
            O_flip_map = np.sign(wb_perm)
            self.cell_enum = np.abs(wb_perm) - 1
            self.flip_modules(O_flip_map)
            self.plug_modules()
            misfit_C = self.get_module_misfits()
            misfit_tC_.append(misfit_C)
            self.reset_modules()
        misfit_tC = np.array(misfit_tC_)
        return misfit_tC

    best_combination_index = tr.Property(depends_on='modules, sample_size')
    @tr.cached_property
    def _get_best_combination_index(self):
        misfit_tC = self.misfit_of_all_combinations
        return np.argmin(np.max(np.fabs(misfit_tC), axis=(1,2)))

    misfit_of_best_combination = tr.Property(depends_on='modules, sample_size')
    @tr.cached_property
    def _get_misfit_of_best_combination(self):
        return self.misfit_of_all_combinations[self.best_combination_index]

    def activate_best_combination(self):
        self.reset_modules()
        best_combination_enum = self.all_perms[
            self.combination_choice[self.best_combination_index]]
        O_flip_map = np.sign(best_combination_enum)
        self.cell_enum = np.abs(self.combination_enum)
        self.flip_modules(O_flip_map)
        self.plug_modules()

    def plot_best_combination_3D(self, plot):
        self.activate_best_combination()
        self.plot_modules_3D(plot, module_numbers=False, facet_numbers=False)
        self.reset_modules()

    def plot_best_combination_yz(self, ax):
        self.activate_best_combination()
        self.plot_modules_yz(ax)
        self.reset_modules()

    module_x_rows = tr.List([[0, 1], [2, 8, 10], [3, 4], [5, 9, 11], [6, 7]])
    def get_support_geometry(self):
        self.activate_best_combination()
        rows = self.module_x_rows
        YZ_rows_, alpha_rows_ = [], []
        for row in rows:
            YZ_row = []
            alpha_row = []
            cells = self.wbs[row]
            for cell in cells:
                YZ_row.append(cell.X_a[1:])
                alpha_row.append(cell.alpha)
            YZ_rows_.append( np.average(np.array(YZ_row, dtype=np.float_), axis=0 ) )
            alpha_rows_.append( np.average(np.array(alpha_row, dtype=np.float_), axis=0 ) )
        return np.array(YZ_rows_, dtype=np.float_), np.array(alpha_rows_, dtype=np.float_)                

    @staticmethod
    def plot_support_vshape(ax, X, D, H, W, G, alpha, 
                            gamma, c, dist_dimline):
        self.activate_best_combination()
        # Calculate coordinates of the V-shape
        x_middle = X
        W2 = W / 2
        G2 = G / 2
        x_left = X - W2
        x_mid_left = X - G2
        x_right = X + W2
        x_mid_right = X + G2
        y_middle = H + D
        y_top_left = y_middle + np.tan(np.pi/2-gamma/2+alpha) * W2
        y_top_right = y_middle + np.tan(np.pi/2-gamma/2-alpha) * W2
        y_mid_left = y_middle + np.tan(np.pi/2-gamma/2+alpha) * G2
        y_mid_right = y_middle + np.tan(np.pi/2-gamma/2-alpha) * G2

        x = [x_left, x_mid_left, x_mid_left, x_left, x_left]
        y = [D, D, y_mid_left, y_top_left, D]
        ax.fill(x, y, c)
        ax.plot(x, y, color='black')

        x = [x_mid_right, x_right, x_right, x_mid_right, x_mid_right]
        y = [D, D, y_top_right, y_mid_right, D]
        ax.fill(x, y, c)
        ax.plot(x, y, color='black')

        # Create dimension lines
        ax.annotate("", xy=(x_left-dist_dimline, D), xytext=(x_left-dist_dimline, y_top_left), arrowprops=dict(arrowstyle='<->'))
        ax.text(x_left-dist_dimline, (y_top_left+D)/2, f"{y_top_left-D:.0f}", ha='right', va='center', rotation=90)

        ax.annotate("", xy=(x_right+dist_dimline, D), xytext=(x_right+dist_dimline, y_top_right), arrowprops=dict(arrowstyle='<->'))
        ax.text(x_right+dist_dimline, (y_top_right+D)/2, f"{y_top_right-D:.0f}", ha='left', va='center', rotation=90)

        ax.annotate("", xy=(x_middle, D), xytext=(x_middle, y_middle), arrowprops=dict(arrowstyle='<->'))
        ax.text(x_middle+dist_dimline/2, (y_middle+D)/2, f"{y_middle-D:.0f}", ha='center', va='center', rotation=90)

        ax.annotate("", xy=(x_middle, 0), xytext=(x_middle, D), arrowprops=dict(arrowstyle='<->'))
        ax.text(x_middle-dist_dimline, D/2, f"{D:.0f}", ha='center', va='center', rotation=90)

        ax.annotate("", xy=(x_left, D+dist_dimline), xytext=(x_right, D+dist_dimline), 
                    arrowprops=dict(arrowstyle='<->'))
        ax.text(x_left, D+dist_dimline, f"{W:.0f}", ha='left', va='bottom', rotation=0)

        # Plot horizontal line
        ax.plot([x_middle-W/4, x_middle+W/4], [y_middle, y_middle], color='black')
        
        # Plot angles
        angle_left = np.degrees(np.pi/2-gamma/2+alpha)
        angle_right = np.degrees(np.pi/2-gamma/2-alpha)
        ax.add_patch(Arc((x_middle, y_middle), W/4, W/4, theta1=0, theta2=angle_left, 
                        edgecolor='black', linestyle='dashed'))
        ax.add_patch(Arc((x_middle, y_middle), W/4, W/4, theta1=180-angle_right, theta2=180,  
                        edgecolor='black', linestyle='dashed'))
        
        ax.text(x_middle+W/8, y_middle+30, f"{angle_right:.1f}$^\circ$", va='bottom', ha='left')
        ax.text(x_middle-W/8, y_middle+30, f"{angle_left:.1f}$^\circ$", va='bottom', ha='right')

        ax.set_aspect('equal')

        # switch off the axes
        ax.axis('off')

    def get_gamma_support(self):
        """Get the average dihedral angle between the supported middle facets    
        """
        self.activate_best_combination()
        gammas_ = []
        for wb in self.wbs:
            n1, n2 = wb.normals_Fa[[3,10]]
            gamma_ = np.pi - np.arccos(np.einsum('i,i->', n1, n2))
            gammas_.append(gamma_)
        gamma = np.average(np.array(gammas_))
        return gamma
    
    def get_bounding_box(self):
        """Get the deepest node of the supplied cells
        """
        self.activate_best_combination()
        X_FNa_ = []
        for wb in self.wbs:
            X_Na = wb.G_crease_nodes_X_Na
            X_FNa_.append(X_Na)
        X_min = np.min(np.array(X_FNa_), axis=(0,1))
        X_max = np.max(np.array(X_FNa_), axis=(0,1))
        return X_min, X_max
    
    D = tr.Float(136) # Distance of the trolley height from the floor

    def plot_longitudinal_supports(self, ax):
        self.activate_best_combination()
        D = self.D

        wb_X_Ca, wb_alpha_C = self.get_support_geometry()

        min_y = np.min(wb_X_Ca[:,1])
        base_height = 78
        y_shift = base_height - min_y

        y_values = [wb.X_a[2] for wb in self.wbs]

        gamma = self.get_gamma_support()
        for X_Ca, alpha_C in zip(wb_X_Ca, wb_alpha_C):
            self.plot_support_vshape(ax, X=X_Ca[0], D=D, H=X_Ca[1]+y_shift, W=300, G=0, alpha=alpha_C, gamma=gamma, c='lightgray', dist_dimline=-50)

        for wb, y_value in zip(self.wbs, y_values):
            X_a = np.copy(wb.X_a)
            X_a[2] += D + y_shift
            wb.X_a = X_a

        show_modules_yz = False
        if show_modules_yz:
            self.plot_modules_yz(ax)

        X_min, X_max = self.get_bounding_box(self.wbs[[6,7]])
        y_arg = X_min[2]
        x_max = X_max[1]
        ax.annotate("", xy=(x_max, 0), xytext=(x_max, y_arg), arrowprops=dict(arrowstyle='<->'))
        ax.text(x_max-50, (y_arg)/2, f"{y_arg:.0f}", ha='center', va='center', rotation=90)
        ax.plot([x_max-100, x_max+100], [y_arg, y_arg], linestyle='dashed', color='black')

        X_min, X_max = self.get_bounding_box(self.wbs[[0,1]])
        y_arg = X_min[2]
        x_min = X_min[1]
        ax.annotate("", xy=(x_min, 0), xytext=(x_min, y_arg), arrowprops=dict(arrowstyle='<->'))
        ax.text(x_min-50, (y_arg)/2, f"{y_arg:.0f}", ha='center', va='center', rotation=90)
        ax.plot([x_min-100, x_min+100], [y_arg, y_arg], linestyle='dashed', color='black')

        # Plot floor and its annotation
        ax.plot([x_min, x_max], [0, 0], color='black')
        ax.text((x_min+x_max)/2, 0, "floor", ha='center', va='top')

        for wb, y_value in zip(self.wbs, y_values):
            X_a = np.copy(wb.X_a)
            X_a[2] = y_value
            wb.X_a = X_a

    def plot_transverse_supports(self, ax):
        D = self.D
        delta_x = 1200
        z_gap = 15
        self.activate_best_combination()
        wb_X_Ca, wb_alpha_C = self.get_support_geometry(self.wbs)

        min_y = np.min(wb_X_Ca[:,1])
        base_height = 78
        y_shift = base_height - min_y

        x_values = [wb.X_a[0] for wb in self.wbs]
        y_values = [wb.X_a[2] for wb in self.wbs]

        for wb, y_value in zip(self.wbs, y_values):
            X_a = np.copy(wb.X_a)
            X_a[2] += D + y_shift
            wb.X_a = X_a

        wb_row_map = [[0, 1], [2, 8, 10], [3, 4], [5, 9, 11], [6, 7]]
        for i, row in enumerate(wb_row_map):
            x_shift = i * delta_x
            cells = self.wbs[row]

            for cell in cells:
                X_a = np.copy(cell.X_a)
                X_a[0] = x_shift
                cell.X_a = X_a
                X_Lia = cell.G_crease_lines_X_Lia[[17,18]]
                z_mountain = np.average(X_Lia[...,2]) - z_gap
            self.plot_modules_xz(cells, ax)
            ax.plot([x_shift-300, x_shift-300],[D,z_mountain])
            ax.annotate("", xy=(x_shift-550, D), xytext=(x_shift-550, z_mountain), arrowprops=dict(arrowstyle='<->'))
            ax.text(x_shift-550-50, D+z_mountain / 2, f"{z_mountain-D:.0f}", ha='center', va='center', rotation=90)

        for wb, x_value in zip(self.wbs, x_values):
            X_a = np.copy(wb.X_a)
            X_a[0] = x_value
            wb.X_a = X_a

        ax.annotate("", xy=(0, 0), xytext=(0, D), arrowprops=dict(arrowstyle='<->'))
        ax.text(50, D/2, f"{D:.0f}", ha='center', va='center', rotation=90)

        X_min, X_max = self.get_bounding_box(self.wbs[[6,7]])
        y_arg = X_min[2]
        x_max = (delta_x * 5 - 400)
        ax.annotate("", xy=(x_max, 0), xytext=(x_max, y_arg), arrowprops=dict(arrowstyle='<->'))
        ax.text(x_max-50, (y_arg)/2, f"{y_arg:.0f}", ha='center', va='center', rotation=90)
        ax.plot([x_max-100, x_max+100], [y_arg, y_arg], linestyle='dashed', color='black')

        X_min, X_max = self.get_bounding_box(self.wbs[[0,1]])
        y_arg = X_min[2]
        x_min = -1000
        ax.annotate("", xy=(x_min, 0), xytext=(x_min, y_arg), arrowprops=dict(arrowstyle='<->'))
        ax.text(x_min-50, (y_arg)/2, f"{y_arg:.0f}", ha='center', va='center', rotation=90)
        ax.plot([x_min-100, x_min+100], [y_arg, y_arg], linestyle='dashed', color='black')

        # Plot floor and its annotation
        ax.plot([x_min, x_max], [D, D], linestyle='dashed', color='black')

        # Plot floor and its annotation
        ax.plot([x_min, x_max], [0, 0], color='black')
        ax.text((x_min+x_max)/2, 0, "floor", ha='center', va='top')

        ax.set_aspect('equal')
        ax.axis('off');