import numpy as np
import scipy
import scipy.optimize
import k3d
from traits.api import HasTraits, List, Array, \
    Str, Property, cached_property, Bool, Float, Int

import os
import pickle
import sympy as sp

def cache_solve(expr, symbols, name, recalculate=False, simplify=False):
    filename = name + '.pkl'
    if os.path.exists(filename) and not recalculate:
        # Load the solution from the file
        with open(filename, 'rb') as f:
            solution = pickle.load(f)
    else:
        # Compute the solution
        solution = sp.solve(expr, symbols)
        
        # Simplify the solution if requested
        if simplify:
            if isinstance(solution, list):
                solution = [sp.simplify(sol) for sol in solution]
            elif isinstance(solution, dict):
                solution = {k: sp.simplify(v) for k, v in solution.items()}
            else:
                solution = sp.simplify(solution)
        
        # Save the solution to a file
        with open(filename, 'wb') as f:
            pickle.dump(solution, f)
    
    return solution

# Define sp.symbols
alpha_C, d_alpha_D = sp.symbols('alpha_C d_alpha_D')
alpha_D = alpha_C + d_alpha_D

X_C = sp.Matrix(sp.symbols('X_C1, X_C2'))
X_D = sp.Matrix(sp.symbols('X_D1, X_D2'))
x_Cb = sp.Matrix(sp.symbols('x_Cb1, x_Cb2'))
x_Ct = sp.Matrix(sp.symbols('x_Ct1, x_Ct2'))
x_Db = sp.Matrix(sp.symbols('x_Db1, x_Db2'))
x_Dt = sp.Matrix(sp.symbols('x_Dt1, x_Dt2'))
n_Cb = sp.Matrix(sp.symbols('n_Cb1, n_Cb2'))
n_Ct = sp.Matrix(sp.symbols('n_Ct1, n_Ct2'))
n_Db = sp.Matrix(sp.symbols('n_Db1, n_Db2'))
n_Dt = sp.Matrix(sp.symbols('n_Dt1, n_Dt2'))

sp_vars = (alpha_C, X_C, x_Cb, x_Ct, x_Db, x_Dt, n_Cb, n_Ct, n_Db, n_Dt)

X_C1, X_C2 = X_C
X_D1, X_D2 = X_D

X_C0 = sp.symbols('X_C0')
X_D0 = sp.symbols('X_D0')
X3_C = sp.Matrix([X_C0, X_C1, X_C2])
X3_D = sp.Matrix([X_D0, X_D1, X_D2])

# Rotation matrices
T_C = sp.Matrix([[sp.cos(alpha_C), -sp.sin(alpha_C)], [sp.sin(alpha_C), sp.cos(alpha_C)]]).T
T_D = sp.Matrix([[sp.cos(alpha_D), -sp.sin(alpha_D)], [sp.sin(alpha_D), sp.cos(alpha_D)]]).T

n_Cb_unit = n_Cb / n_Cb.norm()
n_Dt_unit = n_Dt / n_Dt.norm()

# Calculate dot product of n_Cb_global and n_Db_global
dot_product = n_Cb_unit.dot(n_Dt_unit)

# Check if dot product is equal to product of magnitudes
d_alpha_D_solved = sp.simplify(sp.acos(dot_product))

# Criterion 2: The point X_D1 must be on the line running through X_C2 orthogonal to n_Ct
X_Ct = X_C + T_C * x_Ct
X_Db = X_D + T_D * x_Db
V_C2 = X_Ct - X_Db 
N_C2 = T_C * n_Ct
misfit_Ct = sp.simplify(V_C2.dot(N_C2))
criterion2 = sp.Eq( misfit_Ct, 0)

# Criterion 3: The point X_Dt must be on the line given by the point X_Cb and 
# a line perpendicular to N_Cb
X_Cb = X_C + T_C * x_Cb
X_Dt = X_D + T_D * x_Dt
V_C1 = X_Dt - X_Cb
N_C1 = T_C * n_Cb
misfit_Cb = sp.simplify(V_C1.dot(N_C1))
criterion3 = sp.Eq( misfit_Cb, 0)

# Solve the system of equations
X_D_solved = cache_solve((criterion2, criterion3), [X_D1, X_D2], "X_D_solved") # , recalculate=True, simplify=True)
X_D_solved

get_d_alpha_D = sp.lambdify((alpha_C, X_C, n_Cb, n_Dt), d_alpha_D_solved)
get_X_D1 = sp.lambdify(sp_vars + (d_alpha_D,), X_D_solved[X_D1])
get_X_D2 = sp.lambdify(sp_vars + (d_alpha_D,), X_D_solved[X_D2])
get_misfit_Ct = sp.lambdify(sp_vars + (d_alpha_D, X_D), misfit_Ct)
get_misfit_Cb = sp.lambdify(sp_vars + (d_alpha_D, X_D), misfit_Cb)


class WBScannedCell(HasTraits):
    # Inputs
    file_path = Str()
    label = Str('noname')
    '''
    This follows the numbering of the facets in the .obj file
    This is the list of the indices of the facets that are neighboured.
    Remember that the facets are enumerated counter-clockwise starting with the upper right facet.
    The index 0 in Python means index 1 on the image from 01_wb_cell_scan_analyzer.ipynb
    '''
    F_Cf = Array(dtype=np.uint32,
                 value=[[0,1], [1,2], [2,3], [3,4], [4,5], 
                         [5,6], [6,7], [7,8], [8,9], [9,10], 
                         [10,11], [11,12], [12,13], [13,0],
                         [3,10],[4,9],[2,11]])
    
    # Node index from the intersecting crease lines
    isc_N_L = List([[0, 13, 12],
                    [5, 6, 7],
                    [1, 16, 11],
                    [4, 15, 8],
                    [14, 16, 2, 10],
                    [14, 15, 3, 9],[1,2],[3,4],[8,9],[10,11]])
    
    # The numbers inside the list are the index of the points that form the crease lines
    icrease_lines_N_Li = Array(dtype=np.uint32,
                              value=[[0,2],[1,3],[4,2],[5,3],[2,6],
                                     [4,6],[2,9],[4,9],[3,7],[5,7],
                                     [3,8],[5,8],[4,5],[7,6],[8,9],
                                     [4,0],[5,1]])
    
    sym_Si = Array(dtype=np.uint32, 
                   value=[[2,3],[15,16],[5,9],[7,11],[4,8],[6,10]])
    
    bot_contact_planes_F = Array(dtype=np.uint32,
                                 value=[0,  6,  7, 13])
    
    top_contact_planes_Gi = Array(dtype=np.uint32,
                                  value=[[14, 15],[17, 16]])
    
    bcrease_lines_N_Li = Array(dtype=np.uint32,
                               value=[[0,14],[1,15],[0,10],[1,11],[1,12],[0,13],
                                      [6,10],[7,11],[8,12],[9,13],[10,16],[11,17],
                                      [12,18],[13,19],[14,16],[15,17],[15,18],[14,19]])
        
    facets_N_F = Array(dtype=np.uint32,
        value=[[0,14,16], [0, 16, 10], [0, 10, 6], [0, 6, 2],
        [1, 3, 7], [1,7,11], [1, 11, 17],[1,17,15],
        [1,15,18], [1,18,12], [1,12,8], [1,8,3],
        [0,2,9],[0,9,13],[0,13,19],[0,19,14],
        [2,6,4], [3,5,7],[3,8,5],[2,4,9],
        [4,6,5],[5,6,7],[5,8,4],[4,8,9]
    ])

    X_a = Array(value=[0, 0, 0], dtype=float)
    alpha = Float(0.0)

    # Interim results
    #flip_vertically = Bool(False)
    rotate_system = List([])

    wb_scan_X_Fia = Property(Array, depends_on='file_path')
    planes_Fi = Property(Array, depends_on='file_path')
    normals_Fa = Property(Array, depends_on='file_path')
    centroids_Fa = Property(Array, depends_on='file_path')
    isc_points_Li = Property(Array, depends_on='file_path')
    isc_vectors_Li = Property(Array, depends_on='file_path')
    icrease_nodes_X_Na = Property(Array, depends_on='file_path')
    icrease_lines_X_Lia = Property(Array, depends_on='file_path')
    start_icrease_lines_La = Property(Array, depends_on='file_path')
    vectors_icrease_lines_La = Property(Array, depends_on='file_path')
    lengths_icrease_lines_L = Property(Array, depends_on='file_path')   
    sym_crease_length_diff_S = Property(Array, depends_on='file_path')
    sym_crease_angles_S = Property(Array, depends_on='file_path')

    # Local coordinate system
    O_flip = Int(1) # 1 stay, -1 flip
    O_centroids_flip = Array(value=[7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 16, 17, 14, 15], 
                            dtype=np.int_)
    O_crease_nodes_flip = Array(value=[1, 0, 3, 2, 5, 4, 8, 9, 6, 7, 12, 13, 10, 11, 15, 14, 18, 19, 16, 17], 
                                dtype=np.int_)

    O_basis_ab = Property(Array, depends_on='file_path')
    O_icrease_nodes_X_Na = Property(Array, depends_on='file_path')
    O_icrease_lines_X_Lia = Property(Array, depends_on='file_path')
    O_normals_Fa = Property(Array, depends_on='file_path, O_flip')
    O_centroids_Fa = Property(Array, depends_on='file_path, O_flip')
    O_isc_points_Li = Property(Array, depends_on='file_path')
    O_isc_vectors_Li = Property(Array, depends_on='file_path')
    O_crease_nodes_X_Na = Property(Array, depends_on='file_path, O_flip')
    O_crease_lines_X_Lia = Property(Array, depends_on='file_path')
    O_wb_scan_X_Fia = Property(Array, depends_on='file_path')
    O_thickness_Fi = Property(Array, depends_on='file_path')

    # Global coordinate system
    G_crease_nodes_X_Na = Property(Array, depends_on='file_path, X_a, X_a_items, alpha')
    G_centroids_Fa = Property(Array, depends_on='file_path, X_a, X_a_items, alpha')
    G_crease_nodes_X_Na = Property(Array, depends_on='file_path, X_a, X_a_items, alpha')
    G_crease_lines_X_Lia = Property(Array, depends_on='file_path, X_a, X_a_items, alpha')
    
    # Call the obj_file_points_to_numpy method to read the .obj file and return the points
    @cached_property
    def _get_wb_scan_X_Fia(self):
        X_Fia = self.obj_file_points_to_numpy(self.file_path)        
        if len(self.rotate_system) > 0:
            axes, angles = self.rotate_system
            X_Fia = [self.rotate_3d(X_ia, axes, angles) for X_ia in X_Fia]
        return X_Fia

    # Return an array with the coefficients a,b,c,d of the best fit planes for each facet of the waterbomb cell
    @cached_property
    def _get_planes_Fi(self):
        return np.array([self.best_fit_plane(X_ia) for X_ia in self.wb_scan_X_Fia],
                        dtype=np.float32)

    @cached_property
    def _get_normals_Fa(self):
        return self.planes_Fi[:,:-1]

    # Return an array with the coordinates of the centroids of the planes of the waterbomb cell
    @cached_property
    def _get_centroids_Fa(self):
        return np.array([np.mean(X_Ia, axis=0) for X_Ia in self.wb_scan_X_Fia],
                        dtype=np.float32)

    @cached_property
    def _get_isc_points_Li(self):
        return self.intersection_lines(self.F_Cf, self.planes_Fi, self.centroids_Fa)[0]
    
    @cached_property
    def _get_isc_vectors_Li(self):
        return self.intersection_lines(self.F_Cf, self.planes_Fi, self.centroids_Fa)[1]

    @cached_property
    def _get_icrease_nodes_X_Na(self):
        return self.centroid_of_intersection_points(self.isc_points_Li, self.isc_vectors_Li, self.isc_N_L)[0]

    @cached_property
    def _get_icrease_lines_X_Lia(self):
        return self.icrease_nodes_X_Na[self.icrease_lines_N_Li]

    @cached_property
    def _get_start_icrease_lines_La(self):
        return self.icrease_lines_X_Lia[:,0,:]

    @cached_property
    def _get_vectors_icrease_lines_La(self):
        return self.icrease_lines_X_Lia[:,1,:] - self.start_icrease_lines_La

    @cached_property
    def _get_lengths_icrease_lines_L(self):
        return np.linalg.norm(self.vectors_icrease_lines_La, axis=1)

    @cached_property
    def _get_sym_crease_length_diff_S(self):
        return (self.lengths_icrease_lines_L[self.sym_Si[:,1]] - 
                self.lengths_icrease_lines_L[self.sym_Si[:,0]])

    @cached_property
    def _get_sym_crease_angles_S(self):
        return self.angle_between_lines(
            self.icrease_lines_X_Lia[self.sym_Si[:,0]],
            self.icrease_lines_X_Lia[self.sym_Si[:,1]]
        )
    
    @cached_property
    def _get_O_basis_ab(self):
        """Derive the basis of the waterbomb cell.
        """
        Or, Ol = 4, 5 # left and right crease node on the center line 
        Fu, Fl = 3, 10 # upper and lower facets rotating around Or-Ol line
        
        # Mid point between the two crease nodes
        O_a = (self.icrease_nodes_X_Na[Or] + 
                   self.icrease_nodes_X_Na[Ol]) / 2
        
        # Vector connecting the two crease nodes -> Defines the vector 
        vec_Ox_a = self.icrease_nodes_X_Na[Or] - self.icrease_nodes_X_Na[Ol]
        
        # Normalized vector connecting the two crease nodes
        nvec_Ox_a = vec_Ox_a / np.linalg.norm(vec_Ox_a)
        
        # Averaged normal vector to the lower and upper facets
        _vec_Oz_a = (self.normals_Fa[Fu] + self.normals_Fa[Fl]) / 2
        
        # Normalized Averaged normal vector to the lower and upper facets
        _nvec_Oz_a = _vec_Oz_a / np.linalg.norm(_vec_Oz_a)
        
        # Calculate the basis vector e_y = e_z x e_x perpendicular to the plane defined by e_x and e_z
        nvec_Oy_a = np.cross(_nvec_Oz_a, nvec_Ox_a)
        
        # Recalculate the basis vector e_z = e_x x e_y to assure orthogonality
        nvec_Oz_a = np.cross(nvec_Ox_a, nvec_Oy_a)
        
        O_basis_ab = np.array([nvec_Ox_a, nvec_Oy_a, nvec_Oz_a], dtype=np.float32)
        return O_a, O_basis_ab

    # Transform crease nodes from the global to the local coordinate system using the basis O_basis_ab
    @cached_property
    def _get_O_icrease_nodes_X_Na(self):
        O_a, O_basis_ab = self.O_basis_ab
        O_icrease_nodes_X_Na = self.transform_to_local_coordinates(
            self.icrease_nodes_X_Na, O_a, O_basis_ab
        )
        return O_icrease_nodes_X_Na

    @cached_property
    def _get_O_icrease_lines_X_Lia(self):
        return self.O_icrease_nodes_X_Na[self.icrease_lines_N_Li]
    
    @cached_property
    def _get_O_normals_Fa(self):
        _, O_basis_ab = self.O_basis_ab
        O_normals_Fa = self.transform_to_local_coordinates(
            self.normals_Fa, np.array([0,0,0]), O_basis_ab
        )
        if self.O_flip < 0:
            T_ab = np.array([
                [np.cos(np.pi), np.sin(np.pi), 0 ],
                [-np.sin(np.pi), np.cos(np.pi), 0],
                [0, 0, 1]], dtype=np.float_)
            O_normals_Fa = np.einsum('ab,...a->...b', T_ab, O_normals_Fa)[self.O_centroids_flip]
        return O_normals_Fa

    @cached_property
    def _get_O_centroids_Fa(self):
        O_a, O_basis_ab = self.O_basis_ab
        O_centroids_Fa = self.transform_to_local_coordinates(
            self.centroids_Fa, O_a, O_basis_ab
        )
        if self.O_flip < 0:
            T_ab = np.array([
                [np.cos(np.pi), np.sin(np.pi), 0 ],
                [-np.sin(np.pi), np.cos(np.pi), 0],
                [0, 0, 1]], dtype=np.float_)
            O_centroids_Fa = np.einsum('ab,...a->...b', T_ab, O_centroids_Fa)[self.O_centroids_flip]
        return O_centroids_Fa

    @cached_property
    def _get_O_isc_points_Li(self):
        O_a, O_basis_ab = self.O_basis_ab
        return self.transform_to_local_coordinates(
            self.isc_points_Li, O_a, O_basis_ab
        )

    @cached_property
    def _get_O_isc_vectors_Li(self):
        """This code first calculates the vector from the origin to each line_point and then 
        calculates the dot product between these vectors and their corresponding line_vectors 
        using np.einsum. It then creates a mask of where the dot product is negative, 
        indicating that the line_vector is pointing inwards. Finally, it reverses the direction of 
        these line_vectors by multiplying them by -1.
        """
        _, O_basis_ab = self.O_basis_ab

        O_isc_vectors_La = self.transform_to_local_coordinates(
            self.isc_vectors_Li, np.array([0,0,0]), O_basis_ab
        )

        # Calculate dot product using einsum
        dot_product_L = np.einsum('La,La->L', self.O_isc_points_Li, 
                                     O_isc_vectors_La)
        
        # Find where dot product is negative
        mask = dot_product_L < 0

        # Reverse direction of line_vector where dot product is negative
        O_isc_vectors_La[mask] *= -1

        return O_isc_vectors_La
    
    @staticmethod
    def rotate_3d(points, axes, angles):
        def rotation_matrix(axis, angle):
            a = np.cos(angle / 2)
            b, c, d = -axis * np.sin(angle / 2)
            return np.array([
                [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]
            ])

        for ax, ang in zip(axes, angles):
            axis_vector = np.zeros((3,), dtype=np.float_)
            axis_vector[ax] = 1
            R = rotation_matrix(axis_vector, ang)
            points = np.einsum('ij,...j->...i', R, points)

        return points

    # Define the crease nodes in the local coordinate system
    @cached_property
    def _get_O_crease_nodes_X_Na(self):
        
        # Calculate average length of the creases in the valley with indices 5, 9, 11, 7     
        # These are the valley creases that form the base cell   
        length_valley = np.average(self.lengths_icrease_lines_L[[5,9,11,7]])
        
        # Coordinates of the intersecting points of the valley creases in the local coordinate system
        valley_Ca = self.O_icrease_nodes_X_Na[[2,3,3,2]]
        
        # Vector of the intersecting points of the valley creases in the local coordinate system
        vec_valley_Ca = self.O_isc_vectors_Li[[0,5,7,12]]
        
        # Calculate the coordinates of the valley crease nodes
        valley_node_X_Ca = valley_Ca + vec_valley_Ca * length_valley

        # Calculate average length of the creases in the mountain with indices 13, 14
        # These are the mountain creases that form the base cell
        length_mountain = np.average(self.lengths_icrease_lines_L[[13,14]])
        
        # Coordinates of the intersecting points of the mountain creases in the local coordinate system
        mountain_Ca = self.O_icrease_nodes_X_Na[[0,1]]
        
        # Vector of the intersecting points of the mountain creases in the local coordinate system
        vec_mountain_Ca = self.O_isc_vectors_Li[[13,6]]
        
        # Calculate the coordinates of the mountain crease nodes
        mountain_node_X_Ca = mountain_Ca + vec_mountain_Ca * length_mountain / 2

        # The valley nodes are the four nodes of the corner nodes of the base cell
        corner_node_X_Ca = np.copy(valley_node_X_Ca)
        
        # The first and the last node of the corner crease nodes are the same
        corner_node_X_Ca[:,0] = mountain_node_X_Ca[[0,1,1,0],0] 

        # Stack the valley, mountain and corner nodes to get the crease nodes of the base cell
        O_bcrease_nodes_X_Ca = np.vstack([valley_node_X_Ca, mountain_node_X_Ca, corner_node_X_Ca])
        
        # Flip the crease nodes if the flip attribute is set to -1
        O_crease_nodes_C_Ca = np.vstack([self.O_icrease_nodes_X_Na, O_bcrease_nodes_X_Ca])
        if self.O_flip < 0:
            T_ab = np.array([
                [np.cos(np.pi), np.sin(np.pi), 0 ],
                [-np.sin(np.pi), np.cos(np.pi), 0],
                [0, 0, 1]], dtype=np.float_)
            O_crease_nodes_C_Ca = np.einsum('ab,...a->...b', T_ab, O_crease_nodes_C_Ca)[self.O_crease_nodes_flip]

        return O_crease_nodes_C_Ca

    @cached_property
    def _get_G_crease_nodes_X_Na(self):
        """Global coordinates of the derived crease pattern nodes
        """
        return self.transform_O_to_G(self.O_crease_nodes_X_Na)
    
    @cached_property
    def _get_G_crease_lines_X_Lia(self):
        """Global coordinates of the derived crease pattern nodes
        """
        crease_lines_N_Li = np.vstack([self.icrease_lines_N_Li, self.bcrease_lines_N_Li])
        return self.G_crease_nodes_X_Na[crease_lines_N_Li]
    
    @cached_property
    def _get_G_centroids_Fa(self):
        """Global coordinates of the centroids of the facets of the derived crease pattern
        """
        return self.transform_O_to_G(self.O_centroids_Fa)

    def transform_O_to_G(self, O_points_X_Na):
        """Transforms points from the local coordinate system O to the global coordinate system G.
        """
        # Rotate points around the x-axis by alpha
        alpha = self.alpha
        T_ab = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)]
        ], dtype=np.float_)
        G_points_X_Na = np.einsum(
            'ab,...a->...b', T_ab, O_points_X_Na
        )
        return G_points_X_Na + self.X_a[np.newaxis,:]

    @cached_property
    def _get_O_crease_lines_X_Lia(self):
        crease_lines_N_Li = np.vstack([self.icrease_lines_N_Li, self.bcrease_lines_N_Li])
        return self.O_crease_nodes_X_Na[crease_lines_N_Li]
    
    @cached_property
    def _get_O_wb_scan_X_Fia(self):
        O_a, O_basis_ab = self.O_basis_ab
        return [self.transform_to_local_coordinates(wb_scan_X_ia, O_a, O_basis_ab) 
                for wb_scan_X_ia in self.wb_scan_X_Fia]
        
    @cached_property
    def _get_O_thickness_Fi(self):
        centroids_X_Fa = self.O_centroids_Fa[self.bot_contact_planes_F]
        vectors_X_Fa = self.O_normals_Fa[self.bot_contact_planes_F]
        centroids_X_Fia = self.O_centroids_Fa[self.top_contact_planes_Gi]
        return self.project_points_on_planes(centroids_X_Fa, vectors_X_Fa, 
                                             centroids_X_Fia)


    corner_map = Array(value=[[[17, 10],[15, 3], ], [[16, 10],[14, 3]]], dtype=np.int_)
    diag_corner_map = Array(value=[[(0, 0), (0, 1)], [(1, 0), (1, 1)]], dtype=np.int_)

    def get_corner_f(self, diag_dir, y_dir):
        """For the specified diagonal, return the index of the two facets relevant 
        for compatibility of neighboring facets.
        """
        diag_index = 0 if diag_dir == -1 else 1
        y_index = 0 if y_dir == -1 else 1
        return self.corner_map[tuple(self.diag_corner_map[diag_index][y_index])]
    
    @staticmethod
    def get_x_dir(diag_dir, y_dir):
        """Given a diagonal direction - either 1 or -1 for positive and 
        negative diagonal meaning - bottom-lef -> top-right and bottom-right -> top-left 
        direction, and the y-direction (top, down) for (positive, negative) return the 
        x-direction (positive, negative) for (right, left). 
        """
        return diag_dir * y_dir

    def plug_into(self, wb_fixed, diag_dir, y_dir, d_X_0=565):
        """Plug the cell into the specified fixed cell wb_fixed 
        along the selected diagonal and y-direction
        """
        side_fixed, side_plugged = np.array([1, -1]) * y_dir
        x_Cfa = wb_fixed.O_centroids_Fa[self.get_corner_f(diag_dir,side_fixed),1:]
        n_Cfa = wb_fixed.O_normals_Fa[self.get_corner_f(diag_dir,side_fixed),1:]
        x_Dfa = self.O_centroids_Fa[self.get_corner_f(diag_dir,side_plugged),1:]
        n_Dfa = self.O_normals_Fa[self.get_corner_f(diag_dir,side_plugged),1:]

        alpha_C_ = wb_fixed.alpha
        X_C_ = wb_fixed.X_a[1:]

        d_alpha_D_n_ = y_dir * get_d_alpha_D(alpha_C_, X_C_, n_Cfa[0], n_Dfa[1])
        X_D1_ = get_X_D1(alpha_C_, X_C_, x_Cfa[0], x_Cfa[1], x_Dfa[0], x_Dfa[1], n_Cfa[0], n_Cfa[1], n_Dfa[0], n_Dfa[1], d_alpha_D_n_)
        X_D2_ = get_X_D2(alpha_C_, X_C_, x_Cfa[0], x_Cfa[1], x_Dfa[0], x_Dfa[1], n_Cfa[0], n_Cfa[1], n_Dfa[0], n_Dfa[1], d_alpha_D_n_)

        X_D0_ = wb_fixed.X_a[0] + d_X_0 * self.get_x_dir(diag_dir, y_dir)
        self.alpha = alpha_C_ + d_alpha_D_n_
        self.X_a = [X_D0_, X_D1_, X_D2_]


    def get_misfit(self, wb_C, diag_dir, y_dir=1):
        """Quantify the misfit with respect to the specified cell that 
        is placed according to the specified diagonal and y direction
        """
        side_fixed, side_plugged = np.array([1, -1]) * y_dir
        x_Cfa = wb_C.O_centroids_Fa[self.get_corner_f(diag_dir,side_fixed),1:]
        n_Cfa = wb_C.O_normals_Fa[self.get_corner_f(diag_dir,side_fixed),1:]
        x_Dfa = self.O_centroids_Fa[self.get_corner_f(diag_dir,side_plugged),1:]
        n_Dfa = self.O_normals_Fa[self.get_corner_f(diag_dir,side_plugged),1:]

        alpha_C_ = wb_C.alpha
        d_alpha_D_ = self.alpha - alpha_C_
        X_C_ = wb_C.X_a[1:]
        X_D_ = self.X_a[1:]

        misfit_Cb = get_misfit_Cb(alpha_C_, X_C_, x_Cfa[0], x_Cfa[1], x_Dfa[0], x_Dfa[1], n_Cfa[0], 
                            n_Cfa[1], n_Dfa[0], n_Dfa[1], d_alpha_D_, X_D_)
        misfit_Ct = get_misfit_Ct(alpha_C_, X_C_, x_Cfa[0], x_Cfa[1], x_Dfa[0], x_Dfa[1], n_Cfa[0], 
                            n_Cfa[1], n_Dfa[0], n_Dfa[1], d_alpha_D_, X_D_)

        return misfit_Cb, misfit_Ct

    def plot_plugged_neighbors_yz(self, wb_plugged, diag_dir, y_dir, ax):
        """Plot the transition between the current cell and the neighbor cell
        specifying the normal vectors at the contact interfaces to verify 
        their orientation. This method is used to check if the index maps 
        specifying the upper and lower surface of the facets with the appropriate
        thickness are addressing the correct parts of the cell based on the 
        specified diag_dir and y_dir attributes. The method was primarily written 
        for debugging.
        """
        side_fixed, side_plugged = np.array([1, -1]) * y_dir

        x_Cfa = self.O_centroids_Fa[self.get_corner_f(diag_dir,side_fixed),1:]
        n_Cfa = self.O_normals_Fa[self.get_corner_f(diag_dir,side_fixed),1:]
        x_Dfa = wb_plugged.O_centroids_Fa[self.get_corner_f(diag_dir,side_plugged),1:]
        n_Dfa = wb_plugged.O_normals_Fa[self.get_corner_f(diag_dir,side_plugged),1:]

        scale = 50
        x2_Cfa = x_Cfa + n_Cfa * scale
        x2_Dfa = x_Dfa + n_Dfa * scale

        nC_alf = np.einsum('lfa->alf', np.array([x_Cfa, x2_Cfa]))
        nD_alf = np.einsum('lfa->alf', np.array([x_Dfa, x2_Dfa]))

        ax.plot(*x_Cfa.T, 'o', color='blue')
        ax.plot(*x_Dfa.T, 'o', color='green')

        ax.plot(*nC_alf[:,:,0], color='orange')
        ax.plot(*nD_alf[:,:,1], color='orange')

        O_crease_lines_X_aLi = np.einsum('Lia->aiL', self.O_crease_lines_X_Lia)
        ax.plot(*O_crease_lines_X_aLi[1:,...], color='blue')
        O_crease_lines_X_aLi = np.einsum('Lia->aiL', wb_plugged.O_crease_lines_X_Lia)
        ax.plot(*O_crease_lines_X_aLi[1:,...], color='green')

        ax.set_aspect('equal')

        
    @staticmethod
    def obj_file_points_to_numpy(file_path):
        """Read the contents of the .obj file and return a list of arrays in with the groups of points associated to individual facets of the waterbomb cell. The facets are enumerated counter-clockwise starting with the upper right facet. 
        """
        facets_points = []
        with open(file_path) as file:
            facet_num = None
            for line in file:
                line = line.strip()
                if line.startswith('o'):
                    if facet_num is not None:
                        facets_points.append({facet_num: facet_points})
                    facet_points = []
                    facet_num = int(line[2:])
                elif line.startswith('v'):
                    facet_points.append(np.array(line[2:].split(' '), dtype=np.float32))
            # append also the last set of points
            facets_points.append({facet_num: facet_points})   
        
        # Sort facets and convert them to a list of lists
        facets_points = sorted(facets_points, key=lambda d: list(d.keys())[0])
        facets_points = [np.array(next(iter(dic.values())), dtype=np.float32) for dic in facets_points]

        return facets_points

    @staticmethod
    def best_fit_plane(X_Ia):
        """Given a list of point coordinates in 3D, i.e. X_Ia array in numpy with two dimensions, the first one is the index of the point and the second dimension is the spatial coordinates, i.e. x,y,z. This method identifies the coefficients of a plane with a best fit between these points.
        """
        # calculate the mean of the points
        centroid = np.mean(X_Ia, axis=0)

        # subtract the mean from the points 
        X_Ia_sub = X_Ia - centroid

        # perform singular value decomposition
        u, s, vh = np.linalg.svd(X_Ia_sub)

        # normal of the plane is the last column in vh
        normal = vh[-1]

        # ensure the normal points upwards
        if normal[2] < 0:
            normal = -normal

        # calculate d in the plane equation ax+by+cz=d
        d = -centroid.dot(normal)
        return np.append(normal, d)

    @staticmethod
    def intersection_lines(F_Cf, planes_Fi, centroids_Fa):
        """
        Given an array of planes expressed using the a,b,c,d coefficients.  identify the intersection lines between all pairs of planes occurring in the input array.
        """
        planes_Cfa = planes_Fi[F_Cf]
        centroids_Cfa = centroids_Fa[F_Cf]
        line_points = []
        line_directions = []
        for plane_pair, centroid_pair in zip(planes_Cfa, centroids_Cfa):
            plane1, plane2 = plane_pair
            line_direction = np.cross(plane1[:3], plane2[:3])
            line_point = np.linalg.solve(
                np.array([plane1[:3], plane2[:3], line_direction]),
                np.array([-plane1[3], -plane2[3], 0])
            )

            # Define function to calculate distance from point on line to centroids
            def f(t, line_point, line_direction, centroid_pair):
                point_t = line_point + t * line_direction
                return np.linalg.norm(point_t - centroid_pair[0]) + np.linalg.norm(point_t - centroid_pair[1])

            # Get the t that minimizes the sum of distances to centroids
            result = scipy.optimize.minimize_scalar(f, args=(line_point, line_direction, centroid_pair))

            # Calculate closest point on the line to the centroids
            closest_point = line_point + result.x * line_direction
            # centroid1, centroid2 = centroid_pair
            # centroid_direction = (centroid1 + centroid2) / 2

            # # Ensure the direction vector points away from the origin
            # if np.dot(line_direction, centroid_direction) < 0:
            #     line_direction = -line_direction

            line_points.append(closest_point)
            line_directions.append(line_direction)


        line_points_La = np.array(line_points, dtype=np.float32)
        line_vecs_La = np.array(line_directions, dtype=np.float32)
        len_line_vecs_La = np.linalg.norm(line_vecs_La, axis=1, keepdims=True)
        return (line_points_La, 
                line_vecs_La / len_line_vecs_La)


    @staticmethod
    def centroid_of_intersection_points(isc_points_L_Xa, isc_vec_L_Xa, isc_N_L):
        """First calculate the intersection points of each pair of lines using the line_intersection function, then calculate the centroid of these intersection points. The centroids for each group of lines are returned as a list.
        
        Args:
            isc_points_L_Xa (array): x, y, z-coordinates of the intersection points.
            isc_vec_L_Xa (array): direction vector of the intersection points. 
            isc_N_L (list): index of points that form a crease line. (It is a numenclature, for details refer to the notebooks on the "intersection nodes" snippet)
        Returns:
            icrease_nodes_X_Na (array): Coordinates of the centroid of intersection points.
        
        To define the intersection point between the crease lines, the least 
        squared method is used to find the points on neighboring creases that minimizes the distance between them.
        This is done for each pair of creases that should intersect and their intersection point is defined as the 
        centroid of the average of the points that minimize the distance between them.

        """
        def closest_point_on_lines(line1, line2):
            p1, v1 = np.array(line1[0]), np.array(line1[1])
            p2, v2 = np.array(line2[0]), np.array(line2[1])
            w = p1 - p2
            a = np.dot(v1, v1)
            b = np.dot(v1, v2)
            c = np.dot(v2, v2)
            d = np.dot(v1, w)
            e = np.dot(v2, w)
            D = a * c - b * b
            sc = (b * e - c * d) / D
            tc = (a * e - b * d) / D
            point_on_line1 = p1 + sc * v1
            point_on_line2 = p2 + tc * v2
            return (point_on_line1 + point_on_line2) / 2  # midpoint

        intersection_point_list = []
        intersection_centroids = []
        for group in isc_N_L:
            intersection_points = []
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    line1 = (isc_points_L_Xa[group[i]], isc_vec_L_Xa[group[i]])
                    line2 = (isc_points_L_Xa[group[j]], isc_vec_L_Xa[group[j]])
                    intersection = closest_point_on_lines(line1, line2)
                    intersection_points.append(intersection)
            centroid = np.mean(intersection_points, axis=0)
            intersection_centroids.append(centroid)
            intersection_point_list.append(np.array(intersection_points))
        return (np.array(intersection_centroids, dtype=np.float32),
                intersection_point_list)

    @staticmethod
    def angle_between_lines(lines1, lines2):
        """This function accepts lines1 and lines2 as Nx2x2 arrays, where N is the number of lines, the first dimension is the two points defining the line, and the second dimension is the coordinates of the points. It returns an array of angles between the corresponding lines in lines1 and lines2.
        """
        # Calculate direction vectors for each line
        direction_vectors1 = np.subtract(lines1[:, 1], lines1[:, 0])
        direction_vectors2 = np.subtract(lines2[:, 1], lines2[:, 0])

        # Normalize the direction vectors
        norm1 = np.sqrt(np.einsum('ij,ij->i', direction_vectors1, direction_vectors1))
        norm2 = np.sqrt(np.einsum('ij,ij->i', direction_vectors2, direction_vectors2))
        direction_vectors1_normalized = direction_vectors1 / norm1[:, np.newaxis]
        direction_vectors2_normalized = direction_vectors2 / norm2[:, np.newaxis]

        # Calculate the angle between the lines
        dot_product = np.einsum('ij,ij->i', direction_vectors1_normalized, direction_vectors2_normalized)
        angles = np.arccos(np.clip(dot_product, -1.0, 1.0))
        return angles / 2

    @staticmethod
    def transform_to_local_coordinates(icrease_nodes_X_Na, O_a, O_basis_ab):
        """This function first subtracts the origin O_a from each point in icrease_nodes_X_Na, which shifts the points so that the origin goes to the origin of the local coordinate system. Then it uses np.einsum to transform the coordinates of the points to the local coordinate system. The 'ij,nj->ni' string tells np.einsum to treat O_basis_ab.T and shifted_points_X_Na as matrices and perform sp.Matrix multiplication on them.
        """
        # Shift the points so that the origin goes to the origin of the local coordinate system
        shifted_points_X_Na = icrease_nodes_X_Na - O_a[np.newaxis,:]

        # Transform the points to the local coordinate system
        local_points_X_Na = np.einsum('ij,nj->ni', O_basis_ab, shifted_points_X_Na)

        return local_points_X_Na

    @staticmethod
    def project_points_on_planes(centroids_X_Fa, vectors_X_Fa, centroids_X_Fia):
        """Function for array inputs, the first array contains the reference points 
        called `centroids_X_Fa`, the second array contains the normal vectors of 
        the respective plane called `vectors_X_Fa` and the third array contains 
        `centroids_X_Fia` the points for which we want to evaluate the projected 
        distances. 
        """
        # Normalize the plane normal vectors
        vectors_X_Fa = vectors_X_Fa / np.linalg.norm(vectors_X_Fa, axis=1, keepdims=True)

        # Calculate the vectors from the points on the planes to the corresponding points
        point_vectors_Fia = centroids_X_Fia - centroids_X_Fa[:,np.newaxis,:]
        
        # Project the point vectors onto the corresponding plane normals
        projections_Fi = np.einsum('Fia,Fa->Fi', point_vectors_Fia, vectors_X_Fa)
        
        # The absolute value of the projections are the shortest distances
        distances_Fi = np.abs(projections_Fi)

        return distances_Fi

    def plot_planes(self, plot, point_size=15, color=0x000000, 
                    normal_scale=10, plane_numbers=True):
        self.plot_points(plot, self.centroids_Fa, point_size=point_size, 
                         color=color, plot_numbers=plane_numbers)
        self.plot_lines(plot, self.centroids_Fa, self.normals_Fa * normal_scale)

    def plot_intersection_lines(self, plot, isc_vec_scale=400, color=0x000000, 
                                plot_labels=True, point_sise=30):
        isc_start_points_c_Li = (self.isc_points_Li - self.isc_vectors_Li * isc_vec_scale)
        isc_vectors_c_Li = self.isc_vectors_Li * 2 * isc_vec_scale

        self.plot_lines(plot, isc_start_points_c_Li, isc_vectors_c_Li, scale=1, 
                color=color, plot_labels=plot_labels)
        self.plot_points(plot, isc_start_points_c_Li + isc_vectors_c_Li, color=color,
                         point_size=point_sise)
        
    def plot_icrease_nodes(self, plot, node_numbers=True, point_size=15,
                           color=0x0000ff):
        self.plot_points(plot, self.icrease_nodes_X_Na, point_size=point_size, 
                            color=color, plot_numbers=True)

    def plot_icrease_lines(self, plot, line_numbers=False, color=0x000000):
        icrease_lines_X_Lia = self.icrease_lines_X_Lia
        start_icrease_lines_La = icrease_lines_X_Lia[:,0,:]
        vectors_icrease_lines_La =  icrease_lines_X_Lia[:,1,:] - start_icrease_lines_La 
        self.plot_lines(plot, start_icrease_lines_La, vectors_icrease_lines_La,
                        scale=1, color=color, plot_labels=line_numbers)
        
    def plot_O_basis(self, plot, basis_scale=20):
        O_a, O_basis_ab = self.O_basis_ab
        start_O_basis_ab = O_a[np.newaxis,:] + O_basis_ab * 0
        self.plot_lines(plot, start_O_basis_ab, O_basis_ab * basis_scale)

    def plot_O_crease_nodes(self, plot, node_numbers=True, point_size=15,
                             color=0x0000ff):
        self.plot_points(plot, self.O_crease_nodes_X_Na, point_size=point_size, 
                            color=color, plot_numbers=node_numbers)

    def plot_O_icrease_lines(self, plot, line_numbers=False, color=0x000000):
        icrease_lines_X_Lia = self.O_icrease_lines_X_Lia
        start_icrease_lines_La = icrease_lines_X_Lia[:,0,:]
        vectors_icrease_lines_La =  icrease_lines_X_Lia[:,1,:] - start_icrease_lines_La 
        self.plot_lines(plot, start_icrease_lines_La, vectors_icrease_lines_La,
                        scale=1, color=color, plot_labels=line_numbers)

    def plot_O_crease_lines(self, plot, line_numbers=False, color=0x000000):
        crease_lines_X_Lia = self.O_crease_lines_X_Lia
        start_crease_lines_La = crease_lines_X_Lia[:,0,:]
        vectors_crease_lines_La =  crease_lines_X_Lia[:,1,:] - start_crease_lines_La 
        self.plot_lines(plot, start_crease_lines_La, vectors_crease_lines_La,
                        scale=1, color=color, plot_labels=line_numbers)

    def plot_O_intersection_lines(self, plot, isc_vec_scale=400, color=0x000000, 
                                plot_labels=True):
        isc_start_points_c_Li = ( self.O_isc_points_Li - self.O_isc_vectors_Li * isc_vec_scale )
        isc_vectors_c_Li = self.O_isc_vectors_Li * 2 * isc_vec_scale
        self.plot_lines(plot, isc_start_points_c_Li, isc_vectors_c_Li, scale=1, 
                color=color, plot_labels=plot_labels)

    def plot_O_planes(self, plot, point_size=30, color=0x000000, 
                      normal_scale=10, plane_numbers=True):
        self.plot_points(plot, self.O_centroids_Fa, point_size=point_size, 
                         color=color, plot_numbers=plane_numbers)
        self.plot_lines(plot, self.O_centroids_Fa, self.O_normals_Fa * normal_scale)

    def plot_O_facets(self, plot, color=0x555555, opacity=1):
        mesh = k3d.mesh(self.O_crease_nodes_X_Na, 
                        self.facets_N_F, color=color,
                opacity=opacity,side='double')
        plot += mesh

    def plot_G_facets(self, plot, color=0x555555, opacity=1, module_numbers=True):
        mesh = k3d.mesh(self.G_crease_nodes_X_Na, 
                        self.facets_N_F, color=color,
                opacity=opacity,side='double')
        plot += mesh
        if module_numbers:
            X_a = np.copy(self.X_a)
            X_a[1] += self.O_flip * 100
            label = k3d.text(
                text=str(self.label), 
                position=X_a, 
                color=0x000000, 
                size=0.8
            )
            plot += label

    def plot_G_crease_lines(self, plot, line_numbers=False, color=0x000000):
        crease_lines_X_Lia = self.G_crease_lines_X_Lia
        start_crease_lines_La = crease_lines_X_Lia[:,0,:]
        vectors_crease_lines_La =  crease_lines_X_Lia[:,1,:] - start_crease_lines_La 
        self.plot_lines(plot, start_crease_lines_La, vectors_crease_lines_La,
                        scale=1, color=color, plot_labels=line_numbers)

    @staticmethod
    def plot_points(plot, points, point_size=1.0, color=0xff0000, plot_numbers=False):
        plt_points = k3d.points(
            positions=points, 
            point_size=point_size, 
            colors=np.array([color]*len(points), dtype=np.uint32)
        )
        plot += plt_points

        if plot_numbers:
            for i, point in enumerate(points):
                plt_text = k3d.text(
                    text=str(i), 
                    position=point, 
                    color=color, 
                    size=1.5
                )
                plot += plt_text

        return plot

    @staticmethod
    def plot_groups_of_points(plot, X_Fia):
        # Create a list of colors in the RGB spectrum
        colors = [0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff, 0x00ffff]

        for i, X_ia in enumerate(X_Fia):
            # Cycle through the colors for each facet
            color = colors[i % len(colors)]
            WBScannedCell.plot_points(plot, X_ia, point_size=15, color=color)

    @staticmethod
    def plot_lines(plot, start_points, directions, scale=10.0, color=0xff0000, 
                   plot_labels=False):

        for i, (start, direction) in enumerate(zip(start_points, directions)):
            vector = direction * scale
            end_point = start + vector
            line = k3d.line([start, end_point], color=color)
            plot += line

            if plot_labels:
                text = k3d.text(text=str(i), position=start + vector / 2, color=color, size=1.0)
                plot += text

        return plot

