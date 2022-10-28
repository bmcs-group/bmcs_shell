import numpy as np

def get_facets_normals(X_Ia, I_Fi):
    X_Ia = np.copy(X_Ia)
    I_Fi = np.copy(I_Fi)

    # Flip normals to have same direction
    X_Fia = X_Ia[I_Fi]
    normals_Fi = np.cross(X_Fia[:, 1, :] - X_Fia[:, 0, :], X_Fia[:, 2, :] - X_Fia[:, 0, :])
    To_flip_F = normals_Fi[:, 2] < 0  # all items where z of normal is negative
    I_Fi[To_flip_F] = np.flip(I_Fi[To_flip_F], axis=1)
    # Update X_Fia
    X_Fia = X_Ia[I_Fi]
    normals_Fi = np.cross(X_Fia[:, 1, :] - X_Fia[:, 0, :], X_Fia[:, 2, :] - X_Fia[:, 0, :])
    normals_Fi_norm = normals_Fi / np.sqrt(np.sum(normals_Fi * normals_Fi, axis=1))[:, np.newaxis]
    return normals_Fi_norm


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_dih_angles(X_Ia, I_Fi):
    """
    :return: numpy array with the shape (facets_num, 3) with dihedral angles between each facets and the surrounding
    facets in a trianglor mesh given by X_Ia and I_Fi. If facet has less than 3 surrounding facets, the angles of the
    non-existing facets will be np.nan!
    """
    # TODO: this not a vectorized approach, vectorize it for better performance
    X_Ia = np.copy(X_Ia)
    I_Fi = np.copy(I_Fi)

    # Flip normals to have same direction
    X_Fia = X_Ia[I_Fi]
    normals_Fi = np.cross(X_Fia[:, 1, :] - X_Fia[:, 0, :], X_Fia[:, 2, :] - X_Fia[:, 0, :])
    To_flip_F = normals_Fi[:, 2] < 0  # all items where z of normal is negative
    I_Fi[To_flip_F] = np.flip(I_Fi[To_flip_F], axis=1)
    # Update X_Fia
    X_Fia = X_Ia[I_Fi]
    normals_Fi = np.cross(X_Fia[:, 1, :] - X_Fia[:, 0, :], X_Fia[:, 2, :] - X_Fia[:, 0, :])

    # Get a list of lists mapping each facet i to their surrounding facets (as may have 2 or 3 surrounding facets
    # for each facet)
    facet_surr_facets_mapping = []
    for i, (i1, i2, i3) in enumerate(I_Fi):
        facet_surr_facets_mapping.append([])
        for j, indices in enumerate(I_Fi):
            if i != j:
                if i1 in indices and i2 in indices or i1 in indices and i3 in indices or i2 in indices and i3 in indices:
                    facet_surr_facets_mapping[i].append(j)

    # Get the angle between each facet and its surrounding facets, facet_angles_mapping has the same shape as
    # of facet_surr_facets_mapping
    facet_angles_mapping = []
    for i, surr_facets in enumerate(facet_surr_facets_mapping):
        facet_norm = normals_Fi[i]
        facet_angles_mapping.append([])
        for facet_j in surr_facets:
            angle = angle_between(facet_norm, normals_Fi[facet_j])
            facets_angle  = np.round(np.rad2deg(angle), 1)
            dih_angle = 180 - facets_angle
            facet_angles_mapping[i].append(dih_angle)

    # Convert angles list of lists to numpy array with np.nan for missing elements (for facets which have only
    # 2 surrounding facets)
    # F index for facets, g index for the angle of the surrounding facets
    facet_angles_mapping_Fg = np.zeros((len(facet_angles_mapping), 3))
    for i, row in enumerate(facet_angles_mapping_Fg):
        facet_angles_mapping_Fg[i, 0] = facet_angles_mapping[i][0]
        facet_angles_mapping_Fg[i, 1] = facet_angles_mapping[i][1]
        facet_angles_mapping_Fg[i, 2] = facet_angles_mapping[i][2] if len(facet_angles_mapping[i]) == 3 else np.nan

    # NOTE: using np.max or np.min with facet_angles_mapping_Fg will return np.nan because of included nans, use
    # np.nanmaxn and np.nanmin instead
    max_angle = np.nanmax(facet_angles_mapping_Fg)
    min_angle = np.nanmin(facet_angles_mapping_Fg)

    return facet_angles_mapping_Fg, max_angle, min_angle
