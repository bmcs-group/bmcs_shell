import numpy as np

# Source:
# https://github.com/nghiaho12/rigid_transform_3D
# http://nghiaho.com/?page_id=671
# Input: expects 3xN matrix of points (A and B are two sets of points)
# Returns R,t (best rotation and translation that transform from set A to B)
# R = 3x3 rotation matrix
# t = 3x1 column vector
def get_best_rot_and_trans_3d(A, B):
    # TODO: check scipy scipy.spatial.transform.Rotation.align_vectors (# Rotation.align_vectors()), it's the same thing

    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def get_rot_matrix_around_vector(v, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    v_norm = v / np.sqrt(sum(v * v))

    # See: Rotation matrix from axis and angle (https://en.wikipedia.org/wiki/Rotation_matrix)
    cross_product_matrix = np.cross(v_norm, np.identity(v_norm.shape[0]) * -1)
    return c * np.identity(3) + s * cross_product_matrix + (1 - c) * np.outer(v_norm, v_norm)