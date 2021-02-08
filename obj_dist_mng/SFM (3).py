import numpy as np
from scipy.spatial import distance


def normalize(pts, focal, pp):
    normalized_list = [[x[0] - pp[0], x[1] - pp[1]] for x in pts]
    return np.array(normalized_list)/focal


def unnormalize(pts, focal, pp):
    unnormalized_list = [[focal*pt[0] + pp[0], focal*pt[1] + pp[1]] for pt in pts]
    return np.array(unnormalized_list)


def decompose(EM):
    R = EM[:3, :3]
    foe = np.array([EM[0, 3]/EM[2, 3], EM[1, 3]/EM[2, 3]])
    tZ = EM[2, 3]
    return R, foe, tZ


def rotate(pts, R):
    rotated_list = []
    for pt in pts:
        x_y_z = np.array(list(pt) + [1])
        multiplied_vec = np.matmul(R, x_y_z)
        rotated_pt = np.array([multiplied_vec[0] / multiplied_vec[2], multiplied_vec[1] / multiplied_vec[2]])
        rotated_list.append(rotated_pt)
    return np.array(rotated_list)


def find_corresponding_points(p, norm_pts_rot, foe):
    if len(norm_pts_rot) == 0:
        return None
    # compute the epipolar line between p and foe:
    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = (p[1] * foe[0] - p[0] * foe[1]) / (foe[0] - p[0])
    # run over all norm_pts_rot and find the one closest to the epipolar line
    dist = np.inf
    idx = -1
    closest = None
    for index, point in enumerate(norm_pts_rot):
        distance_up_to_factor = abs(m * point[0] + n - point[1])
        if distance_up_to_factor < dist:
            closest = point
            dist = distance_up_to_factor
            idx = index

    # return the closest point and its index
    if distance.euclidean(closest, p) > 150:
        return None, None

    return idx, closest


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    dist_x_foe = foe[0] - p_curr[0]
    estimate1 = tZ * dist_x_foe / (p_curr[0] - p_rot[0])
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    dist_y_foe = foe[1] - p_curr[1]
    estimate2 = tZ * dist_y_foe / (p_curr[1] - p_rot[1])
    # combine the two estimations and return estimated Z
    return (dist_x_foe**2 * estimate1 + dist_y_foe**2 * estimate2) / (dist_x_foe**2 + dist_y_foe**2)


def calc_TFL_dist(prev_points, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_points, curr_container, focal, pp)
    if(abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    elif (norm_prev_pts.size == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_points, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_points, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)

    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        if corresponding_p_ind is None:
            continue
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec
