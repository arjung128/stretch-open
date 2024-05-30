'''
plane-fitting utils
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

def get_masked_img(points, rgb):
    # pts_ = sort_coordinates(points)
    pts_ = points
    tupVerts=[(pts_[0, 0], pts_[0, 1]),
              (pts_[1, 0], pts_[1, 1]),
              (pts_[2, 0], pts_[2, 1]),
              (pts_[3, 0], pts_[3, 1])]

    x_, y_ = np.meshgrid(np.arange(1920), np.arange(1920)) # make a canvas with coordinates
    x_, y_ = x_.flatten(), y_.flatten()
    points = np.vstack((x_,y_)).T

    p = Path(tupVerts) # make a polygon
    grid = p.contains_points(points)
    mask = grid.reshape(1920,1920) # now you have a mask with points inside a polygon

    modified_rgb = rgb.copy()
    for y_ in range(1920):
        for x_ in range(1920):
            if mask[y_, x_]:
                modified_rgb[y_, x_, 0] = 255
                modified_rgb[y_, x_, 1] = 0
                modified_rgb[y_, x_, 2] = 0

    return mask, modified_rgb

def total_least_squares(X, y):
    # total least squares (i.e. perpendicular loss)
    pointcloud = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)
    centroid = pointcloud.mean(axis=0)
    points_centered = pointcloud - centroid
    u, _, _ = np.linalg.svd(points_centered.T)
    normal = u[:, 2]

    # w
    d = centroid.dot(normal)
    a = -1*normal[0] / normal[2]
    b = -1*normal[1] / normal[2]
    d = d / normal[2]
    w = np.array([d, a, b])

    return w, normal, centroid

def visualize_topdown_fit(drawer_pointcloud, pointcloud, w, inliers=None):
    plt.scatter(pointcloud[:, 0], pointcloud[:, 2])

    if inliers is not None:
        plt.scatter(inliers[:, 0], inliers[:, 2], color='yellow')

    min_x = min(drawer_pointcloud[:, 0])
    min_y = min(drawer_pointcloud[:, 1])
    min_z = min(drawer_pointcloud[:, 2])
    max_x = max(drawer_pointcloud[:, 0])
    max_y = max(drawer_pointcloud[:, 1])
    max_z = max(drawer_pointcloud[:, 2])

    xs = np.linspace(min_x, max_x, 50)
    ys = np.linspace(min_y, max_y, 50)
    zs = []
    for idx in range(50):
        z = w[0] + w[1]*xs[idx] + w[2]*ys[idx]
        zs.append(z)

    plt.plot(xs, zs, color='r')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.xlim(min_x-0.02, max_x+0.02)
    plt.ylim(min_z-0.1, max_z+0.1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gcf().set_size_inches(14, 10)
    plt.show()

def get_inliers(pointcloud, w, inlier_threshold=0.01):
    inliers = []
    for point_ in pointcloud:
        x_, y_, z_ = point_
        # distance to plane
        num = np.abs(-1*w[1]*x_ + -1*w[2]*y_ + z_ + -1*w[0])
        den = np.sqrt(w[1]**2 + w[2]**2 + 1)
        dist = num / den
        if dist < inlier_threshold:
            inliers.append(point_)
    inliers = np.array(inliers)
    return inliers

def fit_plane(mask, x, y, z, multiscan=False, intel_realsense=False):
    HEIGHT = 640
    WIDTH = 480

    # select points from pointcloud within masked region
    drawer_pointcloud = []
    zero_count = 0
    for y_ in range(HEIGHT):
        for x_ in range(WIDTH):
            if mask[y_, x_] and z[y_, x_]:
                point = np.array([x[y_, x_], y[y_, x_], z[y_, x_]])
                drawer_pointcloud.append(point)
            if z[y_, x_]==0:
                zero_count +=1
    # print("ZERO_COUNT: ", zero_count)
    drawer_pointcloud = np.array(drawer_pointcloud)

    # least squares (predicting the Z-coordinate in world frame)
    X = drawer_pointcloud[:, :2]
    y = drawer_pointcloud[:, 2]
    if X.shape[0] > 1000:
        num_to_get = 1000
    else:
        num_to_get = X.shape[0]
    idxs = np.random.choice(X.shape[0], num_to_get, replace=False)
    X_ = X[idxs, :]
    y_ = y[idxs]

    w_total, normal_total, centroid_total = total_least_squares(X_, y_)

    # improve with iterations
    for iter_ in range(5):
        inliers = get_inliers(drawer_pointcloud[np.random.choice(drawer_pointcloud.shape[0], num_to_get, replace=False), :], w_total)
        if len(inliers.shape) == 1:
            # import pdb; pdb.set_trace()
            break
        w_total, normal_total, _ = total_least_squares(inliers[:, :2], inliers[:, 2])

    # visualize_topdown_fit(drawer_pointcloud, drawer_pointcloud, w_total)

    return w_total, normal_total, centroid_total

def LinePlaneCollision(planeNormal, planePoint, rayPoint0, rayPoint1, epsilon=1e-6):
    rayDirection = rayPoint1 - rayPoint0
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")
    w = rayPoint0 - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi


def move_towards_centroid(points, step_len, x_pc, y_pc, z_pc):
    # find centroid in image space
    centroid_2d = np.mean(points, axis=0)

    preds = []
    for point in points:
        # direction towards centroid
        dir_to_centroid = centroid_2d - point
        # step = abs(dir_to_centroid) / dir_to_centroid
        step = np.sign(dir_to_centroid)

        # take 2 steps in this dir
        selected_y, selected_x = point[1], point[0]
        neighbor = np.array([x_pc[selected_y + int(step[1])*step_len, selected_x + int(step[0])*step_len],
                             y_pc[selected_y + int(step[1])*step_len, selected_x + int(step[0])*step_len],
                             z_pc[selected_y + int(step[1])*step_len, selected_x + int(step[0])*step_len]])

        pred = np.array([x_pc[selected_y, selected_x],
                         y_pc[selected_y, selected_x],
                         z_pc[selected_y, selected_x]])

        max_dist = np.max(pred - neighbor)
        if max_dist > 0.05:
            pred = neighbor
        preds.append(pred)

    return np.array(preds)

def project_corners_to_plane(camera_position, normal_total, centroid_total, x_pc, y_pc, z_pc, points):
    for step_len in [2, 5, 10]:
        preds = move_towards_centroid(points, step_len, x_pc, y_pc, z_pc)
        preds_y = preds[:, 1]
        preds_y_argsort = np.argsort(preds_y)
        if abs(preds_y_argsort[-1] - preds_y_argsort[-2]) == 1 or abs(preds_y_argsort[-1] - preds_y_argsort[-2]) == 3:
            break

    poi_1 = LinePlaneCollision(normal_total, centroid_total, camera_position, preds[0])
    poi_2 = LinePlaneCollision(normal_total, centroid_total, camera_position, preds[1])
    poi_3 = LinePlaneCollision(normal_total, centroid_total, camera_position, preds[2])
    poi_4 = LinePlaneCollision(normal_total, centroid_total, camera_position, preds[3])

    return np.array([poi_1, poi_2, poi_3, poi_4])

def deproject_pixel_to_point_plane(x_2d, y_2d, depth_data, mask):
    hfov = float(42) * np.pi / 180. # radians
    vfov = float(56) * np.pi / 180. # float(69) * np.pi / 180. # radians
    K = np.array([
        [1 / np.tan(hfov / 2.), 0., 0., 0.],
        [0., 1 / np.tan(vfov / 2.), 0., 0.],
        [0., 0.,  1, 0],
        [0., 0., 0, 1]])

    W, H = 480, 640

    # Now get an approximation for the true world coordinates -- see if they make sense
    # [-1, 1] for x and [1, -1] for y as array indexing is y-down while world is y-up
    xs, ys = np.meshgrid(np.linspace(-1,1,W), np.linspace(1,-1,H))
    depth = depth_data.reshape(1,H,W)
    xs = xs.reshape(1,H,W)
    ys = ys.reshape(1,H,W)

    # Unproject
    # negate depth as the camera looks along -Z
    xys = np.vstack((xs * depth , ys * depth, -depth, np.ones(depth.shape)))
    xys = xys.reshape(4, -1)
    xy_c0 = np.matmul(np.linalg.inv(K), xys)

    # Camera 1:
    # quaternion_0 = cameras[0].sensor_states['depth'].rotation
    # translation_0 = cameras[0].sensor_states['depth'].position
    # rotation_0 = quaternion.as_rotation_matrix(quaternion_0)
    T_world_camera0 = np.eye(4)
    # T_world_camera0[0:3,0:3] = rotation_0
    # T_world_camera0[0:3,3] = translation_0

    xyz = np.matmul(T_world_camera0, xy_c0)
    xyz = xyz / xyz[3,:]
    xyz_reshaped = xyz.reshape((4, 640, 480))

    x = xyz_reshaped[0]
    y = xyz_reshaped[1]
    z = xyz_reshaped[2]

    handle_3d_lookup = np.array([x[int(y_2d), int(x_2d)],
                                 y[int(y_2d), int(x_2d)],
                                 z[int(y_2d), int(x_2d)]])

    # fit plane
    w, normal, centroid = fit_plane(mask, x, y, z)

    # projection
    position = np.array([0., 0., 0.])
    handle_2d_arr = np.array([[int(x_2d), int(y_2d)],
                              [int(x_2d), int(y_2d)],
                              [int(x_2d), int(y_2d)],
                              [int(x_2d), int(y_2d)]])
    handle_3d_coord = project_corners_to_plane(position, normal, centroid, x, y, z, handle_2d_arr)

    # flip y-coordinate and z-coordinate for same convention
    handle_3d_lookup = np.array([handle_3d_lookup[0],
                                 -handle_3d_lookup[1],
                                 -handle_3d_lookup[2]])

    handle_3d_plane = np.array([handle_3d_coord[0][0],
                                -handle_3d_coord[0][1],
                                -handle_3d_coord[0][2]])

    return handle_3d_lookup, handle_3d_plane, normal







