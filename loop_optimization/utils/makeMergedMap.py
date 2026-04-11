import os 
import sys
import time 
import copy 
from io import StringIO

import pypcd # for the install, use this command: python3.x (use your python ver) -m pip install --user git+https://github.com/DanielPollithy/pypcd.git
from pypcd import pypcd

import numpy as np
from numpy import linalg as LA

import open3d as o3d

from pypcdMyUtils import * 

jet_table = np.load('jet_table.npy')
bone_table = np.load('bone_table.npy')

color_table = jet_table
color_table_len = color_table.shape[0]


##########################
# User only consider this block
##########################

data_dir = "/media/wxy/SB@home/zy99/AAA_map/LTAOM/ours/village03/" # should end with / 
scan_idx_range_to_stack = [0, 10000] # if you want a whole map, use [0, len(scan_files)]
node_skip = 2 # to skip some nodes for faster merge, e.g., use 1 for no skip, 2 for skip every other node, 5 for skip 4 out of 5 nodes, etc. (note that the scan idx is still counted in the original way, so the scan idxes in the map are still the same as the original ones, just some scans are not used for map merge)

# Choose which poses are used to build map:
# - "optimized_poses.txt" : after loop closure optimization
# - "odom_poses.txt"      : before loop closure (raw odometry/keyframe poses)
pose_file_for_map = "optimized_poses.txt"

# IMPORTANT:
# In current LTA-OM pipeline, Scans are usually already in world frame (raw odom world).
# If true:
# - using odom_poses.txt      => no extra rigid transform is applied
# - using optimized_poses.txt => apply relative correction T_opt * inv(T_odom)
scans_in_world_frame = True
raw_pose_file_for_scans = "odom_poses.txt"

num_points_in_a_scan = 150000 # for reservation (save faster) // e.g., use 150000 for 128 ray lidars, 100000 for 64 ray lidars, 30000 for 16 ray lidars, if error occured, use the larger value.

is_live_vis = False # recommend to use false 
is_o3d_vis = False
intensity_color_max = 200

is_near_removal = True
thres_near_removal = 2 # meter (to remove platform-myself structure ghost points)

# Export TUM trajectory files for evo.
export_tum_for_evo = True

##########################


def rotmat_to_quat_xyzw(R):
    """Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q_norm = LA.norm(q)
    if q_norm > 0:
        q /= q_norm
    return q


def quat_xyzw_to_rotmat(qx, qy, qz, qw):
    """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q_norm = LA.norm(q)
    if q_norm == 0:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = q / q_norm

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    R = np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)]
    ], dtype=np.float64)
    return R


def detect_pose_format(pose_path):
    """Detect pose format from first valid line: 'tum', 'kitti12', or 'kitti16'."""
    with open(pose_path, 'r') as f_pose:
        for line in f_pose:
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) == 8:
                return "tum"
            if len(fields) == 12:
                return "kitti12"
            if len(fields) == 16:
                return "kitti16"
            raise ValueError("Unsupported pose format (field count={}): {}".format(len(fields), pose_path))
    raise ValueError("Pose file is empty: {}".format(pose_path))


def load_poses_as_se3(pose_path):
    fmt = detect_pose_format(pose_path)
    poses_local = []
    with open(pose_path, 'r') as f_pose:
        for line in f_pose:
            line = line.strip()
            if not line:
                continue
            vals = [float(i) for i in line.split()]
            if fmt == "tum":
                # timestamp x y z qx qy qz qw
                _, x, y, z, qx, qy, qz, qw = vals
                T = np.eye(4, dtype=np.float64)
                T[0:3, 0:3] = quat_xyzw_to_rotmat(qx, qy, qz, qw)
                T[0:3, 3] = np.asarray([x, y, z], dtype=np.float64)
                poses_local.append(T)
            elif fmt == "kitti12":
                pose_SE3 = np.asarray(vals, dtype=np.float64)
                pose_SE3 = np.vstack((np.reshape(pose_SE3, (3, 4)), np.asarray([0, 0, 0, 1], dtype=np.float64)))
                poses_local.append(pose_SE3)
            elif fmt == "kitti16":
                pose_SE3 = np.asarray(vals, dtype=np.float64)
                pose_SE3 = np.reshape(pose_SE3, (4, 4))
                poses_local.append(pose_SE3)
            else:
                raise ValueError("Unsupported pose format for {}".format(pose_path))
    return poses_local


def read_times_file(times_path):
    times = []
    with open(times_path, 'r') as f_time:
        for line in f_time:
            line = line.strip()
            if not line:
                continue
            times.append(float(line))
    return times


def export_tum_from_pose_file(pose_path, tum_out_path, times_path=None):
    fmt = detect_pose_format(pose_path)

    if fmt == "tum":
        # Already TUM style; normalize output formatting and keep original timestamps.
        with open(pose_path, 'r') as fin, open(tum_out_path, 'w') as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                vals = [float(i) for i in line.split()]
                if len(vals) != 8:
                    continue
                fout.write("{:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n".format(*vals))
        print("TUM saved:", tum_out_path)
        return

    poses_local = load_poses_as_se3(pose_path)
    if times_path is None or (not os.path.exists(times_path)):
        print("Warning: not found times file for", pose_path, "skip TUM export.")
        return

    times = read_times_file(times_path)
    n = min(len(poses_local), len(times))
    if n == 0:
        print("Skip TUM export for", pose_path, "(empty poses or times).")
        return

    if len(poses_local) != len(times):
        print("Warning: pose/time size mismatch for", pose_path,
              "poses:", len(poses_local), "times:", len(times), "use:", n)

    with open(tum_out_path, 'w') as f_tum:
        for idx in range(n):
            T = poses_local[idx]
            t = T[0:3, 3]
            qx, qy, qz, qw = rotmat_to_quat_xyzw(T[0:3, 0:3])
            f_tum.write("{:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n".format(
                times[idx], t[0], t[1], t[2], qx, qy, qz, qw
            ))
    print("TUM saved:", tum_out_path)


#
scan_dir = data_dir + "Scans"
if not os.path.isdir(scan_dir):
    scan_dir = data_dir + "scans"
scan_files = os.listdir(scan_dir) 
scan_files.sort()

pose_path_for_map = data_dir + pose_file_for_map
poses = load_poses_as_se3(pose_path_for_map)
print("Use pose file for map merge:", pose_path_for_map)

raw_poses_for_scans = None
if scans_in_world_frame:
    raw_pose_path = data_dir + raw_pose_file_for_scans
    if not os.path.exists(raw_pose_path):
        raise FileNotFoundError("Scans are configured as world-frame, but raw pose file is missing: {}".format(raw_pose_path))
    raw_poses_for_scans = load_poses_as_se3(raw_pose_path)
    print("Scans are treated as world-frame. Raw pose file:", raw_pose_path)

if export_tum_for_evo:
    time_path = data_dir + "times.txt"
    optimized_pose_path = data_dir + "optimized_poses.txt"
    odom_pose_path = data_dir + "odom_poses.txt"
    if os.path.exists(time_path):
        if os.path.exists(optimized_pose_path):
            export_tum_from_pose_file(optimized_pose_path, data_dir + "optimized_poses_tum.txt", time_path)
        else:
            print("Warning: not found", optimized_pose_path)

        if os.path.exists(odom_pose_path):
            export_tum_from_pose_file(odom_pose_path, data_dir + "odom_poses_tum.txt", time_path)
        else:
            print("Warning: not found", odom_pose_path)
    else:
        print("Warning: not found", time_path, "skip TUM export.")


#
assert (scan_idx_range_to_stack[1] > scan_idx_range_to_stack[0])
print("Merging scans from", scan_idx_range_to_stack[0], "to", scan_idx_range_to_stack[1])


#
if(is_live_vis):
    vis = o3d.visualization.Visualizer() 
    vis.create_window('Map', visible = True) 

nodes_count = 0
pcd_combined_for_vis = o3d.geometry.PointCloud()
pcd_combined_for_save = None

# The scans from 000000.pcd should be prepared if it is not used (because below code indexing is designed in a naive way)

# manually reserve memory for fast write  
num_all_points_expected = int(num_points_in_a_scan * np.round((scan_idx_range_to_stack[1] - scan_idx_range_to_stack[0])/node_skip))

np_xyz_all = np.empty([num_all_points_expected, 3])
np_intensity_all = np.empty([num_all_points_expected, 1])
curr_count = 0

for node_idx in range(len(scan_files)):
    if(node_idx < scan_idx_range_to_stack[0] or node_idx >= scan_idx_range_to_stack[1]):
        continue

    if node_idx >= len(poses):
        print("Warning: pose index out of range at", node_idx, "(num poses:", len(poses), "), stop merge.")
        break

    nodes_count = nodes_count + 1
    if( nodes_count % node_skip != 0): 
        if(node_idx != scan_idx_range_to_stack[0]): # to ensure the vis init 
            continue

    print("read keyframe scan idx", node_idx)

    scan_pose = poses[node_idx]
    raw_pose = raw_poses_for_scans[node_idx] if raw_poses_for_scans is not None and node_idx < len(raw_poses_for_scans) else None

    scan_path = os.path.join(scan_dir, scan_files[node_idx])
    scan_pcd = o3d.io.read_point_cloud(scan_path)
    scan_xyz_local = copy.deepcopy(np.asarray(scan_pcd.points))

    scan_pypcd_with_intensity = pypcd.PointCloud.from_path(scan_path)
    scan_intensity = scan_pypcd_with_intensity.pc_data['intensity']
    scan_intensity_colors_idx = np.round( (color_table_len-1) * np.minimum( 1, np.maximum(0, scan_intensity / intensity_color_max) ) )
    scan_intensity_colors = color_table[scan_intensity_colors_idx.astype(int)]

    if scans_in_world_frame:
        # If scans are already in raw-odom world frame, only apply relative correction when needed.
        if raw_pose is None:
            print("Warning: raw pose missing at", node_idx, "skip this scan")
            continue
        T_corr = np.matmul(scan_pose, LA.inv(raw_pose))
        scan_pcd_global = scan_pcd.transform(T_corr)
    else:
        # Legacy behavior: scan is in local frame and needs full pose transform.
        scan_pcd_global = scan_pcd.transform(scan_pose)

    scan_pcd_global.colors = o3d.utility.Vector3dVector(scan_intensity_colors)
    scan_xyz = np.asarray(scan_pcd_global.points)

    scan_intensity = np.expand_dims(scan_intensity, axis=1) 
    if scans_in_world_frame and raw_pose is not None:
        sensor_center = raw_pose[0:3, 3]
        scan_ranges = LA.norm(scan_xyz - sensor_center, axis=1)
    else:
        scan_ranges = LA.norm(scan_xyz_local, axis=1)

    if(is_near_removal):
        eff_idxes = np.where (scan_ranges > thres_near_removal)
        scan_xyz = scan_xyz[eff_idxes[0], :]
        scan_intensity = scan_intensity[eff_idxes[0], :]

        scan_pcd_global = scan_pcd_global.select_by_index(eff_idxes[0])

    if(is_o3d_vis):
        pcd_combined_for_vis += scan_pcd_global # open3d pointcloud class append is fast 

    if is_live_vis:
        if(node_idx == scan_idx_range_to_stack[0]): # to ensure the vis init 
            vis.add_geometry(pcd_combined_for_vis) 

        vis.update_geometry(pcd_combined_for_vis)
        vis.poll_events()
        vis.update_renderer()

    # save 
    np_xyz_all[curr_count:curr_count + scan_xyz.shape[0], :] = scan_xyz
    np_intensity_all[curr_count:curr_count + scan_xyz.shape[0], :] = scan_intensity

    curr_count = curr_count + scan_xyz.shape[0]
    print(curr_count)
 
#
if(is_o3d_vis):
    print("draw the merged map.")
    o3d.visualization.draw_geometries([pcd_combined_for_vis])


# save ply having intensity
np_xyz_all = np_xyz_all[0:curr_count, :]
np_intensity_all = np_intensity_all[0:curr_count, :]

np_xyzi_all = np.hstack( (np_xyz_all, np_intensity_all) )
xyzi = make_xyzi_point_cloud(np_xyzi_all)

pose_tag = os.path.splitext(os.path.basename(pose_file_for_map))[0]
map_name = data_dir + "map_" + pose_tag + "_" + str(scan_idx_range_to_stack[0]) + "_to_" + str(scan_idx_range_to_stack[1]) + "_with_intensity.pcd"
xyzi.save_pcd(map_name, compression='binary_compressed')
print("intensity map is save (path:", map_name, ")")

# save rgb colored points 
# map_name = data_dir + "map_" + str(scan_idx_range_to_stack[0]) + "_to_" + str(scan_idx_range_to_stack[1]) + ".pcd"
# o3d.io.write_point_cloud(map_name, pcd_combined_for_vis)
# print("the map is save (path:", map_name, ")")


