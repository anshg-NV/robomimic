"""
Utilities for chunk-anchored relative actions.

An absolute action chunk holds end effector poses in a fixed robot frame. The relative
form re-expresses those poses w.r.t. a single anchor pose (the measured end effector pose
at the most recent observation), so the quantity the policy predicts is roughly zero-mean
and stays in [-1,1] after scaling. Everything outside the pose slices (gripper commands,
lower body commands, ...) is passed through untouched.

Each end effector contributes a (position, quaternion) pair to the absolute action, with
the quaternion in xyzw order, and a (position delta, rotation vector) pair to the relative
action -- so the relative action is one dimension shorter per end effector.
"""
import torch


def relative_ac_dim(ac_dim, num_eefs):
    """
    Dimension of the relative action that @absolute_to_relative produces.

    Args:
        ac_dim (int): dimension of the absolute action
        num_eefs (int): number of end effectors (each contributes one quat -> rotvec swap)

    Returns:
        int: dimension of the relative action
    """
    return ac_dim - num_eefs


def quat_multiply(q1, q2):
    """
    Hamilton product @q1 * @q2 of xyzw quaternions with shape (..., 4).
    """
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dim=-1)


def quat_conjugate(q):
    """
    Conjugate (inverse, for unit quaternions) of an xyzw quaternion with shape (..., 4).
    """
    return q * q.new_tensor([-1.0, -1.0, -1.0, 1.0])


def quat_to_rotvec(q, eps=1e-6):
    """
    Convert xyzw quaternions with shape (..., 4) to rotation vectors with shape (..., 3).

    The quaternion is mapped to the hemisphere with w >= 0 first, so the rotation vector is
    always the shortest-path one and the double cover does not put two encodings on one rotation.
    """
    q = torch.where(q[..., 3:4] < 0, -q, q)
    vec, w = q[..., :3], q[..., 3:4]
    norm = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
    # angle / sin(angle / 2), with the small-angle limit 2 + norm^2 / 3 used near identity
    # where the division is ill-conditioned.
    scale = torch.where(
        norm > eps,
        2.0 * torch.atan2(norm, w) / norm.clamp(min=eps),
        2.0 + norm.pow(2) / 3.0,
    )
    return vec * scale


def rotvec_to_quat(v, eps=1e-6):
    """
    Convert rotation vectors with shape (..., 3) to xyzw quaternions with shape (..., 4).
    """
    angle = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    # sin(angle / 2) / angle, with the small-angle limit 1/2 - angle^2 / 48 used near identity.
    scale = torch.where(
        angle > eps,
        torch.sin(0.5 * angle) / angle.clamp(min=eps),
        0.5 - angle.pow(2) / 48.0,
    )
    return torch.cat([v * scale, torch.cos(0.5 * angle)], dim=-1)


def absolute_to_relative(actions, anchor_pos, anchor_quat, pos_indices, quat_indices, pos_scale, rot_scale):
    """
    Re-express the end effector poses of an absolute action chunk w.r.t. an anchor pose.

    Args:
        actions (torch.Tensor): absolute actions, shape (B, T, ac_dim)
        anchor_pos (list): per-end-effector anchor positions, each shape (B, 3)
        anchor_quat (list): per-end-effector anchor xyzw quaternions, each shape (B, 4)
        pos_indices (list): start index of each end effector's position in the absolute action
        quat_indices (list): start index of each end effector's xyzw quaternion in the absolute action
        pos_scale (float): position deltas are divided by this [m]
        rot_scale (float): rotation vectors are divided by this [rad]

    Returns:
        torch.Tensor: relative actions clipped to [-1,1], shape (B, T, ac_dim - len(pos_indices))
    """
    out = []
    cursor = 0
    for eef, (pos_idx, quat_idx) in enumerate(zip(pos_indices, quat_indices)):
        assert quat_idx == pos_idx + 3, "each end effector's quaternion must directly follow its position"
        out.append(actions[..., cursor:pos_idx])

        pos = actions[..., pos_idx:pos_idx + 3] - anchor_pos[eef].unsqueeze(1)
        quat = quat_multiply(actions[..., quat_idx:quat_idx + 4], quat_conjugate(anchor_quat[eef]).unsqueeze(1))
        out.append(pos / pos_scale)
        out.append(quat_to_rotvec(quat) / rot_scale)
        cursor = quat_idx + 4

    out.append(actions[..., cursor:])
    return torch.cat(out, dim=-1).clamp(-1.0, 1.0)


def relative_to_absolute(actions, anchor_pos, anchor_quat, pos_indices, quat_indices, pos_scale, rot_scale):
    """
    Inverse of @absolute_to_relative: turn anchor-relative actions back into absolute ones.

    Args:
        actions (torch.Tensor): relative actions, shape (B, T, ac_dim - len(pos_indices))
        anchor_pos (list): per-end-effector anchor positions, each shape (B, 3)
        anchor_quat (list): per-end-effector anchor xyzw quaternions, each shape (B, 4)
        pos_indices (list): start index of each end effector's position in the absolute action
        quat_indices (list): start index of each end effector's xyzw quaternion in the absolute action
        pos_scale (float): position deltas are multiplied by this [m]
        rot_scale (float): rotation vectors are multiplied by this [rad]

    Returns:
        torch.Tensor: absolute actions, shape (B, T, ac_dim)
    """
    out = []
    cursor = 0
    for eef, pos_idx in enumerate(pos_indices):
        # The relative action dropped one dimension per end effector handled so far.
        rel_pos_idx = pos_idx - eef
        out.append(actions[..., cursor:rel_pos_idx])

        pos = actions[..., rel_pos_idx:rel_pos_idx + 3] * pos_scale + anchor_pos[eef].unsqueeze(1)
        quat = quat_multiply(
            rotvec_to_quat(actions[..., rel_pos_idx + 3:rel_pos_idx + 6] * rot_scale),
            anchor_quat[eef].unsqueeze(1),
        )
        out.append(pos)
        out.append(quat)
        cursor = rel_pos_idx + 6

    out.append(actions[..., cursor:])
    return torch.cat(out, dim=-1)
