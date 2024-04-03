# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import re

import torch as th
import torch.nn as nn

from typing import Dict, Any

from ca_body.utils.quaternion import Quaternion

from typing import Optional, Tuple

import logging

logger = logging.getLogger(__name__)


class ParameterTransform(nn.Module):
    def __init__(self, lbs_cfg_dict: Dict[str, Any]):
        super().__init__()
        # self.pose_names = list(lbs_cfg_dict["joint_names"])
        self.channel_names = list(lbs_cfg_dict["channel_names"])
        transform_offsets = th.FloatTensor(lbs_cfg_dict["transform_offsets"])
        transform = th.FloatTensor(lbs_cfg_dict["transform"])
        self.limits = lbs_cfg_dict["limits"]

        self.nr_scaling_params = lbs_cfg_dict["nr_scaling_params"]
        self.nr_position_params = lbs_cfg_dict["nr_position_params"]
        self.nr_total_params = self.nr_scaling_params + self.nr_position_params

        self.register_buffer("transform_offsets", transform_offsets)
        self.register_buffer("transform", transform)

    def forward(self, pose: th.Tensor) -> th.Tensor:
        """
        Args:
            pose: raw pose inputs, shape [B, n_pose_params]
        Returns:
            skeleton parameters, shape [B, len(channel_names)*nr_skeleton_joints]
        """
        return self.transform.mm(pose.t()).t() + self.transform_offsets


class LinearBlendSkinning(nn.Module):
    def __init__(
        self,
        model_json: Dict[str, Any],
        lbs_config_dict: Dict[str, Any],
        num_max_skin_joints: int = 8,
        scale_path: str = None,
    ):
        super().__init__()

        model = model_json
        self.param_transform = ParameterTransform(lbs_config_dict)

        self.joint_names = []

        nr_joints = len(model["Skeleton"]["Bones"])
        joint_parents = th.zeros((nr_joints, 1), dtype=th.int64)
        joint_rotation = th.zeros((nr_joints, 4), dtype=th.float32)
        joint_offset = th.zeros((nr_joints, 3), dtype=th.float32)
        for idx, bone in enumerate(model["Skeleton"]["Bones"]):
            self.joint_names.append(bone["Name"])
            if bone["Parent"] > nr_joints:
                joint_parents[idx] = -1
            else:
                joint_parents[idx] = bone["Parent"]
            joint_rotation[idx, :] = th.FloatTensor(bone["PreRotation"])
            joint_offset[idx, :] = th.FloatTensor(bone["TranslationOffset"])

        skin_model = model["SkinnedModel"]
        mesh_vertices = th.FloatTensor(skin_model["RestPositions"])
        mesh_normals = th.FloatTensor(skin_model["RestVertexNormals"])

        weights = th.FloatTensor([e[1] for e in skin_model["SkinningWeights"]])
        indices = th.LongTensor([e[0] for e in skin_model["SkinningWeights"]])
        offsets = th.LongTensor(skin_model["SkinningOffsets"])

        nr_vertices = len(offsets) - 1
        skin_weights = th.zeros((nr_vertices, num_max_skin_joints), dtype=th.float32)
        skin_indices = th.zeros((nr_vertices, num_max_skin_joints), dtype=th.int64)

        offset_right = offsets[1:]
        for offset in range(num_max_skin_joints):
            offset_left = offsets[:-1] + offset
            skin_weights[offset_left < offset_right, offset] = weights[
                offset_left[offset_left < offset_right]
            ]
            skin_indices[offset_left < offset_right, offset] = indices[
                offset_left[offset_left < offset_right]
            ]

        mesh_faces = th.IntTensor(skin_model["Faces"]["Indices"]).view(-1, 3)
        mesh_texture_faces = th.IntTensor(skin_model["Faces"]["TextureIndices"]).view(
            -1, 3
        )
        mesh_texture_coords = th.FloatTensor(skin_model["TextureCoordinates"]).view(
            -1, 2
        )

        # zero_pose = torch.zeros((1, len(self.param_transform.pose_names)), dtype=torch.float32)
        zero_pose = th.zeros(
            (1, self.param_transform.nr_total_params), dtype=th.float32
        )
        bind_state = solve_skeleton_state(
            self.param_transform(zero_pose), joint_offset, joint_rotation, joint_parents
        )

        # self.register_buffer('mesh_vertices', mesh_vertices) # we want to train on rest pose
        # self.mesh_vertices = nn.Parameter(mesh_vertices, requires_grad=optimize_mesh)
        self.register_buffer("mesh_vertices", mesh_vertices)

        self.register_buffer("joint_parents", joint_parents)
        self.register_buffer("joint_rotation", joint_rotation)
        self.register_buffer("joint_offset", joint_offset)
        self.register_buffer("mesh_normals", mesh_normals)
        self.register_buffer("mesh_faces", mesh_faces)
        self.register_buffer("mesh_texture_faces", mesh_texture_faces)
        self.register_buffer("mesh_texture_coords", mesh_texture_coords)
        self.register_buffer("skin_weights", skin_weights)
        self.register_buffer("skin_indices", skin_indices)
        self.register_buffer("bind_state", bind_state)
        self.register_buffer("rest_vertices", mesh_vertices)

        # pre-compute joint weights
        self.register_buffer("joints_weights", self.compute_joints_weights())

        if scale_path is not None:
            scale = np.loadtxt(scale_path).astype(np.float32)[np.newaxis]
            scale = scale[:, 0, :] if len(scale.shape) == 3 else scale
            self.register_buffer("scale", th.tensor(scale))

    @property
    def num_verts(self):
        return self.mesh_vertices.size(0)

    @property
    def num_joints(self):
        return self.joint_offset.size(0)

    @property
    def num_params(self):
        return self.skin_weights.shape[-1]

    def compute_rigid_transforms(
        self, global_pose: th.Tensor, local_pose: th.Tensor, scale: th.Tensor
    ):
        """Returns rigid transforms."""
        params = th.cat([global_pose, local_pose, scale], axis=-1)
        params = self.param_transform(params)
        return solve_skeleton_state(
            params, self.joint_offset, self.joint_rotation, self.joint_parents
        )

    def compute_rigid_transforms_matrix(
        self, global_pose: th.Tensor, local_pose: th.Tensor, scale: th.Tensor
    ):
        params = th.cat([global_pose, local_pose, scale], axis=-1)
        params = self.param_transform(params)
        states = solve_skeleton_state(
            params, self.joint_offset, self.joint_rotation, self.joint_parents
        )
        return states_to_matrix(self.bind_state, states)

    def compute_joints_weights(self, drop_empty=False):
        """Compute weights per joint given flattened weights-indices."""
        idxs_verts = th.arange(self.num_verts)[:, np.newaxis].expand(
            -1, self.num_params
        )
        weights_joints = th.zeros(
            (self.num_joints, self.num_verts),
            dtype=th.float32,
            device=self.skin_weights.device,
        )
        weights_joints[self.skin_indices, idxs_verts] = self.skin_weights

        if drop_empty:
            weights_joints = weights_joints[weights_joints.sum(axis=-1).abs() > 0]

        return weights_joints

    def compute_root_rigid_transform(
        self, poses: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Get a transform of the root joint."""
        scales = th.zeros(
            (poses.shape[0], self.nr_total_params - poses.shape[1]),
            dtype=poses.dtype,
            device=poses.device,
        )
        params = th.cat((poses, scales), 1)
        states = solve_skeleton_state(
            self.param_transform(params),
            self.joint_offset,
            self.joint_rotation,
            self.joint_parents,
        )
        mat = states_to_matrix(self.bind_state, states)
        return mat[:, 1, :, 3], mat[:, 1, :, :3]

    def compute_relative_rigid_transforms(
        self, global_pose: th.Tensor, local_pose: th.Tensor, scale: th.Tensor
    ):
        params = th.cat([global_pose, local_pose, scale], axis=-1)
        params = self.param_transform(params)

        batch_size = params.shape[0]

        joint_offset = self.joint_offset
        joint_rotation = self.joint_rotation

        # batch processing for parameters
        jp = params.view((batch_size, -1, 7))
        lt = jp[:, :, 0:3] + joint_offset.unsqueeze(0)
        lr = Quaternion.batchMul(
            joint_rotation.unsqueeze(0), Quaternion.batchFromXYZ(jp[:, :, 3:6])
        )
        return th.cat([lt, lr], axis=-1)

    def skinning(
        self, bind_state: th.Tensor, vertices: th.Tensor, target_states: th.Tensor
    ):
        """
        Apply skinning to a set of states

        Args:
            b/bind_state: 1 x nr_joint x 8 bind state
            v/vertices: 1 x nr_vertices x 3 vertices
            t/target_states: batch_size x nr_joint x 8 current states

        Returns:
            batch_size x nr_vertices x 3 skinned vertices
        """
        assert target_states.size()[1:] == bind_state.size()[1:]

        mat = states_to_matrix(bind_state, target_states)

        # apply skinning to vertices
        vs = th.matmul(
            mat[:, self.skin_indices],
            th.cat((vertices, th.ones_like(vertices[:, :, 0]).unsqueeze(2)), dim=2)
            .unsqueeze(2)
            .unsqueeze(4),
        )
        ws = self.skin_weights.unsqueeze(2).unsqueeze(3)
        res = (vs * ws).sum(dim=2).squeeze(3)

        return res

    def unpose(self, poses: th.Tensor, scales: th.Tensor, verts: th.Tensor):
        """
        :param poses: 100 (tx ty tz rx ry rz) params in blueman
        :param scales: 29 (s) params in blueman
        :return:
        """
        # check shape of poses and scales
        params = th.cat((poses, scales), 1)
        states = solve_skeleton_state(
            self.param_transform(params),
            self.joint_offset,
            self.joint_rotation,
            self.joint_parents,
        )

        return self.unskinning(self.bind_state, states, verts)

    def unskinning(
        self, bind_state: th.Tensor, target_states: th.Tensor, verts: th.Tensor
    ):
        """Apply skinning to a set of states

        Args:
            bind_state: [B, NJ, 8]  -  bind state
            target_states: [B, NJ, 8] - current states
            vertices: [B, V, 3] - vertices

        Returns:
            batch_size x nr_vertices x 3 skinned vertices
        """
        assert target_states.size()[1:] == bind_state.size()[1:]

        mat = states_to_matrix(bind_state, target_states)

        ws = self.skin_weights[None, :, :, None, None]
        sum_mat = (mat[:, self.skin_indices] * ws).sum(dim=2)

        sum_mat4x4 = th.cat((sum_mat, th.zeros_like(sum_mat[:, :, :1, :])), dim=2)
        sum_mat4x4[:, :, 3, 3] = 1.0

        verts_4d = th.cat((verts, th.ones_like(verts[:, :, :1])), dim=2).unsqueeze(3)

        resmesh = []
        for i in range(sum_mat.shape[0]):
            newmat = sum_mat4x4[i, :, :, :].contiguous()
            invnewmat = newmat.inverse()
            tmpvets = invnewmat.matmul(verts_4d[i])
            resmesh.append(tmpvets.unsqueeze(0))
        resmesh = th.cat(resmesh)

        return resmesh.squeeze(3)[..., :3].contiguous()

    def forward(
        self,
        poses: th.Tensor,
        scales: th.Tensor,
        verts_unposed: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """
        Args:
            poses: [B, NP] - pose parametersa
            scales: [B, NS] - additional scaling params
            verts_unposed: [B, N, 3] - unposed vertices

        Returns:
            [B, N, 3] - posed vertices
        """
        params = th.cat((poses, scales), 1)
        params_transformed = self.param_transform(params)
        states = solve_skeleton_state(
            params_transformed,
            self.joint_offset,
            self.joint_rotation,
            self.joint_parents,
        )
        if verts_unposed is None:
            mesh = self.skinning(
                self.bind_state, self.mesh_vertices.unsqueeze(0), states
            )
        else:
            mesh = self.skinning(self.bind_state, verts_unposed, states)
        return mesh


def solve_skeleton_state(
    param: th.Tensor,
    joint_offset: th.Tensor,
    joint_rotation: th.Tensor,
    joint_parents: th.Tensor,
):
    """
    :param param: batch_size x (7*nr_skeleton_joints) ParamTransform Outputs.
    :return: batch_size x nr_skeleton_joints x 8 Skeleton States
        8 stands form 3 translation + 4 rotation (quat) + 1 scale
    """
    batch_size = param.shape[0]
    # batch processing for parameters
    jp = param.view((batch_size, -1, 7))
    lt = jp[:, :, 0:3] + joint_offset.unsqueeze(0)
    lr = Quaternion.batchMul(
        joint_rotation.unsqueeze(0), Quaternion.batchFromXYZ(jp[:, :, 3:6])
    )
    ls = th.pow(
        th.tensor([2.0], dtype=th.float32, device=param.device),
        jp[:, :, 6].unsqueeze(2),
    )

    state = []
    for index, parent in enumerate(joint_parents):
        if int(parent) != -1:
            gr = Quaternion.batchMul(
                state[parent][:, :, 3:7], lr[:, index, :].unsqueeze(1)
            )
            gt = (
                Quaternion.batchRot(
                    state[parent][:, :, 3:7],
                    lt[:, index, :].unsqueeze(1) * state[parent][:, :, 7].unsqueeze(2),
                )
                + state[parent][:, :, 0:3]
            )
            gs = state[parent][:, :, 7].unsqueeze(2) * ls[:, index, :].unsqueeze(1)
            state.append(th.cat((gt, gr, gs), dim=2))
        else:
            state.append(
                th.cat((lt[:, index, :], lr[:, index, :], ls[:, index, :]), dim=1).view(
                    (batch_size, 1, 8)
                )
            )

    return th.cat(state, dim=1)


def states_to_matrix(
    bind_state: th.Tensor, target_states: th.Tensor, return_transform: bool = False
):
    # multiply bind inverse with states
    br = Quaternion.batchInvert(bind_state[:, :, 3:7])
    bs = bind_state[:, :, 7].unsqueeze(2).reciprocal()
    bt = Quaternion.batchRot(br, -bind_state[:, :, 0:3]) * bs

    # applying rotation
    tr = Quaternion.batchMul(target_states[:, :, 3:7], br)
    # applying scaling
    ts = target_states[:, :, 7].unsqueeze(2) * bs
    # applying transformation
    tt = (
        Quaternion.batchRot(
            target_states[:, :, 3:7], bt * target_states[:, :, 7].unsqueeze(2)
        )
        + target_states[:, :, 0:3]
    )

    # convert to matrices
    twx = 2.0 * tr[:, :, 0] * tr[:, :, 3]
    twy = 2.0 * tr[:, :, 1] * tr[:, :, 3]
    twz = 2.0 * tr[:, :, 2] * tr[:, :, 3]
    txx = 2.0 * tr[:, :, 0] * tr[:, :, 0]
    txy = 2.0 * tr[:, :, 1] * tr[:, :, 0]
    txz = 2.0 * tr[:, :, 2] * tr[:, :, 0]
    tyy = 2.0 * tr[:, :, 1] * tr[:, :, 1]
    tyz = 2.0 * tr[:, :, 2] * tr[:, :, 1]
    tzz = 2.0 * tr[:, :, 2] * tr[:, :, 2]
    mat = th.stack(
        (
            th.stack((1.0 - (tyy + tzz), txy + twz, txz - twy), dim=2) * ts,
            th.stack((txy - twz, 1.0 - (txx + tzz), tyz + twx), dim=2) * ts,
            th.stack((txz + twy, tyz - twx, 1.0 - (txx + tyy)), dim=2) * ts,
            tt,
        ),
        dim=3,
    )
    if return_transform:
        return mat, (tr, tt, ts)
    return mat


def load_momentum_cfg(model, lbs_config_txt_fh, nr_scaling_params=None):
    def find(l, x):
        try:
            return l.index(x)
        except ValueError:
            return None

    """Load a parameter configuration file"""
    channelNames = ["tx", "ty", "tz", "rx", "ry", "rz", "sc"]
    paramNames = []
    joint_names = []
    for idx, bone in enumerate(model["Skeleton"]["Bones"]):
        joint_names.append(bone["Name"])

    def findJointIndex(x):
        return find(joint_names, x)

    def findParameterIndex(x):
        return find(paramNames, x)

    limits = []

    # create empty result
    transform_triplets = []
    lines = lbs_config_txt_fh.readlines()

    # read until end
    for line in lines:
        # strip comments
        line = line[: line.find("#")]

        if line.find("limit") != -1:
            r = re.search("limit ([\\w.]+) (\\w+) (.*)", line)
            if r is None:
                continue

            if len(r.groups()) != 3:
                logger.info("Failed to parse limit configuration line :\n   " + line)
                continue

            # find parameter and/or joint index
            fullname = r.groups()[0]
            type = r.groups()[1]
            remaining = r.groups()[2]

            parameterIndex = findParameterIndex(fullname)
            jointName = fullname.split(".")
            jointIndex = findJointIndex(jointName[0])
            channelIndex = -1

            if jointIndex is not None and len(jointName) == 2:
                # find matching channel name
                channelIndex = channelNames.index(jointName[1])
                if channelIndex is None:
                    logger.info(
                        "Unknown joint channel name "
                        + jointName[1]
                        + " in parameter configuration line :\n   "
                        + line
                    )
                    continue

            # only parse passive limits for now
            if type == "minmax_passive" or type == "minmax":
                # match [<float> , <float>] <optional weight>
                rp = re.search(
                    "\\[\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*,\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*\\](\\s*[-+]?[0-9]*\\.?[0-9]+)?",
                    remaining,
                )

                if len(rp.groups()) != 3:
                    logger.info(
                        f"Failed to parse passive limit configuration line :\n {line}"
                    )
                    continue

                minVal = float(rp.groups()[0])
                maxVal = float(rp.groups()[1])
                weightVal = 1.0
                if len(rp.groups()) == 3 and not rp.groups()[2] is None:
                    weightVal = float(rp.groups()[2])

                # result.limits.append([jointIndex * 7 + channelIndex, minVal, maxVal])

                if channelIndex >= 0:
                    valueIndex = jointIndex * 7 + channelIndex
                    limit = {
                        "type": "LimitMinMaxJointValue",
                        "str": fullname,
                        "valueIndex": valueIndex,
                        "limits": [minVal, maxVal],
                        "weight": weightVal,
                    }
                    limits.append(limit)
                else:
                    if parameterIndex is None:
                        logger.info(
                            f"Unknown parameterIndex : {fullname}\n  {line} {paramNames} "
                        )
                        continue
                    limit = {
                        "type": "LimitMinMaxParameter",
                        "str": fullname,
                        "parameterIndex": parameterIndex,
                        "limits": [minVal, maxVal],
                        "weight": weightVal,
                    }
                    limits.append(limit)
            # continue the remaining file
            continue

        # check for parameterset definitions and ignore
        if line.find("parameterset") != -1:
            continue

        # use regex to parse definition
        r = re.search("(\w+).(\w+)\s*=\s*(.*)", line)
        if r is None:
            continue

        if len(r.groups()) != 3:
            logger.info("Failed to parse parameter configuration line :\n   " + line)
            continue

        # find joint name and parameter
        jointIndex = findJointIndex(r.groups()[0])
        if jointIndex is None:
            logger.info(
                "Unknown joint name "
                + r.groups()[0]
                + " in parameter configuration line :\n   "
                + line
            )
            continue

        # find matching channel name
        channelIndex = channelNames.index(r.groups()[1])
        if channelIndex is None:
            logger.info(
                "Unknown joint channel name "
                + r.groups()[1]
                + " in parameter configuration line :\n   "
                + line
            )
            continue

        valueIndex = jointIndex * 7 + channelIndex

        # parse parameters
        parameterList = r.groups()[2].split("+")
        for parameterPair in parameterList:
            parameterPair = parameterPair.strip()

            r = re.search("\s*([+-]?[0-9]*\.?[0-9]*)\s\*\s(\w+)\s*", parameterPair)
            if r is None or len(r.groups()) != 2:
                logger.info(
                    "Malformed parameter description "
                    + parameterPair
                    + " in parameter configuration line :\n   "
                    + line
                )
                continue

            val = float(r.groups()[0])
            parameter = r.groups()[1]

            # check if parameter exists
            parameterIndex = findParameterIndex(parameter)
            if parameterIndex is None:
                # no, create new parameter entry
                parameterIndex = len(paramNames)
                paramNames.append(parameter)
            transform_triplets.append((valueIndex, parameterIndex, val))

    # set (dense) parameter_transformation matrix
    transform = np.zeros(
        (len(channelNames) * len(joint_names), len(paramNames)), dtype=np.float32
    )
    for i, j, v in transform_triplets:
        transform[i, j] = v

    outputs = {
        "model_param_names": paramNames,
        "joint_names": joint_names,
        "channel_names": channelNames,
        "limits": limits,
        "transform": transform,
        "transform_offsets": np.zeros(
            (1, len(channelNames) * len(joint_names)), dtype=np.float32
        ),
    }
    # set number of scales automatically
    if nr_scaling_params is None:
        outputs.update(
            nr_scaling_params=len([s for s in paramNames if s.startswith("scale")])
        )
        outputs.update(
            nr_position_params=len(paramNames) - outputs["nr_scaling_params"]
        )

    return outputs


def compute_normalized_pose_quat(lbs, local_pose, scale):
    """Computes a normalized representation of the pose in quaternion space.
    This is a delta between the per-joint local transformation and the bind state.

    Returns:
        [B, NJ, 4] - normalized rotations
    """
    B = local_pose.shape[0]
    global_pose_zero = th.zeros((B, 6), dtype=th.float32, device=local_pose.device)
    params = lbs.param_transform(th.cat([global_pose_zero, local_pose, scale], axis=-1))
    params = params.reshape(B, -1, 7)
    # applying rotation
    # TODO: what is this?
    rot_quat = Quaternion.batchMul(
        lbs.joint_rotation[np.newaxis], Quaternion.batchFromXYZ(params[:, :, 3:6])
    )
    # removing the bind state
    bind_rot_quat = Quaternion.batchInvert(lbs.bind_state[:, :, 3:7])
    return Quaternion.batchMul(rot_quat, bind_rot_quat)


def compute_root_transform_cuda(lbs_fn, poses, verts=None):
    # NOTE: verts is not really necessary,
    # NOTE: should be used in conjuncation with LBSCuda
    B = poses.shape[0]

    # NOTE: scales are zero (!)
    _, _, _, state_t, state_r, state_s = lbs_fn(poses, vertices=verts)

    bind_r = lbs_fn.joint_state_r_zero[np.newaxis, 1].expand(B, -1, -1)
    bind_t = lbs_fn.joint_state_t_zero[np.newaxis, 1].expand(B, -1)

    R_root = th.matmul(state_r[:, 1], bind_r)
    t_root = (
        th.matmul(state_r[:, 1], (bind_t * state_s[:, 1])[..., np.newaxis])[..., 0]
        + state_t[:, 1]
    )

    return R_root, t_root


def parent_chain(joint_parents, idx, depth):
    if depth == 0 or idx == 0:
        return []
    parent_idx = int(joint_parents[idx])
    return [parent_idx] + parent_chain(joint_parents, parent_idx, depth - 1)


def joint_connectivity(nr_joints, joint_parents, chain_depth=2, pad_ancestors=False):
    children = {j: [] for j in range(nr_joints)}
    parents = {j: None for j in range(nr_joints)}
    ancestors = {j: [] for j in range(nr_joints)}
    for j in range(nr_joints):
        parent_index = int(joint_parents[j])
        ancestors[j] = parent_chain(joint_parents, j, depth=chain_depth)
        if pad_ancestors:
            # adding itself
            ancestors[j] += [j] * (chain_depth - len(ancestors[j]))

        if parent_index == -1:
            continue
        children[parent_index].append(j)
        parents[j] = parent_index

    return {
        "children": children,
        "parents": parents,
        "ancestors": ancestors,
    }


# TODO: merge this with LinearBlendSkinning
class LBSModule(nn.Module):
    def __init__(
        self,
        lbs_model_json,
        lbs_config_dict,
        lbs_template_verts,
        lbs_scale,
        global_scaling,
    ):
        super().__init__()
        self.lbs_fn = LinearBlendSkinning(lbs_model_json, lbs_config_dict)

        self.register_buffer("lbs_scale", th.as_tensor(lbs_scale, dtype=th.float32))
        self.register_buffer(
            "lbs_template_verts", th.as_tensor(lbs_template_verts, dtype=th.float32)
        )
        self.register_buffer("global_scaling", th.as_tensor(global_scaling))

    def pose(self, verts_unposed, motion, template: Optional[th.Tensor] = None):
        scale = self.lbs_scale.expand(motion.shape[0], -1)
        if template is None:
            template = self.lbs_template_verts
        return (
            self.lbs_fn(motion, scale, verts_unposed + template) * self.global_scaling
        )

    def unpose(self, verts, motion):
        B = motion.shape[0]
        scale = self.lbs_scale.expand(B, -1)
        return (
            self.lbs_fn.unpose(motion, scale, verts / self.global_scaling)
            - self.lbs_template_verts
        )

    def template_pose(self, motion):
        B = motion.shape[0]
        scale = self.lbs_scale.expand(B, -1)
        verts = self.lbs_template_verts[np.newaxis].expand(B, -1, -1)
        return self.lbs_fn(motion, scale, verts) * self.global_scaling[np.newaxis]
