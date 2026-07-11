# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import logging

import numpy as np


class MeshViewer(object):

    def __init__(self, width=1200, height=800,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 registered_keys=None):
        super(MeshViewer, self).__init__()

        if registered_keys is None:
            registered_keys = dict()

        self.body_color = body_color
        self.viewer = None
        self.scene = None
        self._logger = logging.getLogger(__name__)

        try:
            import trimesh
            import pyrender
        except Exception as exc:
            self._logger.warning(
                'Unable to import pyrender/trimesh for visualization: %s. '
                'Disabling mesh visualization.', exc)
            return

        self.mat_constructor = pyrender.MetallicRoughnessMaterial
        self.mesh_constructor = trimesh.Trimesh
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh
        self.transf = trimesh.transformations.rotation_matrix

        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0],
                                    ambient_light=(0.3, 0.3, 0.3))

        pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0,
                                        aspectRatio=float(width) / height)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 3])
        self.scene.add(pc, pose=camera_pose)

        try:
            self.viewer = pyrender.Viewer(self.scene,
                                          use_raymond_lighting=True,
                                          viewport_size=(width, height),
                                          cull_faces=False,
                                          run_in_thread=True,
                                          registered_keys=registered_keys)
        except Exception as exc:
            self._logger.warning(
                'Unable to initialize pyrender viewer (%s). '
                'Disabling mesh visualization.', exc)
            self.viewer = None
            self.scene = None

    def is_active(self):
        return self.viewer is not None and getattr(self.viewer, 'is_active', False)

    def close_viewer(self):
        if self.viewer is not None and getattr(self.viewer, 'is_active', False):
            self.viewer.close_external()

    def create_mesh(self, vertices, faces, color=(0.3, 0.3, 0.3, 1.0),
                    wireframe=False):

        material = self.mat_constructor(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=color)

        mesh = self.mesh_constructor(vertices, faces)

        rot = self.transf(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        return self.trimesh_to_pymesh(mesh, material=material)

    def update_mesh(self, vertices, faces):
        if not self.is_active():
            return

        render_lock = getattr(self.viewer, 'render_lock', None)
        if render_lock is None:
            return

        render_lock.acquire()
        try:
            for node in self.scene.get_nodes():
                if node.name == 'body_mesh':
                    self.scene.remove_node(node)
                    break

            body_mesh = self.create_mesh(
                vertices, faces, color=self.body_color)
            self.scene.add(body_mesh, name='body_mesh')
        finally:
            render_lock.release()
