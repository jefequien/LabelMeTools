import os
import argparse
import cv2
import json
import numpy as np
import pickle as pkl
from collections import OrderedDict

import utils.data as data
import utils.utils as uts
import utils.eval_utils as eval_uts
import car_models

class CarPoseVisualizer(object):

    def __init__(self, args):
        self.dataset = data.ApolloScape(args)
        self._data_config = self.dataset.get_3d_car_config()
        self.load_car_models()

    def load_car_models(self):
        """Load all the car models
        """
        self.car_models = OrderedDict([])
        for model in car_models.models:
            car_model = '%s/%s.pkl' % (self._data_config['car_model_dir'],
                                       model.name)
            with open(car_model, 'rb') as f:
                self.car_models[model.name] = pkl.load(f, encoding='latin1')
                # fix the inconsistency between obj and pkl
                self.car_models[model.name]['vertices'][:, [0, 1]] *= -1

    def render_car(self, car_pose, image, intrinsic, fill=False):
        car_name = car_models.car_id2name[car_pose['car_id']].name
        car = self.car_models[car_name]
        pose = np.array(car_pose['pose'])

        # Intrinsic dependent on image size
        h, w = image.shape[:2]
        intrinsic = uts.intrinsic_vec_to_mat(intrinsic, (h, w))

        # project 3D points to 2d image plane
        rmat = uts.euler_angles_to_rotation_matrix(pose[:3])
        rvect, _ = cv2.Rodrigues(rmat)
        imgpts, jac = cv2.projectPoints(np.float32(car['vertices']), rvect, pose[3:], intrinsic, distCoeffs=None)

        mask = np.zeros((h,w), dtype='uint8')
        for face in car['faces'] - 1:
            pts = np.array([[imgpts[idx, 0, 0], imgpts[idx, 0, 1]] for idx in face], np.int32)
            pts = pts.reshape((-1, 1, 2))
            if fill:
                cv2.fillPoly(mask, [pts], 255)
            else:
                cv2.polylines(mask, [pts], True, 255, thickness=2)
        return mask

    def get_intrinsic(self, image_name):
        return self.dataset.get_intrinsic(image_name)

    def get_image(self, image_name):
        image_file = '%s/%s.jpg' % (self._data_config['image_dir'], image_name)
        image = cv2.imread(image_file)
        return image

    def get_distance(self, car_pose):
        x = car_pose["pose"][3]
        y = car_pose["pose"][4]
        z = car_pose["pose"][5]
        return np.linalg.norm([x,y,z])

    def showAnn(self, image_name):
        """Show the annotation of a pose file in an image
        Input:
            image_name: the name of image
        Output:
            image_vis: an image show the overlap of car model and image
        """

        car_pose_file = '%s/%s.json' % (self._data_config['pose_dir'], image_name)
        with open(car_pose_file) as f:
            car_poses = json.load(f)

        image = self.get_image(image_name)
        intrinsic = self.get_intrinsic(image_name)

        mask_all = np.zeros(image.shape)
        for i, car_pose in enumerate(car_poses):
            mask = self.render_car(car_pose, image, intrinsic)
            mask_all[:,:,1] += mask

        mask_all = np.array(mask_all * 255 / np.max(mask_all), dtype='uint8')
        image_vis = cv2.addWeighted(image, 1.0, mask_all, 0.5, 0)
        return image_vis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render car instance and convert car labelled files.')
    parser.add_argument('--image_name', default='180116_053947113_Camera_5',
                        help='the dir of ground truth')
    parser.add_argument('--data_dir', default='../apolloscape/',
                        help='the dir of ground truth')
    parser.add_argument('--split', default='sample_data', help='split for visualization')
    args = parser.parse_args()
    print(args)

    visualizer = CarPoseVisualizer(args)
    image_vis = visualizer.showAnn(args.image_name)

    image_vis = cv2.resize(image_vis, (1200, 800))
    cv2.imshow("", image_vis)
    cv2.waitKey(0)

