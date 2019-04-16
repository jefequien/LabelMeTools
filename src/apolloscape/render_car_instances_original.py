"""
    Brief: Demo for render labelled car 3d poses to the image
    Author: wangpeng54@baidu.com
    Date: 2018/6/10
"""

import argparse
import cv2
import car_models
import numpy as np
import json
import pickle as pkl

import utils.data as data
import utils.utils as uts
import utils.eval_utils as eval_uts
import logging

from collections import OrderedDict

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class CarPoseVisualizer(object):
    def __init__(self, args=None, scale=0.4, linewidth=0.):
        """Initializer
        Input:
            scale: whether resize the image in case image is too large
            linewidth: 0 indicates a binary mask, while > 0 indicates
                       using a frame.
        """
        self.dataset = data.ApolloScape(args)
        self._data_config = self.dataset.get_3d_car_config()

        self.MAX_DEPTH = 1e4
        self.MAX_INST_NUM = 100
        h, w = self._data_config['image_size']

        # must round prop to 4 due to renderer requirements
        # this will change the original size a bit, we usually need rescale
        # due to large image size
        self.image_size = np.uint32(uts.round_prop_to(
            np.float32([h * scale, w * scale])))

        self.scale = scale
        self.linewidth = linewidth
        self.colors = np.random.random((self.MAX_INST_NUM, 3)) * 255

    def load_car_models(self):
        """Load all the car models
        """
        self.car_models = OrderedDict([])
        logging.info('loading %d car models' % len(car_models.models))
        for model in car_models.models:
            car_model = '%s/%s.pkl' % (self._data_config['car_model_dir'],
                                       model.name)
            with open(car_model, 'rb') as f:
                self.car_models[model.name] = pkl.load(f, encoding='latin1')
                # fix the inconsistency between obj and pkl
                self.car_models[model.name]['vertices'][:, [0, 1]] *= -1

    def render_car(self, pose, car_name):
        """Render a car instance given pose and car_name
        """
        car = self.car_models[car_name]
        scale = np.ones((3, ))
        pose = np.array(pose)
        vert = uts.project(pose, scale, car['vertices'])
        K = self.intrinsic
        intrinsic = np.float64([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        depth, mask = render.renderMesh_py(np.float64(vert),
                                           np.float64(car['faces']),
                                           intrinsic,
                                           self.image_size[0],
                                           self.image_size[1],
                                           np.float64(self.linewidth))
        return depth, mask

    def render_car_cv2(self,  pose, car_name, image):
        car = self.car_models[car_name]
        pose = np.array(pose)
        # project 3D points to 2d image plane
        rmat = uts.euler_angles_to_rotation_matrix(pose[:3])
        rvect, _ = cv2.Rodrigues(rmat)
        imgpts, jac = cv2.projectPoints(np.float32(car['vertices']), rvect, pose[3:], self.intrinsic, distCoeffs=None)

        mask = np.zeros(image.shape)
        for face in car['faces'] - 1:
            pts = np.array([[imgpts[idx, 0, 0], imgpts[idx, 0, 1]] for idx in face], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(mask, [pts], True, (0, 255, 0))
            # cv2.fillPoly(mask, [pts], (0, 255, 0))
        return mask

    def compute_reproj_sim(self, car_names, out_file=None):
        """Compute the similarity matrix between every pair of cars.
        """
        if out_file is None:
            out_file = './sim_mat.txt'

        sim_mat = np.eye(len(self.car_model))
        for i in range(len(car_names)):
            for j in range(i, len(car_names)):
                name1 = car_names[i][0]
                name2 = car_names[j][0]
                ind_i = self.car_model.keys().index(name1)
                ind_j = self.car_model.keys().index(name2)
                sim_mat[ind_i, ind_j] = self.compute_reproj(name1, name2)
                sim_mat[ind_j, ind_i] = sim_mat[ind_i, ind_j]

        np.savetxt(out_file, sim_mat, fmt='%1.6f')

    def compute_reproj(self, car_name1, car_name2):
        """Compute reprojection error between two cars
        """
        sims = np.zeros(10)
        for i, rot in enumerate(np.linspace(0, np.pi, num=10)):
            pose = np.array([0, rot, 0, 0, 0,5.5])
            depth1, mask1 = self.render_car(pose, car_name1)
            depth2, mask2 = self.render_car(pose, car_name2)
            sims[i] = eval_uts.IOU(mask1, mask2)

        return np.mean(sims)

    def merge_inst(self,
                   depth_in,
                   inst_id,
                   total_mask,
                   total_depth):
        """Merge the prediction of each car instance to a full image
        """

        render_depth = depth_in.copy()
        render_depth[render_depth <= 0] = np.inf
        depth_arr = np.concatenate([render_depth[None, :, :],
                                    total_depth[None, :, :]], axis=0)
        idx = np.argmin(depth_arr, axis=0)

        total_depth = np.amin(depth_arr, axis=0)
        total_mask[idx == 0] = inst_id

        return total_mask, total_depth


    def rescale(self, image, intrinsic):
        """resize the image and intrinsic given a relative scale
        """

        intrinsic_out = uts.intrinsic_vec_to_mat(intrinsic,
                                                 self.image_size)
        hs, ws = self.image_size
        image_out = cv2.resize(image.copy(), (ws, hs))

        return image_out, intrinsic_out


    def showAnn(self, image_name):
        """Show the annotation of a pose file in an image
        Input:
            image_name: the name of image
        Output:
            depth: a rendered depth map of each car
            masks: an instance mask of the label
            image_vis: an image show the overlap of car model and image
        """

        car_pose_file = '%s/%s.json' % (self._data_config['pose_dir'], image_name)
        with open(car_pose_file) as f:
            car_poses = json.load(f)
        image_file = '%s/%s.jpg' % (self._data_config['image_dir'], image_name)
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)[:, :, ::-1]

        intrinsic = self.dataset.get_intrinsic(image_name)
        image, self.intrinsic = self.rescale(image, intrinsic)

        merged_image = image.copy()
        mask_all = np.zeros(image.shape)
        for i, car_pose in enumerate(car_poses):
            car_name = car_models.car_id2name[car_pose['car_id']].name
            mask = self.render_car_cv2(car_pose['pose'], car_name, image)
            mask_all += mask

        mask_all = mask_all * 255 / mask_all.max()

        alpha = 0.5
        output = cv2.addWeighted(image.astype(np.uint8), 1.0, mask_all.astype(np.uint8), 1 - alpha, 0, merged_image)

        cv2.imshow("", output[:,:,::-1])
        cv2.waitKey(0)

class LabelResaver(object):
    """ Resave the raw labeled file to the required json format for evaluation
    """
    #(TODO Peng) Figure out why running pdb it is correct, but segment fault when
    # running
    def __init__(self, args):
        self.visualizer = CarPoseVisualizer(args, scale=0.5)
        self.visualizer.load_car_models()


    def strs_to_mat(self, strs):
        """convert str to numpy matrix
        """
        assert len(strs) == 4
        mat = np.zeros((4, 4))
        for i in range(4):
            mat[i, :] = np.array([np.float32(str_f) for str_f in strs[i].split(' ')])

        return mat


    def read_car_pose(self, file_name):
        """load the labelled car pose
        """
        cars = []
        lines = [line.strip() for line in open(file_name)]
        i = 0
        while i < len(lines):
            car = OrderedDict([])
            line = lines[i].strip()
            if 'Model Name :' in line:
                car_name = line[len('Model Name : '):]
                car['car_id'] = car_models.car_name2id[car_name].id
                pose = self.strs_to_mat(lines[i + 2: i + 6])
                pose[:3, 3] = pose[:3, 3] / 100.0 # convert cm to meter
                rot = uts.rotation_matrix_to_euler_angles(
                        pose[:3, :3], check=False)
                trans = pose[:3, 3].flatten()
                pose = np.hstack([rot, trans])
                car['pose'] = pose
                i += 6
                cars.append(car)
            else:
                i += 1

        return cars


    def convert(self, pose_file_in, pose_file_out):
        """ Convert the raw labelled file to required json format
        Input:
            file_name: str filename
        """
        car_poses = self.read_car_pose(pose_file_in)
        car_num = len(car_poses)
        MAX_DEPTH = self.visualizer.MAX_DEPTH
        image_size = self.visualizer.image_size
        intrinsic = self.visualizer.dataset.get_intrinsic(
                pose_file_in, 'Camera_5')
        self.visualizer.intrinsic = uts.intrinsic_vec_to_mat(intrinsic,
                image_size)
        self.depth = MAX_DEPTH * np.ones(image_size)
        self.mask = np.zeros(self.depth.shape)
        vis_rate = np.zeros(car_num)

        for i, car_pose in enumerate(car_poses):
            car_name = car_models.car_id2name[car_pose['car_id']].name
            depth, mask = self.visualizer.render_car(car_pose['pose'], car_name)
            self.mask, self.depth = self.visualizer.merge_inst(
                depth, i + 1, self.mask, self.depth)
            vis_rate[i] = np.float32(np.sum(mask == (i + 1))) / (
                    np.float32(np.sum(mask)) + np.spacing(1))

        keep_idx = []
        for i, car_pose in enumerate(car_poses):
            area = np.round(np.float32(np.sum(
                self.mask == (i + 1))) / (self.visualizer.scale ** 2))
            if area > 1:
                keep_idx.append(i)

            car_pose['pose'] = car_pose['pose'].tolist()
            car_pose['area'] = int(area)
            car_pose['visible_rate'] = float(vis_rate[i])
            keep_idx.append(i)

        car_poses = [car_poses[idx] for idx in keep_idx]
        with open(pose_file_out, 'w') as f:
            json.dump(car_poses, f, sort_keys=True, indent=4,
                    ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render car instance and convert car labelled files.')
    parser.add_argument('--image_name', default='180116_053947113_Camera_5',
                        help='the dir of ground truth')
    parser.add_argument('--data_dir', default='../apolloscape/3d_car_instance_sample/',
                        help='the dir of ground truth')
    parser.add_argument('--split', default='sample_data', help='split for visualization')
    args = parser.parse_args()
    assert args.image_name

    if False:
        # (TODO) Debugging correct but segment fault when running. Useless for visualization
        print('Test converter')
        pose_file_in = './test_files/%s.poses' % args.image_name
        pose_file_out = './test_files/%s.json' % args.image_name
        label_resaver = LabelResaver(args)
        label_resaver.convert(pose_file_in, pose_file_out)

    print('Test visualizer')
    visualizer = CarPoseVisualizer(args)
    visualizer.load_car_models()
    visualizer.showAnn(args.image_name)



