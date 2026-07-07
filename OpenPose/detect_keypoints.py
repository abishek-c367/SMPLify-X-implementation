"""
Convert rtmlib Wholebody (to_openpose=True) raw output into the exact JSON
schema expected by SMPLify-X's data_parser.read_keypoints().

CONFIRMED from rtmlib's actual source (wholebody.py / format_result):
    raw instance shape = (134, 2)
    [0   : 18 ]  body   (OpenPose-18/COCO order, NO MidHip, NO feet)
    [18  : 24 ]  feet   (LBigToe, LSmallToe, LHeel, RBigToe, RSmallToe, RHeel)
    [24  : 92 ]  face   (68 points - RTMPose gives 68, OpenPose wants 70;
                         format_result pads missing 2 pupils from body eyes)
    [92  : 113]  left hand  (21 points)
    [113 : 134]  right hand (21 points)

Raw body-18 order (standard OpenPose/COCO):
    0 Nose, 1 Neck, 2 RShoulder, 3 RElbow, 4 RWrist,
    5 LShoulder, 6 LElbow, 7 LWrist, 8 RHip, 9 RKnee, 10 RAnkle,
    11 LHip, 12 LKnee, 13 LAnkle, 14 REye, 15 LEye, 16 REar, 17 LEar

read_keypoints() expects true BODY_25 (MidHip inserted, feet folded in),
so this script reassembles raw-18 + feet-6 -> BODY_25 (25 points),
synthesizing MidHip as the midpoint of RHip/LHip.
"""

import os
import os.path as osp
import json

import cv2
import numpy as np
from rtmlib import Wholebody


class OpenPoseKeypointConverter:
    # ---- Raw layout offsets (from actual rtmlib source) ----
    BODY_START, BODY_END = 0, 18
    FEET_START, FEET_END = 18, 24
    FACE_START, FACE_END = 24, 92          # 68 points
    LHAND_START, LHAND_END = 92, 113       # 21 points
    RHAND_START, RHAND_END = 113, 134      # 21 points

    EXPECTED_TOTAL = 134

    # Raw-18 index constants for readability
    NOSE, NECK, RSHO, RELB, RWRI = 0, 1, 2, 3, 4
    LSHO, LELB, LWRI = 5, 6, 7
    RHIP, RKNEE, RANK = 8, 9, 10
    LHIP, LKNEE, LANK = 11, 12, 13
    REYE, LEYE, REAR, LEAR = 14, 15, 16, 17

    # Feet-6 index constants (relative to the feet slice, i.e. 0-5)
    LBIGTOE, LSMALLTOE, LHEEL = 0, 1, 2
    RBIGTOE, RSMALLTOE, RHEEL = 3, 4, 5

    def __init__(self,
                 to_openpose=True,
                 mode="balanced",
                 backend="onnxruntime",
                 device="cuda"):
        self.wholebody = Wholebody(
            to_openpose=to_openpose,
            mode=mode,
            backend=backend,
            device=device,
        )

    @classmethod
    def _to_body25(cls, raw_kp, raw_sc):
        """
        raw_kp: (134, 2), raw_sc: (134,)
        Returns body25_kp (25, 2), body25_sc (25,)
        """
        body18 = raw_kp[cls.BODY_START:cls.BODY_END]
        body18_sc = raw_sc[cls.BODY_START:cls.BODY_END]
        feet = raw_kp[cls.FEET_START:cls.FEET_END]
        feet_sc = raw_sc[cls.FEET_START:cls.FEET_END]

        mid_hip = (body18[cls.RHIP] + body18[cls.LHIP]) / 2.0
        mid_hip_sc = min(body18_sc[cls.RHIP], body18_sc[cls.LHIP])

        order = [
            body18[cls.NOSE], body18[cls.NECK], body18[cls.RSHO], body18[cls.RELB], body18[cls.RWRI],
            body18[cls.LSHO], body18[cls.LELB], body18[cls.LWRI], mid_hip,
            body18[cls.RHIP], body18[cls.RKNEE], body18[cls.RANK],
            body18[cls.LHIP], body18[cls.LKNEE], body18[cls.LANK],
            body18[cls.REYE], body18[cls.LEYE], body18[cls.REAR], body18[cls.LEAR],
            feet[cls.LBIGTOE], feet[cls.LSMALLTOE], feet[cls.LHEEL],
            feet[cls.RBIGTOE], feet[cls.RSMALLTOE], feet[cls.RHEEL],
        ]
        order_sc = [
            body18_sc[cls.NOSE], body18_sc[cls.NECK], body18_sc[cls.RSHO], body18_sc[cls.RELB],
            body18_sc[cls.RWRI], body18_sc[cls.LSHO], body18_sc[cls.LELB], body18_sc[cls.LWRI],
            mid_hip_sc,
            body18_sc[cls.RHIP], body18_sc[cls.RKNEE], body18_sc[cls.RANK],
            body18_sc[cls.LHIP], body18_sc[cls.LKNEE], body18_sc[cls.LANK],
            body18_sc[cls.REYE], body18_sc[cls.LEYE], body18_sc[cls.REAR], body18_sc[cls.LEAR],
            feet_sc[cls.LBIGTOE], feet_sc[cls.LSMALLTOE], feet_sc[cls.LHEEL],
            feet_sc[cls.RBIGTOE], feet_sc[cls.RSMALLTOE], feet_sc[cls.RHEEL],
        ]
        return np.stack(order), np.array(order_sc, dtype=np.float32)

    @staticmethod
    def _flatten(xy, conf):
        flat = np.concatenate([xy, conf[:, None]], axis=1)
        return flat.reshape(-1).tolist()

    def rtmlib_to_openpose_json(self, keypoints, scores, output_path,
                                 verify_total=True):
        keypoints = np.asarray(keypoints, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)

        if keypoints.ndim == 2:
            keypoints = keypoints[None, ...]
            scores = scores[None, ...]

        num_people, K = keypoints.shape[0], keypoints.shape[1]

        if verify_total and K != self.EXPECTED_TOTAL:
            raise ValueError(
                f"Got {K} keypoints per person, expected {self.EXPECTED_TOTAL}. "
                f"rtmlib's raw layout may differ from what this script assumes - "
                f"re-check against your installed rtmlib's wholebody.py source."
            )

        people = []
        for p in range(num_people):
            kp, sc = keypoints[p], scores[p]

            body25_kp, body25_sc = self._to_body25(kp, sc)

            lhand_kp = kp[self.LHAND_START:self.LHAND_END]
            lhand_sc = sc[self.LHAND_START:self.LHAND_END]
            rhand_kp = kp[self.RHAND_START:self.RHAND_END]
            rhand_sc = sc[self.RHAND_START:self.RHAND_END]

            face_kp = kp[self.FACE_START:self.FACE_END]
            face_sc = sc[self.FACE_START:self.FACE_END]
            pad_kp = np.zeros((2, 2), dtype=np.float32)
            pad_sc = np.zeros((2,), dtype=np.float32)
            face_kp = np.concatenate([face_kp, pad_kp], axis=0)
            face_sc = np.concatenate([face_sc, pad_sc], axis=0)

            person_entry = {
                "person_id": [-1],
                "pose_keypoints_2d": self._flatten(body25_kp, body25_sc),
                "face_keypoints_2d": self._flatten(face_kp, face_sc),
                "hand_left_keypoints_2d": self._flatten(lhand_kp, lhand_sc),
                "hand_right_keypoints_2d": self._flatten(rhand_kp, rhand_sc),
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": [],
                "pose_keypoints_3d": [],
            }
            people.append(person_entry)

        out = {"version": 1.3, "people": people}
        with open(output_path, "w") as f:
            json.dump(out, f)

        return out

    def detect_image(self, img_path):
        #given an image path as input returns keypoints and scores as a  tuple()
        #(keypoints,scores)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        return self.wholebody(img)

    def convert_image_to_json(self, img_path, out_json):
        """
        Given img_path to a single image and output path of *.json file, detects 2d keypoints and scores and saves its corresponding 
        json format in output path
        """
        keypoints, scores = self.detect_image(img_path)
        return self.rtmlib_to_openpose_json(keypoints, scores, out_json)

    def detect_n_save(self, data_folder, keypoint_folder="keypoints"):

        '''
        Input : data_folder --> It expects a data folder where it has sub-folders images and keypoints

        Outputs:  Detects points on each of the images inside data_folder/images and saves 
        the detected keypoints and scores in a json format document
        '''
        if not os.path.exists(data_folder):
            raise FileExistsError(data_folder)

        img_folder = os.path.join(data_folder, "images")
        if not os.path.exists(img_folder):
            raise FileNotFoundError(img_folder)

        output_folder = os.path.join(data_folder, keypoint_folder)
        os.makedirs(output_folder, exist_ok=True)

        for i , file_name in enumerate(sorted(os.listdir(img_folder))):
            in_path = os.path.join(img_folder, file_name)
            if not os.path.isfile(in_path):
                continue

            img_fn = osp.splitext(file_name)[0]
            out_json = os.path.join(output_folder, f"{img_fn}_keypoints.json")

            keypoints, scores = self.detect_image(in_path)
            print("keypoints shape of image{i}: ", keypoints.shape)
            self.rtmlib_to_openpose_json(keypoints, scores, out_json)
            print(f"Saved -> {out_json}")


if __name__ == "__main__":

    data_folder = r"C:\Abishek\SMPLify\DATA_FOLDER"
    

    converter = OpenPoseKeypointConverter()
    converter.detect_n_save(data_folder)
    

   
