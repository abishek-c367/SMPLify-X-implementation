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
import numpy as np
import cv2
from rtmlib import Wholebody

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


def _to_body25(raw_kp, raw_sc):
    """
    raw_kp: (134, 2), raw_sc: (134,)
    Returns body25_kp (25, 2), body25_sc (25,)
    """
    body18 = raw_kp[BODY_START:BODY_END]
    body18_sc = raw_sc[BODY_START:BODY_END]
    feet = raw_kp[FEET_START:FEET_END]
    feet_sc = raw_sc[FEET_START:FEET_END]

    mid_hip = (body18[RHIP] + body18[LHIP]) / 2.0
    mid_hip_sc = min(body18_sc[RHIP], body18_sc[LHIP])  # conservative

    order = [
        body18[NOSE], body18[NECK], body18[RSHO], body18[RELB], body18[RWRI],
        body18[LSHO], body18[LELB], body18[LWRI], mid_hip,
        body18[RHIP], body18[RKNEE], body18[RANK],
        body18[LHIP], body18[LKNEE], body18[LANK],
        body18[REYE], body18[LEYE], body18[REAR], body18[LEAR],
        feet[LBIGTOE], feet[LSMALLTOE], feet[LHEEL],
        feet[RBIGTOE], feet[RSMALLTOE], feet[RHEEL],
    ]
    order_sc = [
        body18_sc[NOSE], body18_sc[NECK], body18_sc[RSHO], body18_sc[RELB],
        body18_sc[RWRI], body18_sc[LSHO], body18_sc[LELB], body18_sc[LWRI],
        mid_hip_sc,
        body18_sc[RHIP], body18_sc[RKNEE], body18_sc[RANK],
        body18_sc[LHIP], body18_sc[LKNEE], body18_sc[LANK],
        body18_sc[REYE], body18_sc[LEYE], body18_sc[REAR], body18_sc[LEAR],
        feet_sc[LBIGTOE], feet_sc[LSMALLTOE], feet_sc[LHEEL],
        feet_sc[RBIGTOE], feet_sc[RSMALLTOE], feet_sc[RHEEL],
    ]
    return np.stack(order), np.array(order_sc, dtype=np.float32)


def _flatten(xy, conf):
    flat = np.concatenate([xy, conf[:, None]], axis=1)
    return flat.reshape(-1).tolist()


def rtmlib_to_openpose_json(keypoints, scores, output_path,
                             verify_total=True):
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    if keypoints.ndim == 2:
        keypoints = keypoints[None, ...]
        scores = scores[None, ...]

    num_people, K = keypoints.shape[0], keypoints.shape[1]

    if verify_total and K != EXPECTED_TOTAL:
        raise ValueError(
            f"Got {K} keypoints per person, expected {EXPECTED_TOTAL}. "
            f"rtmlib's raw layout may differ from what this script assumes - "
            f"re-check against your installed rtmlib's wholebody.py source."
        )

    people = []
    for p in range(num_people):
        kp, sc = keypoints[p], scores[p]

        body25_kp, body25_sc = _to_body25(kp, sc)

        lhand_kp = kp[LHAND_START:LHAND_END]
        lhand_sc = sc[LHAND_START:LHAND_END]
        rhand_kp = kp[RHAND_START:RHAND_END]
        rhand_sc = sc[RHAND_START:RHAND_END]

        face_kp = kp[FACE_START:FACE_END]      # 68 pts
        face_sc = sc[FACE_START:FACE_END]
        # Pad to 70 so read_keypoints()'s fixed slice [17:17+51] still lands
        # correctly (padding sits at indices 68-69, past the used range).
        pad_kp = np.zeros((2, 2), dtype=np.float32)
        pad_sc = np.zeros((2,), dtype=np.float32)
        face_kp = np.concatenate([face_kp, pad_kp], axis=0)
        face_sc = np.concatenate([face_sc, pad_sc], axis=0)

        person_entry = {
            "person_id": [-1],
            "pose_keypoints_2d": _flatten(body25_kp, body25_sc),
            "face_keypoints_2d": _flatten(face_kp, face_sc),
            "hand_left_keypoints_2d": _flatten(lhand_kp, lhand_sc),
            "hand_right_keypoints_2d": _flatten(rhand_kp, rhand_sc),
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

def detect_n_save(data_folder):

    if  not os.path.exists(data_folder):
        raise FileExistsError(data_folder)
    img_folder = os.path.join(data_folder, 'images')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    img_path = [os.path.join(img_folder,item) for item in os.listdir(img_folder)]
    wholebody = Wholebody(
        to_openpose=True,
        mode="balanced",
        backend="onnxruntime",
        device="cuda",
    )
    for img in img_path:
        img_fn = osp.splitext(osp.basename(img))[0]
        keypoint_folder = os.path.join(data_folder, "keypoints")
        if not os.path.exists(keypoint_folder):
            os.makedirs(keypoint_folder)
        out_json = os.path.join(keypoint_folder, f"{img_fn}_keypoints.json")
        img = cv2.imread(img)
        keypoints, scores = wholebody(img)
        print("keypoints shape:", keypoints.shape)  # should be (1, 134, 2)

        rtmlib_to_openpose_json(keypoints, scores, out_json)
        print(f"Saved -> {out_json}")


            

     

if __name__ == "__main__":
    import os.path as osp
    import cv2
    from rtmlib import Wholebody

    img_path = r"C:\Abishek\SMPLify\DATA_FOLDER\images\1.jpg"          # <-- change this
    data_folder = r"C:\Abishek\SMPLify\DATA_FOLDER"           # <-- your SMPLify-X data folder
    img_fn = osp.splitext(osp.basename(img_path))[0]
    out_json = osp.join(data_folder, "keypoints", f"{img_fn}_keypoints.json")

    img = cv2.imread(img_path)

    wholebody = Wholebody(
        to_openpose=True,
        mode="balanced",
        backend="onnxruntime",
        device="cuda",
    )
    keypoints, scores = wholebody(img)

    print("keypoints shape:", keypoints.shape)  # should be (1, 134, 2)

    rtmlib_to_openpose_json(keypoints, scores, out_json)
    print(f"Saved -> {out_json}")

    # ---- Round-trip verification against SMPLify-X's own loader ----
    import sys
    import os
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sys.path.insert(0, repo_root)
    from smplifyx.data_parser import read_keypoints

    result = read_keypoints(out_json, use_hands=True, use_face=True)
    print("Loaded back:", len(result.keypoints), result.keypoints[0].shape)
    # Expect: 1 person, shape (25 + 21 + 21 + 51, 3) = (118, 3)