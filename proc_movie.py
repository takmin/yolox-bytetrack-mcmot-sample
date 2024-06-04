#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2
import numpy as np

from yolox.yolox_onnx import YoloxONNX
from bytetrack.mc_bytetrack import MultiClassByteTrack


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--csv", type=str, default="output.csv")

    # YOLOX parameters
    parser.add_argument(
        "--yolox_model",
        type=str,
        default='model/yolox_nano.onnx',
    )
    parser.add_argument(
        '--input_shape',
        type=str,
        default="416,416",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.3,
        help='Class confidence',
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.45,
        help='NMS IoU threshold',
    )
    parser.add_argument(
        '--nms_score_th',
        type=float,
        default=0.1,
        help='NMS Score threshold',
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )

    # motpy parameters
    parser.add_argument(
        "--track_thresh",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--track_buffer",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--min_box_area",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--mot20",
        action="store_true",
    )

    args = parser.parse_args()

    return args


class dict_dot_notation(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def main():
    # 引数解析 #################################################################
    args = get_args()

    if args.input is None:
        print("\"--input\" option is MUST.")
        return
    
    cap_device = args.input

    # YOLOX parameters
    model_path = args.yolox_model
    input_shape = tuple(map(int, args.input_shape.split(',')))
    score_th = args.score_th
    nms_th = args.nms_th
    nms_score_th = args.nms_score_th
    with_p6 = args.with_p6

    # ByteTrack parameters
    track_thresh = args.track_thresh
    track_buffer = args.track_buffer
    match_thresh = args.match_thresh
    min_box_area = args.min_box_area
    mot20 = args.mot20

    # カメラ準備 ###############################################################
    cap = cv2.VideoCapture(cap_device)
    
    # 入力ビデオの解像度とフレームレートを取得
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_diff = 1.0 / cap_fps
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriterオブジェクトを作成
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 注意: 'mp4v'は小文字
    movie_out = None
    if(args.output is not None):
        movie_out = cv2.VideoWriter(args.output, fourcc, cap_fps, (frame_width, frame_height))
    timestamp = 0.0

    # 出力CSVファイル ###############################################
    csv_file = open(args.csv, "w")
    csv_file.write("timestamp,track_id,class_id,class_name,x1,y1,x2,y2\n")

    # モデルロード #############################################################
    yolox = YoloxONNX(
        model_path=model_path,
        input_shape=input_shape,
        class_score_th=score_th,
        nms_th=nms_th,
        nms_score_th=nms_score_th,
        with_p6=with_p6,
        providers=['CPUExecutionProvider'],
    )

    # ByteTrackerインスタンス生成
    tracker = MultiClassByteTrack(
        fps=cap_fps,
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        min_box_area=min_box_area,
        mot20=mot20,
    )

    # トラッキングID保持用変数
    track_id_dict = {}

    # COCOクラスリスト読み込み
    with open('coco_classes.txt', 'rt') as f:
        coco_classes = f.read().rstrip('\n').split('\n')

    while True:
        print(timestamp)
        start_time = time.time()

        # カメラキャプチャ ################################################
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # 推論実施 ########################################################
        # Object Detection
        bboxes, scores, class_ids = yolox.inference(frame)

        # Multi Object Tracking
        t_ids, t_bboxes, t_scores, t_class_ids = tracker(
            frame,
            bboxes,
            scores,
            class_ids,
        )

        # トラッキングIDと連番の紐付け
        for trakcer_id, bbox in zip(t_ids, bboxes):
            if trakcer_id not in track_id_dict:
                new_id = len(track_id_dict)
                track_id_dict[trakcer_id] = new_id

        elapsed_time = time.time() - start_time

        # 画面反映 #########################################################
        if(movie_out is not None):
            # デバッグ描画
            debug_image = draw_debug(
                debug_image,
                elapsed_time,
                score_th,
                t_ids,
                t_bboxes,
                t_scores,
                t_class_ids,
                track_id_dict,
                coco_classes,
            )
            movie_out.write(debug_image)

        # CSV出力 ######################################
        write_csv(csv_file, timestamp, score_th, t_ids, t_bboxes, t_scores, t_class_ids, coco_classes)
        timestamp += frame_diff

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()


def get_id_color(index):
    temp_index = abs(int(index)) * 3
    color = ((37 * temp_index) % 255, (17 * temp_index) % 255,
             (29 * temp_index) % 255)
    return color


def draw_debug(
    image,
    elapsed_time,
    score_th,
    trakcer_ids,
    bboxes,
    scores,
    class_ids,
    track_id_dict,
    coco_classes,
):
    debug_image = copy.deepcopy(image)

    for tracker_id, bbox, score, class_id in zip(trakcer_ids, bboxes, scores,
                                                 class_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if score_th > score:
            continue

        color = get_id_color(int(track_id_dict[tracker_id]))

        # バウンディングボックス
        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            color,
            thickness=2,
        )

        # トラックID、スコア
        score_txt = str(round(score, 2))
        text = 'Track ID:%s(%s)' % (int(track_id_dict[tracker_id]), score_txt)
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness=2,
        )
        # クラスID
        text = 'Class ID:%s(%s)' % (class_id, coco_classes[class_id])
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness=2,
        )

    # 推論時間
    text = 'Elapsed time:' + '%.0f' % (elapsed_time * 1000)
    text = text + 'ms'
    debug_image = cv2.putText(
        debug_image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )

    return debug_image


def write_csv(
    csv_file,
    time_stamp,
    score_th,
    tracker_ids,
    bboxes,
    scores,
    class_ids,
    coco_classes,
):
    for tracker_id, bbox, score, class_id in zip(tracker_ids, bboxes, scores,
                                                 class_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if score_th > score:
            continue

        csv_file.write(str(time_stamp) + "," + str(tracker_id) + "," \
                    + str(class_id) + "," + coco_classes[class_id] + ","\
                    + str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + "\n")


if __name__ == '__main__':
    main()
