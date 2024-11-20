import os
from typing import Tuple, List, Optional

import cv2
import numpy as np
import tqdm

from common.image_tools.comparison import prepare_image_for_comparison, compare_images


def video_to_frames(video_fullname: str, output_fullpath: Optional[str] = None, comparison_size: int = 128, similarity_thresh: float = 0.8, show_parsed: bool = False) -> Tuple[str, List[str]]:
    """

    :param video_fullname:
    :param output_fullpath:
    :param comparison_size:
    :param similarity_thresh:
    :param show_parsed:
    :return: output_fullpath, saved_frames_names
    """
    vidcap = cv2.VideoCapture(video_fullname)

    if not output_fullpath:
        video_path, video_name = os.path.split(video_fullname)
        output_path = os.path.splitext(video_name)[0] + '_frames'
        output_fullpath = os.path.join(video_path, output_path)

    print("output_fullpath: {}".format(output_fullpath))
    os.makedirs(output_fullpath, exist_ok=True)

    success, current_frame = vidcap.read()
    print("Video parsing success: {}".format(success))

    frame_count = 0
    saved_frames_names = []
    
    last_frame_prepared = np.zeros((comparison_size, comparison_size))
    progress_bar = tqdm.tqdm()
    
    while success:
        success, current_frame = vidcap.read()

        progress_bar.update(1)

        if current_frame is not None:
            output_image_name = "frame_{:03d}.jpg".format(frame_count)
            output_image_fullname = os.path.join(output_fullpath, output_image_name)

            current_frame_prepared = prepare_image_for_comparison(current_frame, comparison_size)
            frame_similarity = compare_images(last_frame_prepared, current_frame_prepared)

            if frame_similarity < similarity_thresh:
                cv2.imwrite(output_image_fullname, current_frame)
                last_frame_prepared = current_frame_prepared.copy()
                saved_frames_names.append(output_image_name)

                if show_parsed:
                    cv2.imshow("Parsed frame", current_frame)

            frame_count += 1

    if not success and frame_count > 1:
        print()
        print("video {} converted {} frames to folder {}".format(video_fullname, len(saved_frames_names), output_fullpath))

    vidcap.release()
    cv2.destroyAllWindows()

    return output_fullpath, saved_frames_names