import cv2


def get_vid_metadata(vid_path: str):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise IOError(f'Could not open video: {vid_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError('FPS is 0, cannot compute frame index')
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if frame_count == 0:
        raise ValueError('Frame count is 0')
    return cap, fps, int(frame_count)


def get_frame(idx: int, cap: cv2.VideoCapture):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise IOError(f'Could not read frame {idx}')
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
