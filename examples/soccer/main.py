import argparse
from enum import Enum
from typing import Iterator, List

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
    draw_paths_on_pitch,
)
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(
    PARENT_DIR, "data/football-player-detection.pt"
)
PITCH_DETECTION_MODEL_PATH = os.path.join(
    PARENT_DIR, "data/football-pitch-detection.pt"
)
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, "data/football-ball-detection.pt")

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ["#FF1493", "#00BFFF", "#FF6347", "#FFD700"]
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex("#FFFFFF"),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex("#FF1493"),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex("#FF1493"),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(COLORS), thickness=2)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS), thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex("#FFFFFF"),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex("#FFFFFF"),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """

    PITCH_DETECTION = "PITCH_DETECTION"
    PLAYER_DETECTION = "PLAYER_DETECTION"
    BALL_DETECTION = "BALL_DETECTION"
    PLAYER_TRACKING = "PLAYER_TRACKING"
    TEAM_CLASSIFICATION = "TEAM_CLASSIFICATION"
    RADAR = "RADAR"
    BALL_TRAJECTORY = "BALL_TRAJECTORY"


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections, players_team_id: np.array, goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_ball_trajectory(
    keypoints: sv.KeyPoints, ball_trajectory: List[np.ndarray]
) -> np.ndarray:
    """Render only ball trajectory on pitch (no players, no Voronoi)"""
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32),
    )

    radar = draw_pitch(config=CONFIG)

    # Draw ball trajectory only
    if ball_trajectory is not None and len(ball_trajectory) > 0:
        ball_xy = np.array([pos for pos in ball_trajectory if pos.size > 0])
        if len(ball_xy) > 0:
            transformed_ball_xy = transformer.transform_points(points=ball_xy)
            radar = draw_paths_on_pitch(
                config=CONFIG,
                paths=[transformed_ball_xy],
                color=sv.Color.WHITE,
                thickness=3,
                pitch=radar,
            )

    return radar


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray,
    ball_trajectory_world: List[np.ndarray] = None,
) -> np.ndarray:
    """Render radar with ball trajectory already in world coordinates"""
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32),
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)

    # Draw ball trajectory if available (already in world coordinates)
    if ball_trajectory_world is not None and len(ball_trajectory_world) > 0:
        radar = draw_paths_on_pitch(
            config=CONFIG,
            paths=[ball_trajectory_world],
            color=sv.Color.WHITE,
            thickness=3,
            pitch=radar,
        )

    radar = draw_points_on_pitch(
        config=CONFIG,
        xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]),
        radius=20,
        pitch=radar,
    )
    radar = draw_points_on_pitch(
        config=CONFIG,
        xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]),
        radius=20,
        pitch=radar,
    )
    radar = draw_points_on_pitch(
        config=CONFIG,
        xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]),
        radius=20,
        pitch=radar,
    )
    radar = draw_points_on_pitch(
        config=CONFIG,
        xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]),
        radius=20,
        pitch=radar,
    )
    return radar


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels
        )
        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels
        )
        yield annotated_frame


def run_team_classification(
    source_video_path: str, device: str
) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE
    )

    crops = []
    for frame in tqdm(frame_generator, desc="collecting crops"):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers
        )

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist()
            + goalkeepers_team_id.tolist()
            + [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup
        )
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup
        )
        yield annotated_frame


def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    from collections import deque

    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)

    MAXLEN = 10  # Increased for smoother transformation
    BALL_ID = 0

    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE
    )

    crops = []
    for frame in tqdm(frame_generator, desc="collecting crops"):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    M = deque(maxlen=MAXLEN)
    ball_tracker = BallTracker(buffer_size=20)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    # Ball detection callback for InferenceSlicer
    def ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    ball_slicer = sv.InferenceSlicer(
        callback=ball_callback,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        key_points = sv.KeyPoints.from_ultralytics(result)

        # Apply ViewTransformer smoothing
        filter = key_points.confidence[0] > 0.5
        frame_reference_points = key_points.xy[0][filter]
        pitch_reference_points = np.array(CONFIG.vertices)[filter]

        transformer = ViewTransformer(
            source=frame_reference_points, target=pitch_reference_points
        )
        M.append(transformer.m)
        transformer.m = np.mean(np.array(M), axis=0)

        # Detect ball with smoothing
        ball_detections = ball_slicer(frame).with_nms(threshold=0.1)
        ball_detections = ball_detections[ball_detections.class_id == BALL_ID]
        ball_detections = ball_tracker.update(ball_detections)

        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers
        )

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist()
            + goalkeepers_team_id.tolist()
            + [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup
        )
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup
        )

        h, w, _ = frame.shape
        # Transform player positions with smoothed transformer
        xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_xy = transformer.transform_points(points=xy)

        radar = draw_pitch(config=CONFIG)

        # Draw ball on radar if detected
        if len(ball_detections) > 0:
            ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            transformed_ball_xy = transformer.transform_points(points=ball_xy)
            radar = draw_points_on_pitch(
                config=CONFIG,
                xy=transformed_ball_xy,
                face_color=sv.Color.WHITE,
                radius=15,
                pitch=radar,
            )

        # Draw players on radar
        radar = draw_points_on_pitch(
            config=CONFIG,
            xy=transformed_xy[color_lookup == 0],
            face_color=sv.Color.from_hex(COLORS[0]),
            radius=20,
            pitch=radar,
        )
        radar = draw_points_on_pitch(
            config=CONFIG,
            xy=transformed_xy[color_lookup == 1],
            face_color=sv.Color.from_hex(COLORS[1]),
            radius=20,
            pitch=radar,
        )
        radar = draw_points_on_pitch(
            config=CONFIG,
            xy=transformed_xy[color_lookup == 2],
            face_color=sv.Color.from_hex(COLORS[2]),
            radius=20,
            pitch=radar,
        )
        radar = draw_points_on_pitch(
            config=CONFIG,
            xy=transformed_xy[color_lookup == 3],
            face_color=sv.Color.from_hex(COLORS[3]),
            radius=20,
            pitch=radar,
        )

        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2, y=h - radar_h, width=radar_w, height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        yield annotated_frame


def run_ball_trajectory(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run ball trajectory visualization - EXACT implementation from Roboflow notebook.
    Shows ONLY ball path on pitch, yields video frames.
    """
    from collections import deque
    from typing import Union

    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)

    BALL_ID = 0
    MAXLEN = 5
    MAX_DISTANCE_THRESHOLD = 500

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    path_raw = []
    M = deque(maxlen=MAXLEN)
    frames_list = []

    def replace_outliers_based_on_distance(
        positions: list[np.ndarray], distance_threshold: float
    ) -> list[np.ndarray]:
        last_valid_position: Union[np.ndarray, None] = None
        cleaned_positions: list[np.ndarray] = []

        for position in positions:
            if len(position) == 0:
                cleaned_positions.append(position)
            else:
                if last_valid_position is None:
                    cleaned_positions.append(position)
                    last_valid_position = position
                else:
                    distance = np.linalg.norm(position - last_valid_position)
                    if distance > distance_threshold:
                        cleaned_positions.append(np.array([], dtype=np.float64))
                    else:
                        cleaned_positions.append(position)
                        last_valid_position = position

        return cleaned_positions

    # First pass: collect all ball positions and frames
    for frame in frame_generator:
        frames_list.append(frame)

        # Detect ball
        result = ball_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        # Detect pitch keypoints
        result = pitch_detection_model(frame, verbose=False)[0]
        key_points = sv.KeyPoints.from_ultralytics(result)

        filter = key_points.confidence[0] > 0.5
        frame_reference_points = key_points.xy[0][filter]
        pitch_reference_points = np.array(CONFIG.vertices)[filter]

        transformer = ViewTransformer(
            source=frame_reference_points, target=pitch_reference_points
        )
        M.append(transformer.m)
        transformer.m = np.mean(np.array(M), axis=0)

        frame_ball_xy = ball_detections.get_anchors_coordinates(
            sv.Position.BOTTOM_CENTER
        )
        pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

        path_raw.append(pitch_ball_xy)

    # Process path
    path = [
        np.empty((0, 2), dtype=np.float32) if coordinates.shape[0] >= 2 else coordinates
        for coordinates in path_raw
    ]

    path = [coordinates.flatten() for coordinates in path]
    path = replace_outliers_based_on_distance(path, MAX_DISTANCE_THRESHOLD)

    # Second pass: generate video frames with cumulative trajectory
    for frame_idx, frame in enumerate(frames_list):
        # Get cumulative path up to current frame
        cumulative_path = path[: frame_idx + 1]

        # Draw trajectory on pitch
        pitch_frame = draw_pitch(CONFIG)
        pitch_frame = draw_paths_on_pitch(
            config=CONFIG,
            paths=[cumulative_path],
            color=sv.Color.WHITE,
            thickness=3,
            pitch=pitch_frame,
        )

        # Overlay on original frame
        h, w, _ = frame.shape
        pitch_resized = sv.resize_image(pitch_frame, (w // 2, h // 2))
        pitch_h, pitch_w, _ = pitch_resized.shape
        rect = sv.Rect(
            x=w // 2 - pitch_w // 2, y=h - pitch_h, width=pitch_w, height=pitch_h
        )
        annotated_frame = sv.draw_image(
            frame.copy(), pitch_resized, opacity=0.5, rect=rect
        )

        yield annotated_frame


def main(
    source_video_path: str,
    target_video_path: str,
    device: str,
    mode: Mode,
    show_display: bool = True,
) -> None:
    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device
        )
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device
        )
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device
        )
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device
        )
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device
        )
    elif mode == Mode.RADAR:
        frame_generator = run_radar(source_video_path=source_video_path, device=device)
    elif mode == Mode.BALL_TRAJECTORY:
        frame_generator = run_ball_trajectory(
            source_video_path=source_video_path, device=device
        )
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    total_frames = video_info.total_frames

    print(f"📹 Video Info:")
    print(f"   - Total frames: {total_frames}")
    print(f"   - FPS: {video_info.fps}")
    print(f"   - Duration: {total_frames / video_info.fps:.1f}s")
    print(f"   - Resolution: {video_info.width}x{video_info.height}")
    print(f"   - Device: {device.upper()}")
    print(f"   - Mode: {mode.value}")
    print()

    with sv.VideoSink(target_video_path, video_info) as sink:
        progress_bar = tqdm(
            total=total_frames,
            desc="Processing",
            unit="frame",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for frame in frame_generator:
            sink.write_frame(frame)
            progress_bar.update(1)

            if show_display:
                try:
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\n⚠️  Stopped by user (Q pressed)")
                        break
                except cv2.error:
                    # GUI not available (headless OpenCV), continue without display
                    pass

        progress_bar.close()
        print(f"\n✅ Done! Output saved to: {target_video_path}")
        try:
            cv2.destroyAllWindows()
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--source_video_path", type=str, required=True)
    parser.add_argument("--target_video_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=Mode, default=Mode.PLAYER_DETECTION)
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable GUI display (faster processing, no risk of accidental Q press)",
    )
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
        show_display=not args.no_display,
    )
