import cv2
from collections import defaultdict

def load_mot_annotations(mot_path):
    """
    Trả về:
    annotations[frame_id] = list of (track_id, x1, y1, x2, y2)
    """
    annotations = defaultdict(list)

    with open(mot_path, "r") as f:
        for line in f:
            items = line.strip().split(",")
            if len(items) < 6:
                continue

            frame_id = int(items[0])
            track_id = int(items[1])
            x, y, w, h = map(float, items[2:6])

            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w)
            y2 = int(y + h)

            annotations[frame_id].append(
                (track_id, x1, y1, x2, y2)
            )

    return annotations

def visualize_mot_video(
    video_path,
    mot_path,
    output_path="vis_output.mp4",
    show_id=True
):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Không mở được video"

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    annotations = load_mot_annotations(mot_path)

    frame_id = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in annotations:
            for track_id, x1, y1, x2, y2 in annotations[frame_id]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if show_id:
                    cv2.putText(
                        frame,
                        f"ID:{track_id}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()
    print(f"✅ Saved visualization to {output_path}")

visualize_mot_video(
    video_path="data/videos/cam01.mp4",
    mot_path="tracking_results_2/camera1/camera_1.txt",
    output_path="data/video/camera_1.mp4"
)
