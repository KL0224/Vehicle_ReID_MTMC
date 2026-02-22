# tools/convert_mot.py
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


def load_mapping(mapping_json: Path) -> Dict[Tuple[str, int], int]:
    data = json.loads(mapping_json.read_text(encoding="utf-8"))
    m: Dict[Tuple[str, int], int] = {}
    for r in data:
        cam = str(r["camera_id"])
        tid = int(r["local_track_id"])
        gid = int(r["global_id"])
        m[(cam, tid)] = gid
    return m


def load_tracks(tracks_json: Path) -> List[dict]:
    text = tracks_json.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("tracks_json is empty")

    data: Any = json.loads(text)

    # Accept: {"tracks": [...]}
    if isinstance(data, dict) and isinstance(data.get("tracks"), list):
        return data["tracks"]

    # Accept: direct list (already flattened or list of tracks)
    if isinstance(data, list):
        return data

    raise ValueError("Unsupported tracks format: expected a list or a dict with key tracks")


def _xyxy_to_xywh(bbox: List[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, bbox)
    return x1, y1, (x2 - x1), (y2 - y1)


def to_mot_lines(
    tracks: List[dict],
    mapping: Dict[Tuple[str, int], int],
    camera_id: str,
    default_conf: float = 1.0,
    default_class: int = -1,
    default_visibility: float = -1.0,
    keep_unmapped: bool = False,
) -> List[str]:
    # Flatten to (frame_id, local_track_id, bbox_xywh, conf)
    rows: List[Tuple[int, int, Tuple[float, float, float, float], float]] = []

    for t in tracks:
        if not isinstance(t, dict):
            continue

        local_track_id = t.get("track_id", t.get("local_track_id"))
        if local_track_id is None:
            continue
        local_track_id = int(local_track_id)

        dets = t.get("detections")
        if isinstance(dets, list):
            for d in dets:
                if not isinstance(d, dict):
                    continue
                frame_id = d.get("frame_id")
                bbox = d.get("bbox")
                if frame_id is None or not isinstance(bbox, list) or len(bbox) != 4:
                    continue

                # bbox in file is xyxy -> convert to xywh
                x, y, w, h = _xyxy_to_xywh(bbox)
                conf = d.get("confidence", d.get("conf", default_conf))
                rows.append((int(frame_id), local_track_id, (x, y, w, h), float(conf)))
            continue

        # Optional: already-flat row support: {"frame_id":..,"local_track_id"/"track_id":..,"bbox":[...]}
        frame_id = t.get("frame_id")
        bbox = t.get("bbox")
        if frame_id is not None and isinstance(bbox, list) and len(bbox) == 4:
            # Heuristic: if looks like xyxy (x2>x1 and y2>y1) then convert, else assume xywh
            x1, y1, x2, y2 = map(float, bbox)
            if x2 > x1 and y2 > y1:
                x, y, w, h = _xyxy_to_xywh(bbox)
            else:
                x, y, w, h = x1, y1, x2, y2
            conf = t.get("confidence", t.get("conf", default_conf))
            rows.append((int(frame_id), local_track_id, (x, y, w, h), float(conf)))

    rows.sort(key=lambda r: (r[0], r[1]))

    lines: List[str] = []
    for frame, local_id, (x, y, w, h), conf in rows:
        gid = mapping.get((str(camera_id), int(local_id)))
        if gid is None and not keep_unmapped:
            continue
        track_id = int(gid) if gid is not None else int(local_id)

        cls = default_class
        vis = default_visibility
        lines.append(f"{frame},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},{cls},{vis}")

    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True, help="camera_id, e.g. 1")
    ap.add_argument("--mapping", required=True, help="mapping JSON, e.g. tracking_results_2/mapping.json")
    ap.add_argument("--tracks", required=True, help="camera tracks JSON, e.g. tracking_results_2/camera1/camera_1.json")
    ap.add_argument("--out", required=True, help="output MOT txt, e.g. tracking_results_2/1_pred.txt")
    ap.add_argument("--keep-unmapped", action="store_true")
    args = ap.parse_args()

    camera_id = str(args.camera)
    mapping = load_mapping(Path(args.mapping))
    tracks = load_tracks(Path(args.tracks))

    lines = to_mot_lines(
        tracks=tracks,
        mapping=mapping,
        camera_id=camera_id,
        keep_unmapped=bool(args.keep_unmapped),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


if __name__ == "__main__":
    main()