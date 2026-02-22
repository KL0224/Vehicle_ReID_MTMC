# file: `tools/load_mapping_id.py`

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from pymilvus import connections, Collection


def fetch_mapping(
    host: str,
    port: str,
    collection_name: str,
    camera_id: Optional[str] = None,
    page_size: int = 2000,
) -> List[dict]:
    # Connect
    connections.connect(alias="default", host=host, port=port)

    # Load collection
    col = Collection(collection_name)
    col.load()

    # Filter
    expr = f'camera_id == "{camera_id}"' if camera_id else "global_id >= 0"

    # Page query and dedupe by max frame_id per (camera_id, local_track_id)
    offset = 0
    best: Dict[Tuple[str, int], dict] = {}

    while True:
        rows = col.query(
            expr=expr,
            output_fields=["camera_id", "local_track_id", "global_id", "frame_id"],
            limit=page_size,
            offset=offset,
        )
        if not rows:
            break

        for r in rows:
            cam = str(r.get("camera_id"))
            tid = r.get("local_track_id")
            gid = r.get("global_id")
            fid = r.get("frame_id")

            if tid is None or gid is None:
                continue

            key = (cam, int(tid))

            # If frame_id is missing, keep the first seen row
            if fid is None:
                if key not in best:
                    best[key] = {
                        "camera_id": cam,
                        "local_track_id": int(tid),
                        "global_id": int(gid),
                        "frame_id": None,
                    }
                continue

            prev = best.get(key)
            if prev is None or prev["frame_id"] is None or int(fid) > int(prev["frame_id"]):
                best[key] = {
                    "camera_id": cam,
                    "local_track_id": int(tid),
                    "global_id": int(gid),
                    "frame_id": int(fid),
                }

        offset += len(rows)

    data = list(best.values())
    data.sort(key=lambda x: (x["camera_id"], x["local_track_id"]))
    return data


def write_outputs(data: List[dict], out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out_base.with_suffix(".csv")
    json_path = out_base.with_suffix(".json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["camera_id", "local_track_id", "global_id", "frame_id"])
        for r in data:
            w.writerow([r["camera_id"], r["local_track_id"], r["global_id"], r["frame_id"]])

    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", default="19531")
    ap.add_argument("--collection", default="vehicle_reid")
    ap.add_argument("--camera", default=None, help="Optional camera_id filter, e.g. 1")
    ap.add_argument("--out", required=True, help="Output base path, e.g. tracking_results_2/mapping")
    ap.add_argument("--page", type=int, default=2000, help="Query page size (offset/limit)")
    args = ap.parse_args()

    out_base = Path(args.out)
    data = fetch_mapping(
        host=args.host,
        port=args.port,
        collection_name=args.collection,
        camera_id=args.camera,
        page_size=args.page,
    )
    write_outputs(data, out_base)


if __name__ == "__main__":
    main()