"""Dataset boundary: lazy RGB frames, seconds, typed labels and qualified IDs.

No resampling, normalization, face detection or label imputation occurs here.
Each label keeps its own clock. Unknown synchronization is explicit.
"""
from dataclasses import dataclass, field
from pathlib import Path
import json
import csv
import re
import numpy as np
if __package__:
    from .ppg_polarity import normalize_recording
else:
    from ppg_polarity import normalize_recording


@dataclass
class Label:
    values: np.ndarray
    timestamps: np.ndarray | None
    unit: str
    scope: str = "sample"
    valid_mask: np.ndarray | None = None

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != 1:
            raise ValueError("Labels must be a vector")
        finite=np.isfinite(self.values)
        self.valid_mask=finite if self.valid_mask is None else np.asarray(self.valid_mask,bool)&finite
        if self.valid_mask.shape!=self.values.shape:raise ValueError('Label validity shape mismatch')
        if self.scope=='recording' and not finite.all():raise ValueError('Invalid recording-level label')
        if self.timestamps is not None:
            self.timestamps = check_time(self.timestamps)
            if len(self.values) != len(self.timestamps):
                raise ValueError("Label/time length mismatch")
        elif self.scope != "recording":
            raise ValueError("Sample labels require timestamps")


def check_time(t):
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) < 0):
        raise ValueError("Expected finite nondecreasing timestamps in seconds")
    return t


@dataclass
class VideoFrames:
    """Re-iterable stream; never materialize full-resolution recordings by default."""
    path: Path
    count: int

    def __iter__(self):
        import cv2
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open {self.path}")
        n = 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                n += 1
                if n > self.count:
                    raise ValueError("More frames than timestamps")
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if n != self.count:
                raise ValueError(f"Incomplete decode: {n}/{self.count}")
        finally:
            cap.release()

    def __len__(self):
        return self.count


@dataclass
class ImageFrames:
    paths: list

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        import cv2
        for p in self.paths:
            bgr = cv2.imread(str(p))
            if bgr is None:
                raise ValueError(f"Cannot decode {p}")
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@dataclass
class Recording:
    frames: VideoFrames | ImageFrames
    timestamps: np.ndarray
    labels: dict[str, Label]
    subject_id: str
    recording_id: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.timestamps = check_time(self.timestamps)
        if len(self.frames) != len(self.timestamps):
            raise ValueError("Frame/time count mismatch")

    def as_tuple(self):
        return self.frames, self.timestamps, self.labels, self.subject_id

    def __iter__(self):
        return iter(self.as_tuple())

    def require_synchronized_ppg(self):
        if not (self.metadata.get("synchronization_verified") or self.metadata.get('operational_alignment_accepted')) or "ppg" not in self.labels:
            raise ValueError("Synchronized PPG unavailable; do not create window ground truth")
        return self.labels["ppg"]


class UBFC2:
    def __init__(self, root):
        self.root = Path(root)

    def recordings(self):
        return sorted(p.parent.name for p in self.root.glob("*/ground_truth.txt"))

    def load(self, recording_id):
        folder = self.root / recording_id
        gt = np.loadtxt(folder / "ground_truth.txt", ndmin=2)
        if gt.shape[0] != 3 or not np.isfinite(gt).all():
            raise ValueError("UBFC2 requires exactly three finite rows")
        t = check_time(gt[2] - gt[2, 0])
        return Recording(VideoFrames(folder / "vid.avi", len(t)), t,
                         {"ppg": Label(gt[0], t, "a.u."),
                          "hr": Label(gt[1], t, "bpm")},
                         f"ubfc2:{recording_id}", recording_id,
                         {"dataset": "UBFC2", "synchronization_verified": True,
                          "clock": "ground_truth seconds",
                          "duplicate_timestamps": int(np.sum(np.diff(t) == 0))})


def video_pts(path):
    """Use presentation timestamps, never assume nominal FPS equals sensor time."""
    import subprocess
    import shutil
    if shutil.which("ffprobe") is None:
        import av
        with av.open(str(path)) as container:
            times = []
            for frame in container.decode(video=0):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError("Missing video timestamps")
                times.append(float(frame.pts * frame.time_base))
        t = check_time(times)
        return t - t[0]
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_frames", "-show_entries", "frame=best_effort_timestamp_time",
                        "-of", "json", str(path)], check=True, capture_output=True, text=True)
    frames = json.loads(p.stdout)["frames"]
    if any("best_effort_timestamp_time" not in f for f in frames):
        raise ValueError("Missing video timestamps")
    t = check_time([float(f["best_effort_timestamp_time"]) for f in frames])
    return t - t[0]


class UBFC1:
    def __init__(self, root):
        self.root = Path(root)

    def recordings(self):
        return sorted(str(p.parent.relative_to(self.root)) for p in self.root.rglob("gtdump.xmp"))

    def load(self, recording_id, frame_timestamps=None, sensor_offset_s=None):
        folder = self.root / recording_id
        gt = np.loadtxt(folder / "gtdump.xmp", delimiter=",", ndmin=2)
        if gt.shape[1] != 4:
            raise ValueError("UBFC1 requires columns: milliseconds, HR, SpO2, PPG")
        videos = list(folder.glob("*.avi"))
        if len(videos) != 1:
            raise ValueError("Expected one AVI per UBFC1 recording")
        t = video_pts(videos[0]) if frame_timestamps is None else check_time(frame_timestamps)
        # Starting two independent clocks at zero is not proof of synchronization.
        lt = (gt[:, 0] - gt[0, 0]) / 1000 + (sensor_offset_s or 0)
        return Recording(VideoFrames(videos[0], len(t)), t,
                         {"hr": Label(gt[:, 1], lt, "bpm"),
                          "spo2": Label(gt[:, 2], lt, "%"),
                          "ppg": Label(gt[:, 3], lt, "a.u.")},
                         f"ubfc1:{recording_id}", recording_id,
                         {"dataset": "UBFC1", "synchronization_verified": sensor_offset_s is not None,
                          "sensor_offset_s": sensor_offset_s,
                          "sensor_raw_start_s": float(gt[0, 0] / 1000),
                          "subject_identity_verified": False,
                          "split_group_id": "ubfc1:all",
                          "allow_within_dataset_split": False,
                          "clock": "video PTS; sensor milliseconds"})


class AiPEX:
    """Explicit manifest binds subject/video/reference; no fabricated PPG.

    Manifest: [{"recording_id": ..., "subject_id": ..., "video": relative_path,
                "mean_hr_bpm": ..., "roi": "forehead|left_cheek|right_cheek"}].
    All three ROIs of one person MUST share subject_id.
    """
    def __init__(self, root, manifest):
        self.root = Path(root)
        rows = json.loads(Path(manifest).read_text(encoding="utf-8-sig"))
        self.rows = {r["recording_id"]: r for r in rows}
        if len(rows) != len(self.rows):
            raise ValueError("Duplicate recording IDs")

    def recordings(self):
        return sorted(self.rows)

    def load(self, recording_id, frame_timestamps=None):
        r = self.rows[recording_id]
        path = self.root / r["video"]
        t = video_pts(path) if frame_timestamps is None else check_time(frame_timestamps)
        return Recording(VideoFrames(path, len(t)), t,
                         {"hr": Label([r["mean_hr_bpm"]], None, "bpm", "recording")},
                         f'aipex:{r["subject_id"]}', recording_id,
                         {"dataset": "AiPEX", "roi": r.get("roi"),
                          "synchronization_verified": False,
                          "benchmark_eligible": False,
                          "reason": "Public export/reference time mapping unresolved"})


class PURE:
    """Preserve nanosecond clocks and independent 60Hz reference sampling."""
    def __init__(self, root):
        self.root = Path(root)

    def recordings(self):
        return sorted(p.name for p in self.root.iterdir()
                      if p.is_dir() and re.fullmatch(r"\d\d-\d\d", p.name))

    def load(self, recording_id):
        folder = self.root / recording_id
        data = json.loads((folder / f"{recording_id}.json").read_text())
        paths = sorted((folder / recording_id).glob("*.png"),
                       key=lambda p: int(re.fullmatch(r"Image(\d+)", p.stem).group(1)))
        if not paths:
            raise ValueError("No PURE PNG frames")
        ns = np.array([int(re.fullmatch(r"Image(\d+)", p.stem).group(1)) for p in paths], np.int64)
        origin = ns[0]
        t = (ns - origin).astype(float) / 1e9
        packets = data["/FullPackage"]
        lt = (np.array([int(p["Timestamp"]) for p in packets], np.int64) - origin).astype(float) / 1e9
        labels = {}
        for key, name, unit in [("waveform", "ppg", "a.u."), ("pulseRate", "hr", "bpm"), ("o2", "spo2", "%")]:
            if all(key in p["Value"] for p in packets):
                labels[name] = Label([p["Value"][key] for p in packets], lt, unit)
        if "ppg" not in labels:
            raise ValueError("Missing PURE reference waveform")
        return Recording(ImageFrames(paths), t, labels,
                         f'pure:{recording_id.split("-")[0]}', recording_id,
                         {"dataset": "PURE", "synchronization_verified": True,
                          "clock": "shared nanoseconds; field names require real-release audit",
                          "release_audit_pending": True})


class Pilot:
    def __init__(self, root=None):
        self.root = root

    def recordings(self):
        return []

    def load(self, recording_id):
        raise NotImplementedError("Pilot schema, clock alignment and consent manifest not yet available")


class ExcludedRecording(ValueError):
    def __init__(self, metadata):
        self.metadata=metadata
        super().__init__(metadata['exclusion_reason'])


@dataclass
class V3Frames:
    path: Path
    count: int
    pts: list | None = None

    def __len__(self):return self.count

    def __iter__(self):
        import av
        n=0
        with av.open(str(self.path)) as c:
            c.streams.video[0].codec_context.thread_count=1
            for frame in c.decode(video=0):
                if n>=self.count:return
                if self.pts is not None:
                    value=str(frame.pts*frame.time_base) if frame.pts is not None else None
                    if value!=self.pts[n]:raise ValueError('V3 retained PTS mismatch')
                n+=1
                yield frame.to_ndarray(format='rgb24')
                if n==self.count:return
        if n!=self.count:raise ValueError('Short V3 retained prefix')


def read_timed_values(path):
    from datetime import datetime,timedelta
    rows=[line.split(maxsplit=1) for line in Path(path).read_text().splitlines() if line.strip()]
    epoch=datetime(1970,1,1)
    return (np.array([float(r[0]) for r in rows]),
            np.array([(datetime.fromisoformat(r[1])-epoch)//timedelta(microseconds=1) for r in rows],dtype=np.int64))


class MCD:
    """Read raw files with V3 mapping; normalize PPG sign once at backend boundary.

    Both PPG channels use the globally fixed MCD convention for all cameras and
    partitions. Access gates stay unchanged; this does not authorize D3 access.
    """
    def __init__(self,root,audit_dir,db_path=None):
        self.root=Path(root);self.audit_dir=Path(audit_dir)
        with (Path(db_path) if db_path else self.root/'db.csv').open() as f:
            self.rows={Path(r['video']).stem:r for r in csv.DictReader(f)}

    @staticmethod
    def exclusion(row):
        if row['patient_id']=='4952':return 'subject4952_excluded_by_user_even_after_path_correction'
        if row['patient_id']=='7412' and row['step']=='after':return '7412_after_no_matching_PPG'
        return None

    def recordings(self):
        return sorted(k for k,r in self.rows.items() if not self.exclusion(r) and
                      (self.audit_dir/'dev60_gate_v3'/(k+'.json')).exists())

    def load(self,recording_id):
        row=self.rows[recording_id]
        meta={'dataset':'MCD-rPPG','subject_id':'mcd:'+row['patient_id'],
              'camera':row['camera'],'step':row['step'],'not_for_hr_label':True,
              'pulse':{'value':row.get('pulse'),'not_for_hr_label':True},
              'exclusion_reason':self.exclusion(row),'source_db_row':dict(row)}
        if meta['exclusion_reason']:raise ExcludedRecording(meta)
        gate=json.loads((self.audit_dir/'dev60_gate_v3'/(recording_id+'.json')).read_text())
        n=int(gate['gate']['candidate_prefix_frames'])
        _,mt=read_timed_values(self.root/row['meta'])
        if len(mt)<n:raise ValueError('V3 retained prefix exceeds metadata')
        origin=mt[0];t=(mt[:n]-origin)/1e6
        pv,pt=read_timed_values(self.root/row['ppg'])
        labels={'ppg':Label(pv,(pt-origin)/1e6,'a.u.')}
        sync=np.loadtxt(self.root/row['ppg_sync'],ndmin=2)
        if len(sync)<n:raise ValueError('Short unchanged published ppg_sync')
        labels['ppg_frame']=Label(sync[:n,0],t,'a.u.')
        for source,name,unit in [('saturation','spo2','%'),('upper_ap','sbp','mmHg'),('lower_ap','dbp','mmHg')]:
            try:value=float(row[source])
            except (KeyError,ValueError):continue
            if np.isfinite(value):labels[name]=Label([value],None,unit,'recording')
        valid=np.ones(n,bool)
        for i in gate.get('decoded_corrupt_indices',[]):
            if i<n:valid[i]=False
        dt=np.diff(np.unique(t))
        meta.update(synchronization_verified=False,operational_alignment_accepted=True,
              synchronization_basis='V3 ordinal prefix and user operational acceptance; no offset correction',
              frame_valid=valid.tolist(),fps_effective=float((len(t)-1)/(t[-1]-t[0])),
              fps_from_median_timestamp=float(1/np.median(dt)),
              clock_origin_microseconds=int(origin),clock='actual metadata and raw contact PPG wallclock',
              mapping='row i = decoded frame i; final decoded frame and excess rows dropped',
              dropped_final_decoded_frames=1,excess_meta_rows_dropped=len(mt)-n,
              source_video_sha256=gate['source_sha256'],ppg_source=row['ppg'],
              ppg_frame_source=row['ppg_sync'],not_for_clinical_sync_claim=True)
        return normalize_recording(Recording(
            V3Frames(self.root/row['video'],n,gate.get('container_pts_rational')),t,
            labels,'mcd:'+row['patient_id'],recording_id,meta))


BACKENDS = {"ubfc2": UBFC2, "ubfc1": UBFC1, "aipex": AiPEX, "pure": PURE, "pilot": Pilot, "mcd":MCD}
