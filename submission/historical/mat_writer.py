"""MATLAB v7.3 metadata plus streaming RGB frames; never silently use a file reference."""
from pathlib import Path
import hashlib
import numpy as np

REQUIRED=('facial_video','roi_data','reference_signal','processed_signal','metrics','params')


def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_mat(path, fields, frames=None, shape=None, video_reference=None):
    import h5py
    import hdf5storage
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    if (frames is None)==(video_reference is None):raise ValueError('Exactly one video storage mode required')
    missing=set(REQUIRED)-{'facial_video'}-set(fields)
    if missing:raise ValueError('Missing fields: '+str(missing))
    data=dict(fields)
    if video_reference is not None:
        data['facial_video']={'storage':'external_reference_NOT_confirmed_compliant',
                             'path':str(video_reference),'sha256':file_sha256(video_reference)}
    hdf5storage.savemat(str(path),data,format='7.3',store_python_metadata=False,
                       matlab_compatible=True,oned_as='column',truncate_existing=True)
    digest=hashlib.sha256(); count=0
    if frames is not None:
        if shape is None or len(shape)!=4 or shape[2]!=3 or min(shape)<=0:
            raise ValueError('Expected H,W,3,T positive dimensions')
        h,w,c,t=map(int,shape)
        with h5py.File(path,'a') as f:
            ds=f.create_dataset('facial_video',shape=(t,c,w,h),dtype='uint8',
                                chunks=(1,c,w,h),compression='gzip',compression_opts=1)
            ds.attrs['MATLAB_class']=np.bytes_('uint8')
            for i,rgb in enumerate(frames):
                rgb=np.asarray(rgb)
                if i>=t or rgb.shape!=(h,w,3) or rgb.dtype!=np.uint8:
                    raise ValueError('Frame count, shape or dtype mismatch')
                ds[i]=rgb.transpose(2,1,0);digest.update(rgb.tobytes());count+=1
            if count!=t:raise ValueError(f'Expected{t} frames; got{count}')
    return {'storage':'embedded_raw_rgb' if frames is not None else 'external_reference',
            'frames':count,'decoded_rgb_sha256':digest.hexdigest() if count else 'NA',
            'file_bytes':path.stat().st_size}


def verify_mat(path, expected_shape=None, expected_rgb_sha256=None, full_mat73=True):
    import mat73
    import h5py
    # Full real-file readback is opt-out, not assumed from a metadata-only check.
    data=mat73.loadmat(str(path),**({} if full_mat73 else {'only_include':list(REQUIRED[1:])}))
    missing=set(REQUIRED if full_mat73 else REQUIRED[1:])-set(data)
    if missing:raise AssertionError('Readback missing '+str(missing))
    if expected_shape is not None:
        if full_mat73:
            raw=data['facial_video']
            if raw.shape!=tuple(expected_shape) or raw.dtype!=np.uint8:raise AssertionError('RGB shape/type mismatch')
            del data['facial_video']; del raw
        digest=hashlib.sha256()
        with h5py.File(path) as f:
            if f['facial_video'].shape!=tuple(reversed(expected_shape)):raise AssertionError('HDF5 axis order mismatch')
            for x in f['facial_video']:digest.update(x.transpose(2,1,0).tobytes())
        if expected_rgb_sha256 and digest.hexdigest()!=expected_rgb_sha256:raise AssertionError('RGB readback hash mismatch')
    for key in REQUIRED[1:]:
        if not isinstance(data[key],dict):raise AssertionError(key+' must be MATLAB struct')
    return {'mat73_readback':'full' if full_mat73 else 'metadata_only',
            'required_fields_present':True,'rgb_hash_verified':expected_rgb_sha256 is not None}
