"""Operational policy tests; no CUDA allocation or model changes."""
import runtime_shared_v4 as r
from unittest.mock import patch
def run():
    with patch.object(r.subprocess,'check_output',return_value='0, 16384, 8311, 100\n1, 16384, 11000, 100\n'):
        assert r.available_gpu() is True
    with patch.object(r.subprocess,'check_output',return_value='0, 16384, 5999, 0\n'):
        assert r.available_gpu() is False
    with patch.object(r.subprocess,'check_output',return_value='0, 16384, 6000, 100\n'):
        assert r.available_gpu() is True
    e=r.subprocess.CalledProcessError(1,['du','-sb','/test'])
    with patch.object(r,'_original_guard',side_effect=[e,{'verified':True}]),patch.object(r.time,'sleep'):
        assert r.guard()=={'verified':True}
    with patch.object(r,'_original_guard',side_effect=e),patch.object(r.time,'sleep'):
        try:r.guard()
        except r.subprocess.CalledProcessError:pass
        else:raise AssertionError('Failed measurement cannot pass')
    with patch.object(r,'_original_guard',side_effect=AssertionError('STOP_STORAGE_T8')):
        try:r.guard()
        except AssertionError as e:assert str(e)=='STOP_STORAGE_T8'
        else:raise AssertionError('Budget violation cannot pass')
    print('PASS: busy GPU allowed, insufficient VRAM refused, storage measurement must succeed')
if __name__=='__main__':run()
