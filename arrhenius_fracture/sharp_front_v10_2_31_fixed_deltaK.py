"""Fixed-dimensional-DeltaK entry for v10.2.31 endurance-knee validation."""
from __future__ import annotations
from . import sharp_front_v10_2_30_fixed_deltaK as _fixed
from . import sharp_front_v10_2_31_endurance_knee as _mapped

def main(argv=None):
    original=_fixed._energy.main
    _fixed._energy.main=_mapped.main
    try:return _fixed.main(argv)
    finally:_fixed._energy.main=original

if __name__=="__main__":main()
