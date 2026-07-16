"""Production v10 entry point: anisotropic multifront sharp-front + unified MPZ."""
from __future__ import annotations

import sys
from . import sharp_front


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if '--material-class' not in args and '--material-manifest' not in args:
        raise SystemExit('v10 requires --material-class {ceramic,weakT,DBTT} or --material-manifest PATH')
    return sharp_front.main(args)


if __name__ == '__main__':
    main()
