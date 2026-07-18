#!/usr/bin/env python3
"""Hard stop for the invalid unsigned v10.2.4 parameter campaign."""


def main() -> None:
    raise SystemExit(
        "STOPPED: v10.2.4 maps source-site hazard multiplicity directly to "
        "unsigned coherent dislocation content and uses the legacy +1/+1 "
        "shielding projection. Generate the v10.2.5 signed 2-D unit-response "
        "kernel, mechanically derived activation-to-line normalization, and "
        "physical source-capacity bounds before running a new campaign. A "
        "K-shield cap or fitted attenuation is not permitted."
    )


if __name__ == "__main__":
    main()
