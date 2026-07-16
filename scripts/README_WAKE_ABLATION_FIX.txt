v10.0.2.1 uses explicit BooleanOptionalAction tokens:
  wake on  -> --wake-shielding
  wake off -> --no-wake-shielding

The previous v10.0.2 ablation omitted the negative token, so argparse retained its default wake_shielding=True in both cases.
