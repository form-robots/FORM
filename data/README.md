# Paper data

The archives are currently local; hosting will be chosen separately. Raw files,
simulation arrays, and presentation media are not committed to Git.

```bash
python -m reproduce.data unpack /path/to/form-*.tar.gz
python -m reproduce.data verify
```

Each archive can be installed independently. Use `verify --bundle elastic` (or
another bundle name) to check a partial installation. `manifest.json` records
sizes, SHA-256 checksums, and each file's runtime path. Installation creates only
relative links within this checkout.

| Bundle | Contents |
|---|---|
| `elastic` | Stereo observations, identification, robot assets, insertion and putting plans/results |
| `shaping_sim` | Pressing observations, material fits, six-pinch planning and execution arrays |
| `shaping_real` | RGB-D recordings, force logs, reconstructions, fits, plans, and final scans |
| `pouring` | Identification video, calibration, robot logs, simulation results, and endpoint photographs |
| `benchmark` | Available reference outputs from the comparison study; generate trajectories with NCLaw |
| `presentation` | Frozen footage, numerical inputs, equations, and licensed fonts |

To build archives from an installed dataset:

```bash
python -m reproduce.data pack data/bundles
```

`archives.json` records the prepared archive checksums; packing also writes a
local `bundles.json`. Existing archives and conflicting local
data are never overwritten. Historical provenance metadata retains its original
paths; portable entry points resolve the corresponding files within this checkout.
