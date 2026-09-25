"""Identification with learned constitutive bases."""
import os
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[2]
STAGING_ROOT = ENGINE_ROOT


def artifact_dir(name='fe_ls_baseline'):
    key = 'FE_LS_OUT' if name == 'fe_ls_baseline' else 'DIFFSIM_OUT'
    return Path(os.environ[key]) if key in os.environ else ENGINE_ROOT / 'out' / name
