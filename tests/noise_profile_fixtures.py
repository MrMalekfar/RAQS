"""Explicit legacy wire-format inputs for tests that mutate v1-v3 keys."""
import json

from rydberg_aoa.config.noise import parse_noise_config
from rydberg_aoa.config.profile_document import noise_document


def legacy_noise_profile(path):
    profile = json.loads(path.read_text(encoding='utf-8'))
    profile['noise'] = noise_document(parse_noise_config(profile['noise']))
    # The strict wire parser expects JSON arrays, including in detector settings.
    return json.loads(json.dumps(profile))
