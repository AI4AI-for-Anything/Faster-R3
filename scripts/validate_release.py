#!/usr/bin/env python3
"""Check published source provenance and aggregate measurement arithmetic (no GPU)."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate():
    data = json.loads((ROOT / 'docs/validation/h100.json').read_text())
    assert data['world_size'] == 8
    expected_files = set(data['runtime_files'])
    actual_files = {str(p.relative_to(ROOT)) for folder in ('modeling', 'train')
                    for p in (ROOT / folder).rglob('*.py')}
    assert actual_files == expected_files, 'Runtime file set differs from measured source'
    for name, expected in data['runtime_files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    final_digest = '7ee0e76bcc53ddd6b5b8d905bb99a861302d2bba9829699f101a43bd99f4d41a'
    for row in data['rows']:
        t, b = row['timing'], row['bitwise']
        assert (t['warmup'], t['window'], t['windows']) == (10, 2, 3), row['name']
        assert all(len(t[k]) == 3 for k in ('ref_a', 'cand', 'ref_b', 'ratios'))
        ratios = [min(a, b) / c for a, b, c in zip(t['ref_a'], t['ref_b'], t['cand'])]
        spread = max(abs(a - b) / min(a, b) for a, b in zip(t['ref_a'], t['ref_b']))
        assert all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(ratios, t['ratios'])), row['name']
        assert math.isclose(min(ratios), t['speedup_min_of_ratios'], abs_tol=1e-10)
        assert math.isclose(spread, t['ref_spread'], abs_tol=1e-10)
        assert b['identical'] and not b['differences'] and b['steps'] == 14, row['name']
        assert b['final_digest_rank0'] == final_digest, row['name']
        for name, m in row['memory'].items():
            assert not m['oom'], (row['name'], name)
            assert set(m['completed_steps_by_rank']) == {str(i) for i in range(8)}
            assert set(m['completed_steps_by_rank'].values()) == {14 if name.startswith('bitwise') else 16}
            assert 0 < m['peak_allocated_gib'] <= m['peak_reserved_gib']
    rows = {r['name']: r for r in data['rows']}
    for name in ('fusion_b_r', 'fusion_b_total2'):
        t = rows[name]['timing']
        assert t['speedup_min_of_ratios'] >= max(1.02, 1 + 2 * t['ref_spread'])
        assert rows[name]['tree_source'] == rows['fusion_b_r']['tree_source']
    return data


if __name__ == '__main__':
    data = validate()
    print(f"PASS: {len(data['runtime_files'])} runtime source hashes; {len(data['rows'])} round records; "
          "8-rank completion; final bitwise verdicts and paired benefit thresholds. No GPU run performed.")
