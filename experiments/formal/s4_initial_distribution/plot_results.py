"""Pair the unchanged published M2 curve with within-Gaussian S4 measurements.

Each panel retains the existing Python figure design. Vector composition changes
only placement and prefixes SVG identifiers; no reference values are regenerated.
The two field plates remain separate figures, as requested by the protocol.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
import sys
from xml.etree import ElementTree as ET

from experiments.formal._shared.figure_evidence import bind_result, finish_figures, sha256, verify_svg
from experiments.formal.m2_recursive_prediction.plot_mean_error import (
    POPULATION, draw, read_source, write_source)
from scripts.verify_published_results import verify_package

ROOT = Path(__file__).resolve().parents[3]
SVG = 'http://www.w3.org/2000/svg'
ET.register_namespace('', SVG)


def pair_vectors(left: Path, right: Path, output: Path):
    """Preserve each panel's internal coordinates and editable vector content."""
    panels = [ET.parse(path).getroot() for path in (left, right)]
    boxes = [list(map(float, panel.attrib['viewBox'].split())) for panel in panels]
    if any(box[:2] != [0., 0.] for box in boxes):
        raise ValueError('Expected zero-origin plot panels')
    gap, heading = 18., 22.
    width, height = sum(box[2] for box in boxes) + gap, max(box[3] for box in boxes) + heading
    document = ET.Element(f'{{{SVG}}}svg', width=f'{width}pt', height=f'{height}pt',
                          viewBox=f'0 0 {width} {height}', version='1.1')
    x = 0.
    for index, (panel, box, title) in enumerate(zip(panels, boxes,
            ('a  Four-vortex initial conditions (M2)', 'b  Gaussian random-field initial conditions (S4)'))):
        label = ET.SubElement(document, f'{{{SVG}}}text', x=str(x + 8), y='13',
                              style='font-family:Arial,sans-serif;font-size:9px;font-weight:bold')
        label.text = title
        group = ET.SubElement(document, f'{{{SVG}}}g', transform=f'translate({x},{heading})')
        prefix = f'panel{index}_'
        identifiers = {node.attrib['id'] for node in panel.iter() if 'id' in node.attrib}
        for node in panel.iter():
            for key, value in list(node.attrib.items()):
                if key == 'id':
                    node.set(key, prefix + value)
                elif value.startswith('#') and value[1:] in identifiers:
                    node.set(key, '#' + prefix + value[1:])
                elif 'url(#' in value:
                    node.set(key, re.sub(r'url\(#([^)]*)\)', lambda m: 'url(#' + prefix + m[1] + ')', value))
        for child in panel:
            group.append(copy.deepcopy(child))
        x += box[2] + gap
    with output.open('xb') as stream:
        ET.ElementTree(document).write(stream, encoding='utf-8', xml_declaration=True)
    verify_svg(output)
    return {'width_pt': width, 'height_pt': height, 'reference_panel_modified': False,
            'operation': 'vector translation and unique ID prefixes only'}


def incomplete_population_note(source):
    """Disclose, inside the figure, every curve that stops early and why.

    A method whose later report times no longer carry the full test population is
    drawn only where that population is complete. The omission is stated on the
    panel rather than left implicit, so a truncated curve cannot read as a
    complete-population result.
    """
    notes = []
    detail = []
    for method in dict.fromkeys(row['method'] for row in source):
        rows = [row for row in source if row['method'] == method]
        complete = [row for row in rows if row['complete_population']]
        later = sorted((float(row['absolute_time']), int(row['finite_count_min']))
                       for row in rows if not row['complete_population'])
        if not later:
            continue
        last = max(float(row['absolute_time']) for row in complete)
        worst = min(count for _, count in later)
        notes.append('%s ends at t=%.1f: %d of %d trajectories become non-finite later'
                     % (method, last, POPULATION - worst, POPULATION))
        detail.append('%s at %s of %d' % (
            method, ', '.join('t=%.1f %d' % (time, count) for time, count in later), POPULATION))
    return notes, detail


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, default=ROOT/'results/formal/M2_recursive_prediction')
    args = parser.parse_args(argv)
    reference = args.reference_dir.resolve(strict=True)
    verify_package(reference, 'M2')
    evidence = bind_result(args.result_dir, 'S4', ('summary/metrics.csv', 'raw/predictions.h5'))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    curve_dir = output/'curves'
    curve_dir.mkdir()
    source = read_source(args.result_dir/'summary/metrics.csv', cohort='test_2260_2439')
    write_source(curve_dir/'source_data.csv', source)
    figure = draw(source)
    notes, detail = incomplete_population_note(source)
    if notes:
        figure.axes[0].text(0.03, 0.97, '\n'.join(notes + ['finite counts ' + '; '.join(detail)]),
                            transform=figure.axes[0].transAxes, ha='left', va='top',
                            fontsize=5.0, color='#3A3A3A', linespacing=1.45, zorder=20)
    right = curve_dir/'gaussian_mean_relative_l2_vs_time.svg'
    figure.savefig(right, bbox_inches='tight')
    limits = list(figure.axes[0].get_ylim())
    import matplotlib.pyplot as plt
    plt.close(figure)
    left = reference/'figures/mean_relative_l2_vs_time.svg'
    layout = pair_vectors(left, right, curve_dir/'initial_distribution_comparison.svg')
    # Include exact original numbers, not a new measurement or a refitted model.
    write_source(curve_dir/'reference_source_data.csv', read_source(reference/'summary/metrics.csv'))
    finish_figures(curve_dir, evidence, Path(__file__), {
        'reference': {'experiment': 'M2', 'published_manifest_sha256': sha256(reference/'published.json'),
                      'curve_sha256': sha256(left), 'reuse': 'left panel, unchanged scientific content'},
        'layout': layout, 'gaussian_y_limits': limits, 'population_per_distribution': 180,
        'training_seeds_per_gift_regime': 3, 'baseline_models_per_method': 1,
        'comparison': 'within-distribution training and testing; not distribution transfer',
        'failed_trajectories_omitted': False, 'error_bars': 'none', 'hypothesis_tests': 'none',
        'curves_truncated_at_last_complete_population': notes,
        'finite_counts_after_truncation': detail,
        'curve_renderer_sha256': sha256(ROOT/'experiments/formal/m2_recursive_prediction/plot_mean_error.py')})
    subprocess.run([sys.executable, '-B', '-m', 'experiments.formal.m2_recursive_prediction.plot_keyframes',
        '--experiment', 'S4', '--input', str(args.result_dir/'raw/predictions.h5'),
        '--output-dir', str(output/'keyframes'), '--trajectory-id', '2265'], check=True, cwd=ROOT)
    print(json.dumps({'status': 'complete', 'output': str(output),
        'reference_keyframe': 'results/formal/M2_recursive_prediction/figures/keyframes/traj1045_recursive_keyframes.svg',
        'reference_plate_reused_without_changes': True}))


if __name__ == '__main__':
    main()
