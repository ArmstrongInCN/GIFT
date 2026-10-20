"""Draw W3 absolute percentage errors, with one figure per parameter.

The horizontal coordinate is always the measured absolute percentage error. Only
the categorical vertical position is separated to expose coincident cases.
Filled boxes show empirical quartiles, matching-color lines show medians,
and whiskers end at the farthest observations within 1.5 IQR of the box.
All original cases remain in the dot layer, including those beyond the whiskers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np

METHODS = ('GIFT', 'PDE-FIND', 'PDE-FIND-KC', 'PINN-SR', 'PINN-SR-KC')
PARAMETERS = ('nu', 'beta', 'gamma')
COLORS = dict(zip(METHODS, ('#174EB6', '#D45500', '#008143', '#8423B5', '#007F8F')))
MEDIAN_COLORS = COLORS.copy()
TITLES = {'nu': 'Viscosity ν', 'beta': 'Advection coefficient β', 'gamma': 'Forcing amplitude γ'}
LIMITS = {'nu': (0., 300.), 'beta': (0., 300.), 'gamma': (0., 110.)}
TICKS = {'nu': (0, 50, 100, 150, 200, 250, 300), 'beta': (0, 50, 100, 150, 200, 250, 300),
         'gamma': (0, 25, 50, 75, 100, 110)}
WIDTH_MM, HEIGHT_MM = 228., 118.
POINT_DIAMETER = 2.1
LANES_PT = np.array([0., -2.7, 2.7, -5.4, 5.4, -8.1, 8.1])
POINT_EDGE_LAYERS = 12
POINT_RADIAL_SIGMA = .42
POINT_PEAK_ALPHA = .95


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scope_path(path: Path, root: Path) -> str:
    """Use portable paths for published files and explicit paths for user exports."""
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def load_errors(source: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """Require the complete paired population and verify each saved absolute error."""
    records = {}
    with source.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            key = row['case_id'], row['method'], row['parameter']
            if key in records:
                raise ValueError(f'Duplicate parameter result: {key}')
            truth, estimate, ape = map(float, (row['truth'], row['estimate'], row['APE_percent']))
            if row['status'] != 'complete' or not np.isfinite([truth, estimate, ape]).all() or truth <= 0:
                raise ValueError(f'Incomplete or undefined parameter result: {key}')
            absolute = 100. * abs(estimate - truth) / abs(truth)
            if not np.isclose(absolute, ape, rtol=1e-12, atol=1e-12):
                raise ValueError(f'Stored APE disagrees with estimate: {key}')
            records[key] = absolute
    cases = [f'parameter_{index:03d}' for index in range(50)]
    expected = {(case, method, parameter) for case in cases for method in METHODS for parameter in PARAMETERS}
    if set(records) != expected:
        raise ValueError('W3 requires exactly 50 paired cases × five methods × three parameters.')
    values = {parameter: np.array([[records[case, method, parameter] for case in cases]
                                  for method in METHODS]) for parameter in PARAMETERS}
    return cases, values


def summarize(values: np.ndarray) -> dict:
    q1, median, q3 = np.quantile(values, [.25, .5, .75], method='linear')
    return {'n': len(values), 'minimum': float(values.min()), 'q1': float(q1),
            'median': float(median), 'q3': float(q3), 'maximum': float(values.max()),
            'mean': float(values.mean())}


def tukey_whiskers(values: np.ndarray, statistics: dict) -> dict:
    """Use actual observations inside the inclusive 1.5-IQR fences as endpoints."""
    iqr = statistics['q3'] - statistics['q1']
    lower_fence = statistics['q1'] - 1.5*iqr
    upper_fence = statistics['q3'] + 1.5*iqr
    inside = values[(values >= lower_fence) & (values <= upper_fence)]
    return {'iqr': iqr, 'lower_fence': lower_fence, 'upper_fence': upper_fence,
            'lower_observation': float(inside.min()), 'upper_observation': float(inside.max()),
            'lower_outside_count': int((values < lower_fence).sum()),
            'upper_outside_count': int((values > upper_fence).sum())}


def vertical_offsets(x_points: np.ndarray) -> tuple[np.ndarray, int]:
    """Align dots on seven fixed horizontal lanes without changing measured x.

    Stable error order and center-first lane order make the layout deterministic.
    Dense clusters may overlap when all lanes are occupied; exact error values
    take precedence over horizontal rounding, binning, or moving observations.
    """
    placed = []
    offsets = np.zeros(len(x_points))
    separation = POINT_DIAMETER + .16
    for index in np.argsort(x_points, kind='stable'):
        x = x_points[index]
        neighbors = [(xp, yp) for xp, yp in placed if abs(x - xp) < separation]
        valid = [y for y in LANES_PT if all((x-xp)**2+(y-yp)**2 >= separation**2-1e-8
                                             for xp, yp in neighbors)]
        if valid:
            chosen = valid[0]
        else:
            def penalty(y):
                return sum(max(0., separation - np.hypot(x-xp, y-yp)) ** 2 for xp, yp in neighbors)
            chosen = min(LANES_PT, key=penalty)
        offsets[index] = chosen
        placed.append((x, chosen))
    overlaps = sum((x_points[i]-x_points[j])**2+(offsets[i]-offsets[j])**2 < POINT_DIAMETER**2-1e-8
                   for i in range(len(offsets)) for j in range(i))
    return offsets, int(overlaps)


def style_axis(ax, limits, ticks) -> None:
    ax.set_xlim(limits)
    ax.set_ylim(4.65, -.65)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    ax.set_yticks(range(5), METHODS)
    ax.tick_params(axis='y', length=0, pad=11, labelsize=10.2)
    # Inward ticks point above the bottom axis; preserve the former label baseline.
    ax.tick_params(axis='x', direction='in', length=4., width=1., color='black', labelsize=9.5,
                   pad=7.5, bottom=True, top=False, labelbottom=True, labeltop=False)
    for label in ax.get_yticklabels():
        label.set_color('black')
        label.set_weight('bold')
    for name in ('top', 'right'):
        ax.spines[name].set_visible(False)
    for name in ('left', 'bottom'):
        ax.spines[name].set_visible(True)
        ax.spines[name].set_color('black')
        ax.spines[name].set_linewidth(1.2)
        ax.spines[name].set_linestyle('solid')
        ax.spines[name].set_zorder(0)
    # Extend only the drawn axis, retaining the existing scale and point positions.
    ax.spines['bottom'].set_bounds(limits[0], limits[1]+.03*(limits[1]-limits[0]))
    ax.spines['bottom'].set_clip_on(False)
    ax.grid(False)


def draw_soft_point(ax, x, y, color, gid):
    """Draw one observation as concentric vector circles with a fading edge.

    Circle diameters are in points; all layers share the measured coordinate.
    The opacity profile affects the marker alone, not the empirical data or
    their density. Incremental alpha accounts for compositing nested circles.
    """
    radii = np.linspace(1., 1./POINT_EDGE_LAYERS, POINT_EDGE_LAYERS)
    target_alpha = POINT_PEAK_ALPHA*np.exp(-.5*((radii-.5/POINT_EDGE_LAYERS)/POINT_RADIAL_SIGMA)**2)
    layer_alpha = 1.-(1.-target_alpha)/np.r_[1., 1.-target_alpha[:-1]]
    colors = np.tile(to_rgba(color), (POINT_EDGE_LAYERS, 1))
    colors[:, 3] = layer_alpha
    dot = ax.scatter(np.full(POINT_EDGE_LAYERS, x), np.full(POINT_EDGE_LAYERS, y),
                     s=(POINT_DIAMETER*radii)**2, marker='o', facecolors=colors,
                     edgecolors='none', linewidths=0., clip_on=False, zorder=3)
    dot.set_gid(gid)


def draw_rows(ax, parameter, cases, values) -> list[dict]:
    fig = ax.figure
    scale = 72. / fig.dpi
    step_pt = abs(ax.transData.transform((0., 1.))[1] - ax.transData.transform((0., 0.))[1]) * scale
    diameter = POINT_DIAMETER
    rows = []
    for row, (method, errors) in enumerate(zip(METHODS, values)):
        statistics = summarize(errors)
        whisker_statistics = tukey_whiskers(errors, statistics)
        x_pt = ax.transData.transform(np.column_stack((errors, np.full(len(errors), row))))[:, 0] * scale
        offsets, overlaps = vertical_offsets(x_pt)
        ys = row - .10 + offsets / step_pt
        name = 'full'
        group = {'method': method, 'statistics': statistics, 'view': name,
                 'whisker_statistics': whisker_statistics,
                 'shown_count': len(errors),
                 'marker_diameter_points': diameter, 'overlapping_disk_pairs': overlaps, 'points': []}
        for case_index, y in enumerate(ys):
            gid = f'observed-{parameter}-{method.lower()}-{cases[case_index]}-{name}'
            draw_soft_point(ax, errors[case_index], y, COLORS[method], gid)
            group['points'].append({'case_id': cases[case_index], 'x': float(errors[case_index]),
                                    'y': float(y), 'gid': gid})
        # Summaries use all 50 absolute errors, not absolute values of signed summaries.
        y_summary = row + .34
        whisker, = ax.plot([whisker_statistics['lower_observation'], whisker_statistics['upper_observation']], [y_summary]*2,
                           color=COLORS[method], linewidth=1.1, zorder=2)
        whisker.set_gid(f'tukey-whisker-{parameter}-{method.lower()}-{name}')
        for end in ('lower_observation', 'upper_observation'):
            cap, = ax.plot([whisker_statistics[end]]*2, [y_summary-.045, y_summary+.045],
                           color=COLORS[method], linewidth=1.1)
            cap.set_gid(f'cap-{end}-{parameter}-{method.lower()}-{name}')
        box = Rectangle((statistics['q1'], y_summary-.075), statistics['q3']-statistics['q1'], .15,
                        facecolor=COLORS[method], edgecolor=COLORS[method], linewidth=1.35, zorder=2)
        box.set_gid(f'quartile-box-{parameter}-{method.lower()}-{name}')
        ax.add_patch(box)
        median, = ax.plot([statistics['median']]*2, [y_summary-.16, y_summary+.16],
                          color=MEDIAN_COLORS[method], linewidth=1.6, solid_capstyle='butt', zorder=4)
        median.set_gid(f'median-{parameter}-{method.lower()}-{name}')
        median_text = f"Median {statistics['median']:.2f}%"
        median_label = ax.text(1.035, row, median_text, transform=ax.get_yaxis_transform(),
                               fontsize=9.4, color='black', va='center', ha='left', clip_on=False)
        median_label.set_gid(f'median-label-{parameter}-{method.lower()}')
        group['median_label'] = {'text': median_text, 'gid': median_label.get_gid(),
                               'x_axes_fraction': 1.035, 'y': row}
        group['summary_y'] = y_summary
        group['axis_limits'] = list(ax.get_xlim())
        rows.append(group)
    return rows


def draw_parameter(parameter, cases, values):
    fig = plt.figure(figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4))
    main = fig.add_axes([.18, .17, .67, .79], label=parameter+'-full')
    style_axis(main, LIMITS[parameter], TICKS[parameter])
    symbol = {'nu': 'ν', 'beta': 'β', 'gamma': 'γ'}[parameter]
    main.set_xlabel(f'Absolute percentage error of {symbol} (%)', fontsize=10.5, labelpad=11.)
    fig.canvas.draw()
    rows = draw_rows(main, parameter, cases, values)
    fig.canvas.draw()
    bounds = []
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(match=matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            bounds.append({'text': artist.get_text(), 'bounds_points': (np.array(box.bounds)*72./fig.dpi).tolist()})
    axes = []
    for axis in fig.axes:
        axes.append({'name': axis.get_label(), 'bounds_figure_fraction': list(axis.get_position().bounds),
                     'x_limits': list(axis.get_xlim()), 'y_limits': list(axis.get_ylim()), 'x_ticks': axis.get_xticks().tolist(),
                     'bottom_spine_bounds': list(axis.spines['bottom'].get_bounds())})
    return fig, {'parameter': parameter, 'title': TITLES[parameter], 'axes': axes,
                 'rows': rows, 'visible_text_bounds': bounds}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--preview-dir', type=Path, help='Defaults to output-dir; writes three PNG previews.')
    args = parser.parse_args(argv)
    cases, values = load_errors(args.source)
    for parameter, errors in values.items():
        if errors.min() < LIMITS[parameter][0] or errors.max() > LIMITS[parameter][1]:
            raise ValueError(f'{parameter} observed error exceeds the declared full display range.')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 8.,
                         'axes.unicode_minus': True, 'svg.hashsalt': 'gift-w3-parameter-distributions-v1'})
    output = args.output_dir
    preview = args.preview_dir or output
    repo = Path(__file__).resolve().parents[3]
    output.mkdir(parents=True, exist_ok=True)
    preview.mkdir(parents=True, exist_ok=True)
    source_points = output/'source_points.csv'
    with source_points.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['case_id', 'method']+[p+'_absolute_percentage_error' for p in PARAMETERS])
        for method_index, method in enumerate(METHODS):
            for case_index, case in enumerate(cases):
                writer.writerow([case, method]+[values[p][method_index, case_index] for p in PARAMETERS])
    statistics_path = output/'source_statistics.csv'
    with statistics_path.open('w', encoding='utf-8', newline='') as handle:
        columns = ['parameter', 'method']+list(summarize(values['nu'][0]))
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for parameter in PARAMETERS:
            for method, errors in zip(METHODS, values[parameter]):
                writer.writerow({'parameter': parameter, 'method': method, **summarize(errors)})
    figures = []
    for parameter in PARAMETERS:
        fig, record = draw_parameter(parameter, cases, values[parameter])
        stem = 'w3_parameter_'+parameter+'_errors'
        svg, png = output/(stem+'.svg'), preview/(stem+'.png')
        fig.savefig(svg, metadata={'Date': None})
        fig.savefig(png, dpi=300.)
        record.update({'svg': svg.name, 'svg_sha256': digest(svg),
                       'preview': scope_path(png, repo), 'preview_sha256': digest(png)})
        figures.append(record)
        plt.close(fig)
    metadata = {'schema': 'gift.w3-parameter-distributions.metadata.v1', 'status': 'complete',
                'experiment': 'W3', 'figure_number': 4, 'archetype': 'quantitative grid',
                'conclusion': 'Compare absolute parameter-identification error, spread and extremes for each physical parameter.',
                'source_csv_sha256': digest(args.source), 'plot_source_sha256': digest(Path(__file__)),
                'methods': list(METHODS), 'case_ids': cases, 'colors': COLORS,
                'case_count_per_method': 50, 'parameter_rows': 750, 'point_count_per_full_figure': 250,
                'metric': '100 * abs(estimate - truth) / abs(truth)', 'units': 'percent',
                'scales': 'linear, common across methods within each parameter',
                'box': 'Q1–Q3, NumPy linear sample quantiles, opaque method-color fill',
                'mean': 'retained in source statistics but not drawn or annotated',
                'median': '50th sample percentile; long line in the exact box color through and beyond the box and explicit numeric label',
                'center_statistic': 'median',
                'boxplot_convention': 'Tukey box plot, Q1–Q3 and median, whiskers at observed values within 1.5 IQR',
                'whiskers': 'smallest and largest observed APE within inclusive Q1−1.5IQR and Q3+1.5IQR fences',
                'whisker_iqr_multiplier': 1.5,
                'outside_whisker_points': 'all retained once in the original 50-case dot layer; no duplicated flier markers',
                'vertical_separation': 'deterministic center-first packing on seven fixed horizontal lanes; measured x is unchanged',
                'lane_offsets_points': LANES_PT.tolist(), 'point_diameter_points': POINT_DIAMETER,
                'point_alpha': POINT_PEAK_ALPHA,
                'point_rendering': {'shape': 'soft-edged circular marker', 'edge': 'no outline',
                                    'vector_layers_per_observation': POINT_EDGE_LAYERS,
                                    'radial_alpha_profile': 'Gaussian in normalized marker radius',
                                    'radial_sigma': POINT_RADIAL_SIGMA,
                                    'alpha_compositing': 'incremental concentric opacity; nominal peak alpha 0.95',
                                    'data_integrity': 'all layers share one real observation center; no density estimation or synthetic observations'},
                'method_labels': {'color': 'black', 'weight': 'bold', 'font_size_points': 10.2},
                'axis_style': {'left': 'solid black', 'bottom': 'solid black', 'width_points': 1.2, 'grid': False,
                               'x_tick_direction': 'in', 'x_tick_label_position': 'below',
                               'x_tick_label_pad_points': 7.5, 'bottom_spine_extension_fraction': .03},
                'summary_line_widths_points': {'tukey_whiskers': 1.1, 'caps': 1.1, 'median': 1.6, 'IQR_outline': 1.35},
                'summary_heights_category_units': {'IQR_box': .15, 'median_bar': .32, 'caps': .09},
                'median_bar_colors': MEDIAN_COLORS, 'median_color_white_blend_fraction': 0., 'box_fill_opacity': 1.,
                'visible_text_policy': 'parameter-specific x-axis name, numeric x ticks, black bold method names and per-method Median APE values',
                'axes_per_figure': 1,
                'not_multiple_seeds': True, 'uncertainty_or_significance_claim': False,
                'width_mm': WIDTH_MM, 'height_mm': HEIGHT_MM, 'figures': figures}
    meta_path = output/'figure_metadata.json'
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    package = args.source.resolve().parent.parent
    file_paths = [output/(f'w3_parameter_{p}_errors.svg') for p in PARAMETERS]+[source_points, statistics_path, meta_path]
    manifest = {'schema': 'gift.w3-parameter-distributions.figure-manifest.v1', 'status': 'complete',
                'experiment': 'W3', 'figure_number': 4, 'panels': dict(zip('abc', PARAMETERS)),
                'sources': [{'scope': 'numerical-package', 'path': scope_path(args.source, package),
                             'sha256': digest(args.source), 'rows': 750}],
                'plot_source': {'scope': 'repository', 'path': scope_path(Path(__file__), repo), 'sha256': digest(Path(__file__))},
                'files': [{'path': path.name, 'bytes': path.stat().st_size, 'sha256': digest(path)} for path in file_paths],
                'previews': [{'scope': 'repository', 'path': scope_path(preview/(f'w3_parameter_{p}_errors.png'), repo),
                              'sha256': digest(preview/(f'w3_parameter_{p}_errors.png'))} for p in PARAMETERS],
                'data_contract': {key: metadata[key] for key in ('case_count_per_method', 'parameter_rows',
                                  'point_count_per_full_figure', 'metric', 'units', 'scales', 'box', 'mean', 'median',
                                  'center_statistic', 'boxplot_convention', 'whiskers', 'whisker_iqr_multiplier', 'outside_whisker_points',
                                  'vertical_separation', 'lane_offsets_points', 'point_diameter_points', 'point_alpha', 'point_rendering',
                                  'method_labels', 'axis_style', 'summary_line_widths_points', 'summary_heights_category_units',
                                  'median_bar_colors', 'median_color_white_blend_fraction', 'box_fill_opacity', 'visible_text_policy',
                                  'axes_per_figure', 'not_multiple_seeds', 'uncertainty_or_significance_claim')}}
    (output/'figure_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'complete', 'figures': 3, 'observations_per_full_figure': 250,
                      'source_sha256': digest(args.source), 'output': str(output)}))


if __name__ == '__main__':
    main()
