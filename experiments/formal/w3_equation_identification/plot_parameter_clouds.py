"""Plot W3 joint signed errors as continuous aerosol envelopes and soft dots.

Exactly 250 measured centers keep their original signed-percent coordinates.
Deterministic geometric support samples inside the empirical nonellipsoidal
bodies are Gaussian-reconstructed into continuous fog, not displayed as extra
observations. Shared optical visibility and approximate depth slabs are display
conventions, not probability, uncertainty or an accuracy score.
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
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
import contourpy
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from scipy.special import logsumexp
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.stats import gaussian_kde, qmc

METHODS = ('GIFT', 'PDE-FIND', 'PDE-FIND-KC', 'PINN-SR', 'PINN-SR-KC')
PARAMETERS = ('nu', 'beta', 'gamma')
SYMBOLS = ('ν', 'β', 'γ')
STYLES = {
    'GIFT': ('#16386C', 'o'), 'PDE-FIND': ('#C58A36', 'o'),
    'PDE-FIND-KC': ('#236B4C', 'o'), 'PINN-SR': ('#925AB1', 'o'),
    'PINN-SR-KC': ('#287780', 'o'),
}
SMOOTH_TAU = .05  # Common geometric smoothing, in sample-standard-deviation units.
COVERAGE_PADDING = 1e-9
COMMON_LIMITS = np.array([[-120.,400.],[-285.,150.],[-110.,30.]])
# Pair limits and box lengths to preserve the current common raw-percent metric.
BOX_ASPECT = np.diff(COMMON_LIMITS,axis=1).ravel()*np.array([1.,17./14.,51./16.])
BOX_ASPECT /= BOX_ASPECT[0]
SPHERE_SAMPLES = 4096


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_points(source: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """Validate the stored APE, then compute signed coordinates from estimates."""
    records = {}
    with source.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            key = (row['case_id'], row['method'], row['parameter'])
            if key in records:
                raise ValueError(f'Duplicate W3 observation: {key}')
            ape, truth, estimate = map(float, (row['APE_percent'], row['truth'], row['estimate']))
            if row['status'] != 'complete' or not np.isfinite([ape, truth, estimate]).all() or truth <= 0:
                raise ValueError(f'Incomplete or undefined W3 observation: {key}')
            signed = 100. * (estimate - truth) / truth
            if not np.isclose(ape, abs(signed), rtol=1e-12, atol=1e-12):
                raise ValueError(f'Invalid stored APE: {key}')
            records[key] = signed
    cases = [f'parameter_{i:03d}' for i in range(50)]
    required = {(c, m, p) for c in cases for m in METHODS for p in PARAMETERS}
    if set(records) != required:
        raise ValueError('All 50 cases, five methods and three parameters are required')
    points = {m: np.array([[records[c, m, p] for p in PARAMETERS] for c in cases]) for m in METHODS}
    for method, values in points.items():
        if np.linalg.matrix_rank(values - values.mean(axis=0)) != 3:
            raise ValueError(f'{method} has a degenerate cloud; do not perturb its measurements')
    return cases, points


def envelope_field(z: np.ndarray, detail: dict) -> np.ndarray:
    """Smooth maximum of the empirical hull's unique supporting planes."""
    planes = np.asarray(detail['halfspaces'])
    return detail['tau']*logsumexp((z @ planes[:,:3].T+planes[:,3])/detail['tau'],axis=-1)


def radial_surface(directions: np.ndarray, detail: dict) -> np.ndarray:
    """Find the analytic boundary on each ray from the sample mean (z=0)."""
    directions = directions/np.linalg.norm(directions,axis=1,keepdims=True)
    lo = np.zeros(len(directions)); hi = np.ones(len(directions))
    level = detail['level']
    for _ in range(30):
        inside = envelope_field(directions*hi[:,None],detail) <= level
        if not inside.any(): break
        hi[inside] *= 2.
    else:
        raise ValueError('Unable to bracket the bounded empirical envelope')
    for _ in range(52):
        middle = (lo+hi)*.5
        inside = envelope_field(directions*middle[:,None],detail) <= level
        lo = np.where(inside,middle,lo); hi = np.where(inside,hi,middle)
    return directions*((lo+hi)*.5)[:,None]


def smooth_envelope(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Smooth an empirical convex outer envelope without a fixed shape family.

    F(z)=tau*logsumexp((n_j dot z+b_j)/tau), with unique hull planes.
    c=max_i F(z_i)+coverage_padding gives a convex sublevel F<=c containing
    every measured point and therefore their entire convex hull. No ellipsoid,
    covariance contour, density contour or symmetry constraint is imposed.
    """
    mean = values.mean(axis=0); scale = values.std(axis=0,ddof=1)
    z = (values-mean)/scale
    hull = ConvexHull(z)
    _,indices = np.unique(np.round(hull.equations,12),axis=0,return_index=True)
    planes = hull.equations[np.sort(indices)]
    detail = {'algorithm':'empirical_hull_logsumexp_sublevel',
              'tau':SMOOTH_TAU,'coverage_padding':COVERAGE_PADDING,
              'normalization_mean':mean.tolist(),'normalization_sample_std':scale.tolist(),
              'standardization_ddof':1,'halfspaces':planes.tolist(),
              'halfspace_count':len(planes),'triangulated_hull_face_count':len(hull.equations),
              'hull_input_count':len(values),'ray_origin':mean.tolist(),
              'ray_origin_is_symmetry_constraint':False,'origin_in_envelope_input':False,
              'predefined_shape_family':None,'confidence_region':False,'convex_outer_envelope':True}
    detail['level'] = float(envelope_field(z,detail).max()+COVERAGE_PADDING)
    if envelope_field(np.zeros((1,3)),detail)[0] >= detail['level']:
        raise ValueError('The sample mean must lie strictly within the envelope')
    k = np.arange(SPHERE_SAMPLES)
    height = 1.-2.*(k+.5)/SPHERE_SAMPLES
    angle = k*(np.pi*(3.-np.sqrt(5.)))
    radius = np.sqrt(1.-height**2)
    directions = np.column_stack((radius*np.cos(angle),radius*np.sin(angle),height))
    norms = np.linalg.norm(z,axis=1)
    directions = np.vstack((directions,np.eye(3),-np.eye(3),z[norms>0]/norms[norms>0,None]))
    surface_z = radial_surface(directions,detail)
    mesh = ConvexHull(surface_z)
    vertices = mean+surface_z*scale
    mesh_excess = float(np.max(z @ mesh.equations[:,:3].T+mesh.equations[:,3]))
    residual = float(np.max(np.abs(envelope_field(surface_z,detail)-detail['level'])))
    if mesh_excess > 1e-8 or residual > 1e-10:
        raise ValueError('The sampled smooth envelope must contain all observations')
    # F>=each supporting plane, so this shifted polytope rigorously bounds the
    # entire continuous smooth body (not just the finite rendering mesh).
    outer_planes = planes.copy(); outer_planes[:,3] -= detail['level']
    outer_z = HalfspaceIntersection(outer_planes,np.zeros(3)).intersections
    outer = mean+outer_z*scale
    detail.update({
        'max_observation_implicit_level':float((envelope_field(z,detail)-detail['level']).max()),
        'max_mesh_halfspace_excess':mesh_excess,'max_surface_abs_residual':residual,
        'outer_bounds':[outer.min(axis=0).tolist(),outer.max(axis=0).tolist()],
        'outer_bound_standardized_vertices':outer_z.tolist(),
        'outer_bound_reason':'F(z)>=n_j dot z+b_j for every supporting plane',
        'mesh_vertex_count':len(vertices),'mesh_triangle_count':len(mesh.simplices),
        'mesh_contains_all_50_points':True,'sphere_direction_count':SPHERE_SAMPLES,
        'mesh_volume_over_empirical_hull':float(mesh.volume/hull.volume),
        'surface_extends_beyond_observed_cloud':True,
    })
    return vertices,mesh.simplices,detail


class _ProjectedAxisArrow(FancyArrowPatch):
    """A vector overlay at already projected, fixed-camera 3D coordinates."""

    def do_3d_projection(self):
        return 0.


def draw_origin_axes(ax, limits: np.ndarray, tick_values: list[list[float]]) -> list[dict]:
    """Draw exactly three projected Cartesian axes through the data origin.

    Tick locations are real 3D coordinates. Short tick strokes and numeric
    labels are offset in display points for consistent readable typography.
    No projected coordinate is substituted for an observation's data value.
    """
    definitions = []
    names = ('nu', 'beta', 'gamma')
    origin_projected = proj3d.proj_transform(0., 0., 0., ax.get_proj())
    for dimension, (name, symbol) in enumerate(zip(names, SYMBOLS)):
        endpoints = np.zeros((2, 3)); endpoints[:, dimension] = limits[dimension]
        xy = np.array(proj3d.proj_transform(*endpoints.T, ax.get_proj())[:2]).T
        screen = ax.transData.transform(xy)
        unit = screen[1] - screen[0]; unit /= np.linalg.norm(unit)
        normal = np.array([unit[1], -unit[0]])
        if dimension in (0, 1): normal = -normal
        # z-axis tick text goes on its right; x/y tick text is below its line.
        if dimension == 2: normal = np.array([1., 0.])
        arrow = _ProjectedAxisArrow(tuple(xy[0]), tuple(xy[1]), arrowstyle='<|-|>',
                                mutation_scale=8, linewidth=.8, color='#555C66',
                                transform=ax.transData, clip_on=False, zorder=100000)
        arrow.set_gid('axis-'+name); ax.add_artist(arrow)
        for value in tick_values[dimension]:
            if value == 0: continue
            location = np.zeros(3); location[dimension] = value
            px, py, _ = proj3d.proj_transform(*location, ax.get_proj())
            anchor = ax.transData.transform((px, py))
            half_tick = normal * (2.0 * ax.figure.dpi / 72.)
            ends = ax.transData.inverted().transform([anchor-half_tick, anchor+half_tick])
            stroke = Line2D(ends[:, 0], ends[:, 1], color='#555C66', linewidth=.65,
                            transform=ax.transData, clip_on=False, zorder=100001)
            stroke.set_gid('tick-'+name+'-'+str(int(value))); ax.add_line(stroke)
            signed_label = f'{value:+.0f}'
            label = ax.annotate(signed_label, (px, py), xytext=tuple(normal*7.0),
                                textcoords='offset points', ha='left' if dimension == 2 else 'center', va='center',
                                fontsize=8.5, color='#444B55', zorder=100002, annotation_clip=False)
            label.set_gid('tick-label-'+name+'-'+str(int(value)))
        axis_text = ('x: '+symbol+' error (%)', 'y: '+symbol+' error (%)',
                     'z: '+symbol+' error (%)')[dimension]
        label_offset = unit*22.+normal*(16. if dimension==0 else 12.)
        label = ax.annotate(axis_text, tuple(xy[1]), xytext=tuple(label_offset),
                            textcoords='offset points', ha=('left' if unit[0]>=0 else 'right') if dimension<2 else 'center', va='center',
                            fontsize=9.5, color='#252A33', zorder=100002, annotation_clip=False)
        label.set_gid('axis-label-'+name)
        definitions.append({'name':name, 'symbol':symbol, 'dimension':dimension,
                            'endpoints':endpoints.tolist(), 'origin':[0.,0.,0.],
                            'ticks':tick_values[dimension], 'zero_tick_shared_at_origin':True,
                            'tick_labels':[f'{v:+.0f}' if v else '0' for v in tick_values[dimension]],
                            'SVG_axis_id':'axis-'+name,
                            'projected_endpoints':xy.tolist()})
    origin = ax.scatter([0], [0], [0], color='#252A33', marker='+', s=36,
                        linewidths=.9, depthshade=False, zorder=100003)
    origin.set_gid('origin-reference')
    zero = ax.annotate('0', origin_projected[:2], xytext=(-9, 8),
                       textcoords='offset points', ha='right', va='bottom',
                       fontsize=8.5, color='#252A33', zorder=100004, annotation_clip=False)
    zero.set_gid('origin-label')
    return definitions


SUPPORT_POWER = 15
DEPTH_SLABS = 12
FOG_GRID_POINTS = .60
FOG_SIGMA_POINTS = 2.8
MEAN_OPTICAL_DEPTH = .85
CONTOUR_LEVELS = 48
FINAL_SOFTEN_POINTS = .20
DOT_RADIUS_POINTS = 3.0
RADIAL_STOPS = np.linspace(0.,1.,41)
DOT_KERNEL = np.exp(-4.5*RADIAL_STOPS**2)
DOT_FADE = np.clip((1.-RADIAL_STOPS)/.25,0.,1.)
DOT_KERNEL *= DOT_FADE**2*(3.-2.*DOT_FADE)


def volume_support(vertices,triangles,detail,observations):
    """Uniform deterministic integration support, never extra observations."""
    center=np.asarray(detail['normalization_mean']);std=np.asarray(detail['normalization_sample_std'])
    tetrahedra=vertices[triangles]-center
    volumes=np.abs(np.linalg.det(tetrahedra))/6.
    cumulative=np.cumsum(volumes)/volumes.sum()
    unit=qmc.Sobol(5,scramble=False).random_base2(SUPPORT_POWER)
    unit=np.minimum(unit+.5/len(unit),1.-np.finfo(float).eps)
    selected=np.minimum(np.searchsorted(cumulative,unit[:,0],side='right'),len(triangles)-1)
    barycentric=-np.log(unit[:,1:]);barycentric/=barycentric.sum(axis=1,keepdims=True)
    positions=center+np.einsum('ni,nij->nj',barycentric[:,1:],tetrahedra[selected])
    field=envelope_field((positions-center)/std,detail)
    interior=np.maximum(detail['level']-field,0.)/(detail['level']-envelope_field(np.zeros((1,3)),detail)[0])
    fade=-np.expm1(-(interior/.38)**2)
    kde=gaussian_kde(observations.T,bw_method='scott')
    log_density=kde.logpdf(positions.T);lo,hi=np.quantile(log_density,[.05,.95])
    concentration=np.clip((log_density-lo)/(hi-lo),0.,1.)
    weight=fade*(.8+.2*concentration)
    if (field>detail['level']+1e-10).any():raise ValueError('Fog supports must remain inside empirical envelope')
    return positions,{'tetrahedron_index':selected,'barycentric_weights':barycentric,
        'interior_depth':interior,'boundary_fade':fade,'log_KDE':log_density,
        'concentration':concentration,'weight':weight}, {'Scott_factor':float(kde.factor),
        'kernel_covariance':kde.covariance.tolist(),'support_log_KDE_Q05_Q95':[float(lo),float(hi)],
        'concentration_weight':'.8+.2*clip((log_KDE-Q05)/(Q95-Q05),0,1)',
        'boundary_fade':'1-exp(-(max(level-F,0)/(0.38*(level-F(0))))**2)',
        'tetrahedron_total_volume':float(volumes.sum())}


def gaussian_fog_field(xy,mass):
    """Integrate projected support mass on a physical-point Gaussian grid."""
    padding=5.*FOG_SIGMA_POINTS
    low=np.floor((xy.min(axis=0)-padding)/FOG_GRID_POINTS)*FOG_GRID_POINTS
    high=np.ceil((xy.max(axis=0)+padding)/FOG_GRID_POINTS)*FOG_GRID_POINTS
    x=np.arange(low[0],high[0]+.5*FOG_GRID_POINTS,FOG_GRID_POINTS)
    y=np.arange(low[1],high[1]+.5*FOG_GRID_POINTS,FOG_GRID_POINTS)
    ix=np.floor((xy[:,0]-low[0])/FOG_GRID_POINTS+.5).astype(int)
    iy=np.floor((xy[:,1]-low[1])/FOG_GRID_POINTS+.5).astype(int)
    histogram=np.zeros((len(y),len(x)));np.add.at(histogram,(iy,ix),mass)
    tau=gaussian_filter(histogram,FOG_SIGMA_POINTS/FOG_GRID_POINTS,
                        mode='constant',cval=0.,truncate=4.)/FOG_GRID_POINTS**2
    if not np.isclose(tau.sum()*FOG_GRID_POINTS**2,mass.sum(),rtol=1e-10,atol=1e-10):
        raise ValueError('Gaussian fog field lost projected optical mass')
    generator=contourpy.contour_generator(x=x,y=y,z=tau,fill_type='OuterCode')
    step=float(tau.max())/CONTOUR_LEVELS
    paths=[];thresholds=[]
    for threshold in (np.arange(CONTOUR_LEVELS)+.5)*step:
        vertices,codes=generator.filled(float(threshold),float(tau.max())+1.)
        for verts,code in zip(vertices,codes):
            path=MplPath(verts,code)
            paths.append(path);thresholds.append(float(threshold))
    return x,y,histogram,tau,paths,np.asarray(thresholds),step


def soften_layer(image,dpi,sigma_points):
    """Gaussian-filter premultiplied color, preserving soft transparent edges."""
    sigma=sigma_points*dpi/72.;pad=int(np.ceil(3*sigma))+1
    image=np.pad(image,((pad,pad),(pad,pad),(0,0)))
    alpha=gaussian_filter(image[:,:,3],sigma)
    rgb=np.stack([gaussian_filter(image[:,:,i]*image[:,:,3],sigma) for i in range(3)],axis=2)
    rgb/=np.maximum(alpha[:,:,None],1e-12)
    return np.dstack((rgb,alpha)),-pad,-pad


class _FogField(Artist):
    """Continuous vector opacity contours, without visible support particles."""
    def __init__(self,paths,color,step,gid):
        super().__init__();self.paths=paths;self.color=np.asarray(to_rgba(color))[:3]
        self.step=step;self.set_gid(gid);self.set_clip_on(False)

    def draw(self,renderer):
        svg=hasattr(renderer,'writer');renderer.open_group('aerosol-field',gid=self.get_gid())
        if svg:
            writer=renderer.writer;name=self.get_gid()+'-soften'
            writer.start('defs');writer.start('filter',id=name,x='-10%',y='-10%',width='120%',height='120%',
                **{'color-interpolation-filters':'sRGB'})
            writer.element('feGaussianBlur',stdDeviation=str(FINAL_SOFTEN_POINTS));writer.end('filter');writer.end('defs')
            writer.start('g',filter='url(#'+name+')')
        else:renderer.start_filter()
        gc=renderer.new_gc();gc.set_clip_rectangle(None);gc.set_linewidth(0.)
        gc.set_alpha(float(-np.expm1(-self.step)));gc.set_antialiased(False)
        transform=Affine2D().scale(renderer.points_to_pixels(1.))
        for path in self.paths:renderer.draw_path(gc,path,transform,rgbFace=tuple(self.color))
        gc.restore()
        if svg:renderer.writer.end('g')
        else:renderer.stop_filter(lambda im,dpi:soften_layer(im,dpi,FINAL_SOFTEN_POINTS))
        renderer.close_group('aerosol-field');self.stale=False


class _SoftObservedDot(Artist):
    """One flat radial-opacity circle at one unchanged measured center."""
    def __init__(self,xy,radius,color,alpha,gid):
        super().__init__();self.xy=np.asarray(xy);self.radius=float(radius)
        self.color=np.asarray(to_rgba(color))[:3];self.alpha=float(alpha)
        self.set_gid(gid);self.set_clip_on(False)

    def draw(self,renderer):
        renderer.open_group('observed-dot',gid=self.get_gid())
        if hasattr(renderer,'writer'):
            writer=renderer.writer;x,y=self.xy;x=float(x);y=self.figure.get_figheight()*72.-float(y)
            name=self.get_gid()+'-radial-opacity'
            writer.start('defs');writer.start('radialGradient',id=name,cx='50%',cy='50%',r='50%')
            from matplotlib.colors import to_hex
            for stop,opacity in zip(RADIAL_STOPS,DOT_KERNEL*self.alpha):
                writer.element('stop',offset=str(float(stop)),style='stop-color:'+to_hex(self.color)+';stop-opacity:'+str(float(opacity)))
            writer.end('radialGradient');writer.end('defs')
            writer.element('circle',cx=str(x),cy=str(y),r=str(self.radius),fill='url(#'+name+')',stroke='none')
        else:
            scale=renderer.points_to_pixels(1.);center=self.xy*scale;radius=self.radius*scale
            x0=int(np.floor(center[0]-radius-1));x1=int(np.ceil(center[0]+radius+1))
            y0=int(np.floor(center[1]-radius-1));y1=int(np.ceil(center[1]+radius+1))
            xx,yy=np.meshgrid((np.arange(x0,x1)+.5-center[0])/radius,(np.arange(y0,y1)+.5-center[1])/radius)
            alpha=np.interp(np.hypot(xx,yy),RADIAL_STOPS,DOT_KERNEL*self.alpha,right=0.)
            rgba=np.empty(alpha.shape+(4,));rgba[:,:,:3]=self.color;rgba[:,:,3]=alpha
            gc=renderer.new_gc();gc.set_clip_rectangle(None)
            renderer.draw_image(gc,x0,y0,np.uint8(np.round(np.clip(rgba,0,1)*255)));gc.restore()
        renderer.close_group('observed-dot');self.stale=False


def draw(points):
    """Project empirical fog geometry and preserve every real signed error dot."""
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':9,'legend.fontsize':9.5,'legend.frameon':False,'axes.linewidth':.7,
        'svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'gift-w3-aerosol-soft-dots',
        'figure.facecolor':'white','savefig.facecolor':'white'})
    envelopes={m:smooth_envelope(points[m]) for m in METHODS};limits=COMMON_LIMITS.copy()
    for method in METHODS:
        bounds=np.asarray(envelopes[method][2]['outer_bounds'])
        if not ((bounds>=limits[:,0])&(bounds<=limits[:,1])).all():raise ValueError('Axes clip '+method)
    ticks=[[-100.,0.,100.,200.,300.,400.],[-250.,-150.,-50.,0.,50.,100.,150.],[-100.,-50.,0.,30.]]
    fig=plt.figure(figsize=(270/25.4,220/25.4))
    ax=fig.add_axes([.035,.08,.78,.84],projection='3d',computed_zorder=False)
    ax.set_proj_type('persp',focal_length=1.15);ax.view_init(elev=30,azim=45)
    ax.set_box_aspect(BOX_ASPECT,zoom=1.25)
    ax.set(xlim=limits[0],ylim=limits[1],zlim=limits[2]);ax.set_axis_off();ax.grid(False)
    fig.canvas.draw();projection=ax.get_proj()
    def project(values):
        p=np.array(proj3d.proj_transform(*values.T,projection)).T
        xy=ax.transData.transform(p[:,:2])*(72./fig.dpi)
        return np.column_stack((xy,p[:,2]))
    supports={};support_arrays={};support_detail={};projected={};meshes={}
    for method in METHODS:
        vertices,triangles,detail=envelopes[method]
        supports[method],support_arrays[method],support_detail[method]=volume_support(vertices,triangles,detail,points[method])
        projected[method]=project(supports[method]);meshes[method+'_vertices']=vertices;meshes[method+'_triangles']=triangles
    all_depth=np.concatenate([projected[m][:,2] for m in METHODS])
    slab_edges=np.linspace(float(all_depth.min()),float(all_depth.max()),DEPTH_SLABS+1)
    data={'shared_depth_slab_edges':slab_edges,'dot_radial_stops':RADIAL_STOPS,'dot_radial_opacity':DOT_KERNEL}
    metadata={'methods':{},'text_bounds':[]};artists=[];layers=[]
    for method in METHODS:
        vertices,triangles,detail=envelopes[method];p=projected[method];weight=support_arrays[method]['weight']
        area=float(ConvexHull(project(vertices)[:,:2]).volume)
        mass=MEAN_OPTICAL_DEPTH*area*weight/weight.sum()
        slab=np.clip(np.searchsorted(slab_edges,p[:,2],side='right')-1,0,DEPTH_SLABS-1)
        for key,array in support_arrays[method].items():data[method+'_'+key]=array
        data[method+'_positions']=supports[method];data[method+'_projected_point_coordinates']=p
        data[method+'_optical_mass']=mass;data[method+'_depth_slab']=slab
        method_layers=[]
        for index in range(DEPTH_SLABS):
            selected=np.flatnonzero(slab==index)
            if not len(selected):continue
            x,y,histogram,tau,paths,thresholds,step=gaussian_fog_field(p[selected,:2],mass[selected])
            gid='fog-'+method.lower()+'-depth_'+f'{index:02d}'
            mean_depth=float(np.average(p[selected,2],weights=mass[selected]))
            artist=_FogField(paths,STYLES[method][0],step,gid)
            artists.append((mean_depth,artist,'fog',len(layers)))
            prefix=method+'_slab_'+f'{index:02d}'
            for key,array in {'x':x,'y':y,'histogram':histogram,'tau':tau,'thresholds':thresholds}.items():data[prefix+'_'+key]=array
            path_sizes=np.array([len(path.vertices) for path in paths],dtype=int)
            data[prefix+'_path_sizes']=path_sizes
            data[prefix+'_path_vertices']=np.concatenate([path.vertices for path in paths])
            data[prefix+'_path_codes']=np.concatenate([path.codes for path in paths])
            row={'gid':gid,'method':method,'depth_slab':index,'sample_count':len(selected),
                'representative_projected_depth':mean_depth,'optical_mass':float(mass[selected].sum()),
                'integrated_field_mass':float(tau.sum()*FOG_GRID_POINTS**2),'tau_max':float(tau.max()),
                'contour_optical_step':step,'contour_incremental_alpha':float(-np.expm1(-step)),
                'path_count':len(paths),'prefix':prefix}
            layers.append(row);method_layers.append(len(layers)-1)
        metadata['methods'][method]={'point_count':50,'observed_point_count':50,'coordinates':points[method].tolist(),
            'coordinate_transform':'none','color':STYLES[method][0],'marker':'soft flat circle','envelope':detail,
            'display_limits':limits.tolist(),'display_ticks':ticks,'geometric_support_count':len(p),
            'projected_silhouette_area_points_squared':area,'total_optical_mass':float(mass.sum()),
            'support':support_detail[method],'fog_layer_indices':method_layers}
    measured=np.concatenate([points[m] for m in METHODS]);observed=project(measured)
    homogeneous=np.column_stack((measured,np.ones(250)))@projection.T
    reference_w=float(np.median(homogeneous[:,3]));radii=DOT_RADIUS_POINTS*reference_w/homogeneous[:,3]
    near=(observed[:,2].max()-observed[:,2])/(observed[:,2].max()-observed[:,2].min())
    alphas=.84+.12*near
    for index in range(250):
        method=METHODS[index//50];gid='observed-'+method.lower()+'-parameter_'+f'{index%50:03d}'
        artist=_SoftObservedDot(observed[index,:2],radii[index],STYLES[method][0],alphas[index],gid)
        artists.append((float(observed[index,2]),artist,'observed',index))
    order=np.argsort(-np.array([v[0] for v in artists]),kind='stable')
    for rank,index in enumerate(order):
        artist=artists[index][1];artist.set_zorder(100+rank);ax.add_artist(artist)
    data.update({'observed_projected_point_coordinates':observed,'observed_radius_points':radii,
        'observed_alpha':alphas,'observed_near_factor':near,'joint_painter_order':order,
        'joint_item_depth':np.array([v[0] for v in artists]),
        'joint_item_type':np.array([0 if v[2]=='fog' else 1 for v in artists]),
        'joint_item_index':np.array([v[3] for v in artists])})
    metadata['axis_definitions']=draw_origin_axes(ax,limits,ticks)
    fig.text(.415,.95,'Signed relative parameter errors',ha='center',va='center',fontsize=12)
    handles=[Line2D([],[],color=STYLES[m][0],marker='o',linestyle='none',markersize=5,label=m) for m in METHODS]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.80,.84),borderaxespad=0,labelspacing=1.1,handletextpad=.7)
    fig.text(.805,.57,'50 parameter combinations\nper method',va='top',fontsize=9,linespacing=1.6)
    fig.text(.805,.45,'Positive: overestimate\nNegative: underestimate\n\n+  Ideal identification\n\nDots: observed cases\nMist: empirical envelope',va='top',
             fontsize=8.5,linespacing=1.7,color='#4B525D')
    fig.text(.415,.055,'250 observed cases; continuous mist depicts geometry, not additional evaluations.',
             ha='center',va='center',fontsize=8.2,color='#4B525D')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for artist in fig.findobj(match=matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():metadata['text_bounds'].append({'text':artist.get_text(),'bbox_pixels':list(artist.get_window_extent(renderer).bounds)})
    metadata.update({'width_mm':270,'height_mm':220,'box_aspect':BOX_ASPECT.tolist(),
        'shared_display_limits':limits.tolist(),'panels':['joint'],'data_panel_count':1,
        'camera':{'projection':'persp','focal_length':1.15,'elev':30,'azim':45,'box_zoom':1.25},
        'coordinate_definition':'100 * (estimate - truth) / truth','coordinate_units':'percent',
        'axis_intersection':[0.,0.,0.],'projection_matrix':projection.tolist(),
        'axes_data_to_display_matrix':ax.transData.get_affine().get_matrix().tolist(),
        'projected_origin':list(proj3d.proj_transform(0.,0.,0.,projection)),
        'figure_dpi':fig.dpi,'canvas_pixels':list(fig.canvas.get_width_height()),
        'grid_visible':False,'panes_visible':False,'bounding_box_visible':False,
        'axes_overlay_is_reference_annotation':True,'axis_lines_through_origin':True,
        'tick_labels_preserved':True,'no_jitter':True,'no_outlier_removal':True,
        'observed_coordinates_smoothed':False,'envelope_smoothed':True,
        'envelope_is_confidence_region':False,'envelope_volume_is_accuracy_score':False,
        'mesh_file':'envelope_meshes.npz','aerosol_render_file':'aerosol_render_data.npz',
        'geometric_support_count':5*2**SUPPORT_POWER,'solid_faces_drawn':0,'outline_count':0,'section_curve_count':0,
        'fog_rendering':{'sampling':'5D unscrambled Sobol, half-cell shifted; volume-weighted tetrahedra and normalized exponential barycentric weights',
            'support_power':SUPPORT_POWER,'supports_per_method':2**SUPPORT_POWER,'supports_are_observations':False,
            'individual_support_particles_drawn':False,'reconstruction':'projected weighted histogram followed by normalized Gaussian convolution',
            'grid_spacing_points':FOG_GRID_POINTS,'Gaussian_sigma_points':FOG_SIGMA_POINTS,'Gaussian_truncate_sigma':4.,
            'grid_padding_sigma':5.,'mean_optical_depth_per_projected_silhouette':MEAN_OPTICAL_DEPTH,
            'visibility_normalization':'total support optical mass = common mean optical depth times projected full-mesh silhouette area',
            'concentration':'within-method 50-point full-covariance Scott KDE log-Q05/Q95; weight .8+.2*concentration',
            'boundary_fade_fraction':.38,'depth_slabs':DEPTH_SLABS,
            'composition':'approximate common projected-depth slabs, representative mass-weighted mean depth, jointly ordered with all 250 measured centers',
            'optical_to_opacity':'1-exp(-tau)', 'vector_approximation':f'{CONTOUR_LEVELS} nested midpoint optical-depth contour levels per field; no contour lines',
            'maximum_optical_quantization_error':'tau_max/(2*contour_levels) for each field',
            'contour_levels':CONTOUR_LEVELS,'final_field_soften_sigma_points':FINAL_SOFTEN_POINTS,
            'solid_surface_drawn':False,'geometry_changed_for_appearance':False,'layers':layers},
        'observation_rendering':{'point_count':250,'glyph_count':250,'all_methods_marker':'soft flat circle',
            'base_outer_radius_points':DOT_RADIUS_POINTS,'perspective_radius_rule':'radius * median(observed clip-w) / own clip-w',
            'reference_clip_w':reference_w,'alpha_rule':'.84+.12*near',
            'radial_opacity':'exp(-4.5*r**2) times smoothstep((1-r)/.25), sampled at 41 stops on normalized radius [0,1]',
            'color_constant_within_dot':True,'highlight':False,'stroke':False,'new_measured_centers':0},
        'painter_order_gids':[artists[i][1].get_gid() for i in order],
        'palette_meaning':'method hue identity; fog concentration is qualitative within method, not cross-method accuracy',
        'fog_is_probability_or_confidence':False,'fog_intensity_is_accuracy':False})
    return fig,metadata,meshes,data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preview', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() or args.preview.exists():
        raise FileExistsError('Use a new output directory and preview filename')
    cases, points = load_points(args.source)
    fig, metadata, meshes, render_data = draw(points)
    args.output_dir.mkdir(parents=True); args.preview.parent.mkdir(parents=True, exist_ok=True)
    svg = args.output_dir/'w3_parameter_joint_errors.svg'
    fig.savefig(svg, metadata={'Date':None}, dpi=600); fig.savefig(args.preview, dpi=300)
    data = args.output_dir/'source_points.csv'
    with data.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['case_id','method','nu_signed_relative_error_percent',
                         'beta_signed_relative_error_percent','gamma_signed_relative_error_percent'])
        for method in METHODS:
            for case, values in zip(cases, points[method]):
                writer.writerow([case,method,*values])
    mesh_file = args.output_dir/'envelope_meshes.npz'; np.savez_compressed(mesh_file, **meshes)
    surface_file = args.output_dir/'aerosol_render_data.npz'; np.savez_compressed(surface_file, **render_data)
    metadata.update({'case_ids':cases, 'source_file_sha256':digest(args.source),
                     'plot_source_sha256':digest(Path(__file__)), 'mesh_file_sha256':digest(mesh_file),
                     'aerosol_render_file_sha256':digest(surface_file)})
    (args.output_dir/'figure_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    plt.close(fig)
    print(json.dumps({'svg':str(svg), 'preview':str(args.preview), 'points':250,
                      'source_sha256':metadata['source_file_sha256'],
                      'limits':{m:metadata['methods'][m]['display_limits'] for m in METHODS}}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
