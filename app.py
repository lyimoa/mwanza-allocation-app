# Mwanza multi-stage healthcare allocation - interactive demo
# DSAI 6220 Advanced Machine Learning, Group 1 (NM-AIST)
#
# Reads the tables produced by the Colab notebook and re-runs the two
# optimization stages live, so the budgets can be changed with sliders.
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pulp
import streamlit as st
from scipy.spatial import cKDTree

TILE_M = 640                         # one tile = 640 m x 640 m
CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
CLASS_COLOURS = ['#e9c46a', '#1b4332', '#95d5b2', '#6c757d', '#9d4edd',
                 '#b7e4c7', '#f4a261', '#d62828', '#48cae4', '#023e8a']
NAVY, TEAL, MINT, INK, GREY = '#1F3A5F', '#1A7F6E', '#D8F0EA', '#1C2430', '#5A6472'
H_MAIN, H_SMALL = 540, 340           # every chart and table in a row shares one height

# Stage 2 assumptions (fixed; shown in the app for transparency)
CAPS = {'Dispensary/Clinic': 10_000, 'Health Centre': 50_000, 'Hospital': 150_000}
COST_STAFF, CAP_STAFF, MAX_STAFF = 1, 5_000, 4
COST_UPG, CAP_UPG = 6, 40_000
CHOICES = 5

st.set_page_config(page_title='Mwanza healthcare allocation', page_icon='🛰️', layout='wide')

st.markdown(f"""
<style>
  .stApp, [data-testid="stHeader"] {{ background: #FFFFFF; }}
  .block-container {{ padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1400px; }}
  [data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid #E3E8EF; }}
  h1, h2, h3 {{ color: {NAVY}; letter-spacing: -0.01em; }}
  .app-title {{ font-size: 2.0rem !important; font-weight: 800; color: {NAVY}; line-height: 1.2; margin: 4px 0 0 0; }}
  .app-sub {{ color: {GREY}; font-size: 0.95rem; margin-top: 0.25rem; }}
  .app-tag {{ color: {TEAL}; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }}
  [data-testid="stMetric"] {{ background: #FFFFFF; border: 1px solid #E3E8EF; border-radius: 10px;
                              padding: 14px 16px; box-shadow: 0 1px 2px rgba(16,24,40,0.04); }}
  [data-testid="stMetricLabel"] {{ color: {GREY}; }}
  [data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 700; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid #E3E8EF; }}
  .stTabs [data-baseweb="tab"] {{ padding: 10px 16px; font-weight: 600; color: {GREY}; }}
  .stTabs [aria-selected="true"] {{ color: {TEAL}; }}
  .panel-title {{ font-weight: 700; color: {NAVY}; font-size: 1.0rem; margin: 0.4rem 0 0.35rem 0; }}
  .step {{ border: 1px solid #E3E8EF; border-radius: 10px; padding: 16px; height: 100%; background: #FFFFFF; }}
  .step b {{ color: {NAVY}; }}
  .step .n {{ display: inline-block; width: 28px; height: 28px; line-height: 28px; text-align: center;
              border-radius: 50%; background: {TEAL}; color: #fff; font-weight: 700; margin-bottom: 8px; }}
  .step p {{ color: {INK}; font-size: 0.92rem; margin: 6px 0 0 0; }}
  .note {{ background: {MINT}; border-radius: 10px; padding: 12px 16px; color: {INK}; font-size: 0.95rem; }}
  .assume {{ color: {GREY}; font-size: 0.85rem; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def find(name):
    for folder in ('data', 'Data', '.'):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p
    st.error(f'Could not find **{name}**. Put it in a folder called `data` next to app.py.')
    st.stop()


@st.cache_data
def load_data():
    tiles = pd.read_csv(find('mwanza_tiles_access.csv'))
    fac = pd.read_csv(find('mwanza_health_facilities_osm.csv'))
    fac['name'] = fac['name'].fillna('Unnamed')
    if 'x_m' not in fac.columns:                       # fallback: GPS -> UTM 36S
        from pyproj import Transformer
        t = Transformer.from_crs('EPSG:4326', 'EPSG:32736', always_xy=True)
        fac['x_m'], fac['y_m'] = t.transform(fac.lon.values, fac.lat.values)
    tiles['population'] = tiles['population'].fillna(0)
    return tiles, fac.reset_index(drop=True)


tiles, fac = load_data()
N_ROWS, N_COLS = int(tiles.row.max()) + 1, int(tiles.col.max()) + 1
X0 = float(np.median(tiles.x_m - (tiles.col + 0.5) * TILE_M))   # grid corner, metres
Y0 = float(np.median(tiles.y_m + (tiles.row + 0.5) * TILE_M))


def to_grid(values, frame=tiles):
    g = np.full((N_ROWS, N_COLS), np.nan)
    g[frame.row.values, frame.col.values] = values
    return g


def grid_xy(x_m, y_m):
    return (np.asarray(x_m) - X0) / TILE_M - 0.5, (Y0 - np.asarray(y_m)) / TILE_M - 0.5


POP_GRID = to_grid(tiles.population.values)
POP_LOG = np.where(POP_GRID > 0, np.log10(np.where(POP_GRID > 0, POP_GRID, 1)), np.nan)   # empty tiles stay white


# ----------------------------------------------------------------------------
# Chart helpers (every map is a Plotly figure with a fixed height)
# ----------------------------------------------------------------------------
def map_layout(fig, height=H_MAIN, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=46 if legend else 8), template='plotly_white',
        paper_bgcolor='white', plot_bgcolor='white', showlegend=legend,
        legend=dict(orientation='h', yanchor='top', y=-0.01, xanchor='left', x=0,
                    bgcolor='rgba(255,255,255,0)', borderwidth=0,
                    font=dict(size=11, color=INK)),
        xaxis=dict(visible=False, range=[-0.5, N_COLS - 0.5], constrain='domain'),
        yaxis=dict(visible=False, range=[N_ROWS - 0.5, -0.5], scaleanchor='x', constrain='domain'))
    return fig


def base_layer(fig, opacity=0.45):
    fig.add_trace(go.Heatmap(z=POP_LOG, colorscale='Greys', showscale=False,
                             opacity=opacity, hoverinfo='skip'))


def facility_layer(fig, frame, name='Existing facilities', size=6):
    fc, fr = grid_xy(frame.x_m, frame.y_m)
    fig.add_trace(go.Scatter(x=fc, y=fr, mode='markers', name=name, text=frame.name,
                             hovertemplate='%{text}<extra></extra>',
                             marker=dict(size=size, color='#22D3EE', line=dict(color='#0F172A', width=0.6))))


def chart_layout(fig, height=H_SMALL):
    fig.update_layout(height=height, margin=dict(l=10, r=20, t=10, b=40), template='plotly_white',
                      paper_bgcolor='white', plot_bgcolor='white', showlegend=False,
                      font=dict(color=INK, size=12))
    return fig


def show(fig):
    st.plotly_chart(fig, width='stretch', theme=None, config={'displayModeBar': False})


def table(frame, height=H_MAIN):
    st.dataframe(frame, hide_index=True, width='stretch', height=height)


def panel(title):
    st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Stage 1: access (maximal covering location problem)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def stage1_inputs(limit_km):
    dist = cKDTree(fac[['x_m', 'y_m']].values).query(tiles[['x_m', 'y_m']].values)[0] / 1000
    under = (tiles.population > 0) & (dist > limit_km)
    dem = tiles[under].reset_index(drop=True)
    land = tiles[~tiles.land_class.isin(['SeaLake', 'River'])].reset_index(drop=True)
    if len(dem) == 0:
        return dist, dem, land.iloc[:0], []
    near = cKDTree(land[['x_m', 'y_m']].values).query_ball_point(
        dem[['x_m', 'y_m']].values, r=limit_km * 1000)
    cand = land.loc[sorted({j for lst in near for j in lst})].reset_index(drop=True)
    cover = cKDTree(cand[['x_m', 'y_m']].values).query_ball_point(
        dem[['x_m', 'y_m']].values, r=limit_km * 1000)
    return dist, dem, cand, [list(c) for c in cover]


@st.cache_data(show_spinner=False)
def solve_stage1(K, limit_km):
    """Choose K sites that cover the most underserved people within limit_km."""
    _, dem, cand, cover = stage1_inputs(limit_km)
    if K == 0 or len(dem) == 0:
        return [], 0.0
    prob = pulp.LpProblem('Stage1_Access', pulp.LpMaximize)
    build = pulp.LpVariable.dicts('build', range(len(cand)), cat='Binary')
    covered = pulp.LpVariable.dicts('covered', range(len(dem)), cat='Binary')
    prob += pulp.lpSum(float(dem.population[i]) * covered[i] for i in range(len(dem)))
    for i in range(len(dem)):
        prob += covered[i] <= pulp.lpSum(build[j] for j in cover[i])
    prob += pulp.lpSum(build.values()) <= K
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
    chosen = [j for j in range(len(cand)) if (build[j].value() or 0) > 0.5]
    people = sum(float(dem.population[i]) for i in range(len(dem))
                 if (covered[i].value() or 0) > 0.5)
    return chosen, people


def covered_pop(xy, dem, limit_km):
    if len(xy) == 0 or len(dem) == 0:
        return 0.0
    idx = set()
    for lst in cKDTree(dem[['x_m', 'y_m']].values).query_ball_point(xy, r=limit_km * 1000):
        idx.update(lst)
    return float(dem.population.values[list(idx)].sum())


# ----------------------------------------------------------------------------
# Stage 2: capacity (clinicians + upgrades), given the Stage 1 sites
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def solve_stage2(K, limit_km, B, allow_staff=True, allow_upgrade=True):
    chosen, _ = solve_stage1(K, limit_km)
    _, _, cand, _ = stage1_inputs(limit_km)
    new = cand.loc[chosen, ['x_m', 'y_m', 'lat', 'lon']].copy()
    new['name'] = [f'New Dispensary S1-{i + 1}' for i in range(len(new))]
    new['level'] = 'Dispensary/Clinic'
    new['stage'] = 1
    keep = ['name', 'level', 'lat', 'lon', 'x_m', 'y_m', 'stage']
    fac2 = pd.concat([fac.assign(stage=0)[keep], new[keep]], ignore_index=True)
    fac2['capacity'] = fac2.level.map(CAPS).fillna(CAPS['Dispensary/Clinic'])

    pt = tiles[tiles.population > 0].reset_index(drop=True)
    k = min(CHOICES, len(fac2))
    d, ix = cKDTree(fac2[['x_m', 'y_m']].values).query(pt[['x_m', 'y_m']].values, k=k)
    d, ix = d.reshape(len(pt), -1), ix.reshape(len(pt), -1)
    options = []
    for dr, fr in zip(d, ix):
        ok = [int(f) for dd, f in zip(dr, fr) if dd <= limit_km * 1000]
        options.append(ok if ok else [int(fr[0])])

    prob = pulp.LpProblem('Stage2_Capacity', pulp.LpMinimize)
    I, F = range(len(pt)), range(len(fac2))
    send = {(i, f): pulp.LpVariable(f's_{i}_{f}', lowBound=0) for i in I for f in options[i]}
    unmet = pulp.LpVariable.dicts('unmet', I, lowBound=0)
    staff = pulp.LpVariable.dicts('staff', F, 0, MAX_STAFF if allow_staff else 0, cat='Integer')
    upg = pulp.LpVariable.dicts('upg', F, cat='Binary')
    spend = pulp.lpSum(COST_STAFF * staff[f] + COST_UPG * upg[f] for f in F)
    prob += pulp.lpSum(unmet.values()) + 0.001 * spend       # fewest unserved, then cheapest
    pop = pt.population.values
    for i in I:
        prob += pulp.lpSum(send[i, f] for f in options[i]) + unmet[i] == float(pop[i])
    users = {f: [] for f in F}
    for (i, f) in send:
        users[f].append(i)
    for f in F:
        prob += (pulp.lpSum(send[i, f] for i in users[f])
                 <= float(fac2.capacity[f]) + CAP_STAFF * staff[f] + CAP_UPG * upg[f])
        if fac2.level[f] != 'Dispensary/Clinic' or not allow_upgrade:
            prob += upg[f] == 0
    prob += spend <= B
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=90))

    plan = fac2.copy()
    plan['served'] = [sum((send[i, f].value() or 0) for i in users[f]) for f in F]
    plan['extra_staff'] = [int(round(staff[f].value() or 0)) for f in F]
    plan['upgrade'] = [int(round(upg[f].value() or 0)) for f in F]
    pt = pt[['row', 'col']].copy()
    pt['unmet'] = [unmet[i].value() or 0 for i in I]
    return plan, pt, pulp.LpStatus[prob.status]


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
PRESETS = {'Recommended plan': (6, 16), 'Tight budget': (3, 8), 'Generous budget': (10, 30),
           'No intervention': (0, 0)}
ACTIONS = {'Clinicians and upgrades': (True, True), 'Clinicians only': (True, False),
           'Upgrades only': (False, True)}

with st.sidebar:
    st.markdown(f'<div class="app-tag">Decision controls</div>', unsafe_allow_html=True)
    preset = st.selectbox('Scenario', list(PRESETS) + ['Custom'], index=0,
                          help='Pick a ready-made scenario, or choose Custom to set the budgets yourself.')
    k_def, b_def = PRESETS.get(preset, (6, 16))
    custom = preset == 'Custom'
    with st.form('controls'):
        st.markdown('**Stage 1 · Access (Year 1)**')
        K = st.slider('New dispensaries to build', 0, 10, k_def, disabled=not custom)
        limit_km = st.slider('Access target (km to nearest facility)', 3.0, 10.0, 5.0, 0.5)
        st.markdown('**Stage 2 · Capacity (Year 2)**')
        B = st.slider('Budget (1 unit = 1 clinician per year)', 0, 40, b_def, disabled=not custom)
        action = st.radio('What may the Year-2 budget buy?', list(ACTIONS), index=0,
                          help='Compare what happens when only one kind of investment is allowed.')
        st.form_submit_button('Run optimization', type='primary', width='stretch')
    if not custom:
        K, B = k_def, b_def
    st.caption('Choose a scenario or set your own budgets, then press **Run optimization**.')
allow_staff, allow_upgrade = ACTIONS[action]

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
logo = next((p for p in ('logo.png', 'logo.jpg', 'logo.jpeg', 'data/logo.png', 'data/logo.jpg')
             if os.path.exists(p)), None)
if logo:
    hc = st.columns([1, 9], vertical_alignment='center')
    hc[0].image(logo, width=110)
    head = hc[1]
else:
    head = st.container()
head.markdown(
    '<div class="app-tag">The Nelson Mandela African Institution of Science and Technology · DSAI 6220 · Group 1</div>'
    '<div class="app-title">Satellite-Guided Multi-Stage Allocation of Primary Healthcare</div>'
    '<div class="app-sub">55 km × 55 km study area around Mwanza city, Tanzania · CNN land-use classification of '
    'Sentinel-2 imagery combined with two-stage optimization</div>', unsafe_allow_html=True)
st.write('')

# ----------------------------------------------------------------------------
# Compute
# ----------------------------------------------------------------------------
dist, dem, cand, _ = stage1_inputs(limit_km)
total_pop = float(tiles.population.sum())
under_before = float(dem.population.sum()) if len(dem) else 0.0
with st.spinner('Stage 1: choosing the best sites...'):
    chosen, covered_people = solve_stage1(K, limit_km)
new_sites = cand.loc[chosen].reset_index(drop=True)
all_xy = (np.vstack([fac[['x_m', 'y_m']].values, new_sites[['x_m', 'y_m']].values])
          if len(new_sites) else fac[['x_m', 'y_m']].values)
dist_after = cKDTree(all_xy).query(tiles[['x_m', 'y_m']].values)[0] / 1000
mask_before = ((tiles.population > 0) & (dist > limit_km)).values
mask_after = ((tiles.population > 0) & (dist_after > limit_km)).values
under_after = float(tiles.population[mask_after].sum())

tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ['Overview', 'Image analysis', 'Access gap', 'Stage 1 · Access', 'Stage 2 · Capacity'])

# ---------------------------- Overview --------------------------------------
with tab0:
    c = st.columns(4)
    c[0].metric('Tiles analysed (640 m each)', f'{len(tiles):,}')
    c[1].metric('Estimated population', f'{total_pop:,.0f}')
    c[2].metric('Existing health facilities', f'{len(fac):,}')
    c[3].metric(f'Underserved (beyond {limit_km:g} km)', f'{under_before:,.0f}',
                f'{100 * under_before / total_pop:.1f}% of people', delta_color='off')
    st.write('')
    steps = [('Image analysis', 'A CNN (EfficientNetB0 fine-tuned on EuroSAT, 95.9% test accuracy) labels every '
                                '640 m tile of a cloud-free Sentinel-2 image of Mwanza.'),
             ('Need assessment', 'Building counts give population per tile. Distance to the nearest facility shows '
                                 'who is underserved.'),
             ('Stage 1 · Access', 'With a Year-1 budget, the model places new dispensaries where they reach the most '
                                  'underserved people, only on land the CNN did not label as water.'),
             ('Stage 2 · Capacity', 'With a Year-2 budget, a second model adds clinicians or upgrades dispensaries '
                                    'where demand exceeds capacity, taking the Stage 1 sites as given.')]
    for col, (i, (t, d)) in zip(st.columns(4), enumerate(steps, 1)):
        col.markdown(f'<div class="step"><div class="n">{i}</div><br><b>{t}</b><p>{d}</p></div>',
                     unsafe_allow_html=True)
    st.write('')
    a, b_ = st.columns(2)
    with a:
        panel('Where people live (darker = more people; hover for numbers)')
        fig = go.Figure(go.Heatmap(z=POP_LOG, customdata=POP_GRID, colorscale='YlOrRd', showscale=False,
                                   hovertemplate='%{customdata:,.0f} people<extra></extra>'))
        show(map_layout(fig, legend=False))
    with b_:
        panel('What the CNN sees (land-use class of each tile)')
        codes = to_grid(tiles.land_class.map({n: i for i, n in enumerate(CLASSES)}).values)
        scale = [[i / 10 if j == 0 else (i + 1) / 10, CLASS_COLOURS[i]] for i in range(10) for j in (0, 1)]
        names = np.vectorize(lambda v: CLASSES[int(v)] if v == v else '')(codes)
        fig = go.Figure(go.Heatmap(z=codes, zmin=-0.5, zmax=9.5, colorscale=scale, showscale=False,
                                   customdata=names, hovertemplate='%{customdata}<extra></extra>'))
        show(map_layout(fig, legend=False))

# ---------------------------- Image analysis --------------------------------
with tab1:
    left, right = st.columns(2)
    with left:
        panel('CNN land-use classification (one square = one 640 m tile)')
        fig = go.Figure(go.Heatmap(z=codes, zmin=-0.5, zmax=9.5, colorscale=scale, showscale=False,
                                   customdata=names, hovertemplate='%{customdata}<extra></extra>'))
        for n, col in zip(CLASSES, CLASS_COLOURS):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=n,
                                     marker=dict(size=9, color=col, symbol='square')))
        show(map_layout(fig))
    with right:
        panel('Land-use classes, and a check against Google Open Buildings')
        s = tiles.groupby('land_class').agg(Tiles=('row', 'size')).reset_index()
        s['Area (km²)'] = (s.Tiles * 0.4096).round(1)
        s['Share (%)'] = (100 * s.Tiles / len(tiles)).round(1)
        if 'buildings' in tiles.columns:
            s['Mean buildings per tile'] = tiles.groupby('land_class').buildings.mean().round(1).values
        table(s.rename(columns={'land_class': 'CNN class'}).sort_values('Tiles', ascending=False))
    c = st.columns(3)
    c[0].metric('CNN test accuracy (EuroSAT)', '95.9%')
    if 'confidence' in tiles.columns:
        c[1].metric('Mean prediction confidence in Mwanza', f'{tiles.confidence.mean():.3f}')
    c[2].metric('AUC against building data', '0.913')

# ---------------------------- Access gap ------------------------------------
with tab2:
    a, b_ = st.columns(2)
    with a:
        panel('Population and existing health facilities')
        fig = go.Figure(go.Heatmap(z=POP_LOG, customdata=POP_GRID, colorscale='YlOrRd',
                                   showscale=False, hovertemplate='%{customdata:,.0f} people<extra></extra>'))
        for lv, sym, size in [('Dispensary/Clinic', 'circle', 6), ('Health Centre', 'square', 9),
                              ('Hospital', 'star', 13)]:
            sub = fac[fac.level == lv]
            fc, fr = grid_xy(sub.x_m, sub.y_m)
            fig.add_trace(go.Scatter(x=fc, y=fr, mode='markers', name=f'{lv} ({len(sub)})', text=sub.name,
                                     hovertemplate='%{text}<extra></extra>',
                                     marker=dict(size=size, symbol=sym, color='#0E7490',
                                                 line=dict(color='#FFFFFF', width=0.7))))
        show(map_layout(fig))
    with b_:
        panel(f'Distance to the nearest facility (green = near, red = far; black line = {limit_km:g} km)')
        dgrid = to_grid(dist)
        dshow = np.where(POP_GRID > 0, dgrid, np.nan)
        fig = go.Figure(go.Heatmap(z=dshow, zmin=0, zmax=12, colorscale='RdYlGn', reversescale=True,
                                   showscale=False,
                                   hovertemplate='%{z:.1f} km<extra></extra>'))
        fig.add_trace(go.Contour(z=dgrid, contours=dict(start=limit_km, end=limit_km, size=1, coloring='none'),
                                 line=dict(color='black', width=1.2), showscale=False, hoverinfo='skip'))
        show(map_layout(fig, legend=False))
    c = st.columns(3)
    c[0].metric(f'People within {limit_km:g} km of care', f'{100 * (1 - under_before / total_pop):.1f}%')
    c[1].metric('Underserved people', f'{under_before:,.0f}')
    c[2].metric('Underserved tiles', f'{int(mask_before.sum()):,}')

# ---------------------------- Stage 1 ---------------------------------------
with tab3:
    c = st.columns(4)
    c[0].metric('New dispensaries', K)
    c[1].metric('Underserved people covered', f'{covered_people:,.0f}',
                f'{100 * covered_people / under_before:.1f}% of underserved' if under_before else None,
                delta_color='off')
    c[2].metric('Still underserved', f'{under_after:,.0f}')
    c[3].metric(f'Access within {limit_km:g} km', f'{100 * (1 - under_after / total_pop):.1f}%',
                f'{100 * (under_before - under_after) / total_pop:+.1f} points')
    left, right = st.columns(2)
    with left:
        panel(f'Stage 1 plan: {K} new dispensaries with {limit_km:g} km coverage circles')
        status = np.where(mask_after, 1.0, np.where(mask_before, 0.0, np.nan))
        fig = go.Figure()
        base_layer(fig)
        fig.add_trace(go.Heatmap(z=to_grid(status), zmin=0, zmax=1, showscale=False, hoverinfo='skip',
                                 colorscale=[[0, '#F59E0B'], [0.5, '#F59E0B'], [0.5, '#DC2626'], [1, '#DC2626']]))
        for n, col in [('Underserved, now covered', '#F59E0B'), ('Still underserved', '#DC2626')]:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=n,
                                     marker=dict(size=10, color=col, symbol='square')))
        facility_layer(fig, fac)
        if len(new_sites):
            nc, nr = grid_xy(new_sites.x_m, new_sites.y_m)
            r = limit_km * 1000 / TILE_M
            for x, y in zip(nc, nr):
                fig.add_shape(type='circle', x0=x - r, x1=x + r, y0=y - r, y1=y + r,
                              line=dict(color=NAVY, width=1.3, dash='dash'))
            fig.add_trace(go.Scatter(x=nc, y=nr, mode='markers', name='New dispensaries',
                                     text=[f'S1-{i + 1}' for i in range(len(nc))],
                                     hovertemplate='%{text}<extra></extra>',
                                     marker=dict(size=17, symbol='star', color='#FACC15',
                                                 line=dict(color='#0F172A', width=1))))
        show(map_layout(fig))
    with right:
        panel('Sites chosen by the optimization')
        tbl = pd.DataFrame({
            'Site': [f'S1-{i + 1}' for i in range(len(new_sites))],
            'Latitude': new_sites.lat.round(4) if len(new_sites) else [],
            'Longitude': new_sites.lon.round(4) if len(new_sites) else [],
            'CNN land class': new_sites.land_class if len(new_sites) else [],
            'Underserved within reach': [round(covered_pop(new_sites[['x_m', 'y_m']].values[i:i + 1], dem, limit_km))
                                         for i in range(len(new_sites))]})
        table(tbl.sort_values('Underserved within reach', ascending=False))
    left, right = st.columns(2)
    with left:
        panel('Is optimization better than simple rules? (same number of dispensaries)')
        if K > 0 and len(dem):
            top = dem.nlargest(K, 'population')[['x_m', 'y_m']].values
            rng = np.random.default_rng(42)
            rnd = np.mean([covered_pop(cand.sample(min(K, len(cand)), random_state=int(sd))[['x_m', 'y_m']].values,
                                       dem, limit_km) for sd in rng.integers(0, 10 ** 6, 100)])
            vals = [covered_people, covered_pop(top, dem, limit_km), rnd]
        else:
            vals = [0, 0, 0]
        fig = go.Figure(go.Bar(x=vals, y=['Optimized (ours)', 'Most-populated tiles first', 'Random sites'],
                               orientation='h', marker_color=[TEAL, '#9BBFB8', '#C9CED6'],
                               text=[f'{v:,.0f}' for v in vals], textposition='outside',
                               hovertemplate='%{x:,.0f} people<extra></extra>'))
        fig.update_yaxes(autorange='reversed')
        fig.update_xaxes(title='Underserved people covered', range=[0, max(vals + [1]) * 1.22])
        show(chart_layout(fig))
    with right:
        panel('Coverage for each budget (diminishing returns)')
        with st.spinner('Solving for each budget...'):
            curve = [solve_stage1(k, limit_km)[1] for k in range(1, 11)]
        extra = np.diff([0] + curve)
        fig = go.Figure(go.Scatter(x=list(range(1, 11)), y=curve, mode='lines+markers', customdata=extra,
                                   line=dict(color=NAVY, width=2.5), marker=dict(size=8, color=NAVY),
                                   hovertemplate='%{x} dispensaries<br>%{y:,.0f} covered'
                                                 '<br>+%{customdata:,.0f} from the last one<extra></extra>'))
        if K >= 1:
            fig.add_vline(x=K, line_dash='dash', line_color=TEAL)
        fig.update_xaxes(title='New dispensaries', dtick=1)
        fig.update_yaxes(title='People covered')
        show(chart_layout(fig))

# ---------------------------- Stage 2 ---------------------------------------
with tab4:
    with st.spinner('Stage 2: allocating clinicians and upgrades (can take up to a minute)...'):
        plan0, pt0, _ = solve_stage2(K, limit_km, 0)
        plan, pt, status = solve_stage2(K, limit_km, B, allow_staff, allow_upgrade)
    gap0, gap = float(pt0.unmet.sum()), float(pt.unmet.sum())
    acts = plan[(plan.extra_staff > 0) | (plan.upgrade > 0)].copy()
    spent = int(COST_STAFF * acts.extra_staff.sum() + COST_UPG * acts.upgrade.sum())
    c = st.columns(4)
    c[0].metric('Capacity gap before Stage 2', f'{gap0:,.0f}')
    c[1].metric('Still unserved after Stage 2', f'{gap:,.0f}', f'{gap - gap0:,.0f}', delta_color='inverse')
    c[2].metric('Budget used', f'{spent} of {B} units')
    c[3].metric('Upgrades · clinicians', f'{int(acts.upgrade.sum())} · {int(acts.extra_staff.sum())}')
    left, right = st.columns(2)
    with left:
        panel(f'Stage 2 plan with {B} units ({action.lower()})')
        g = to_grid(pt0.unmet.values, pt0)
        g[g <= 0.5] = np.nan
        fig = go.Figure()
        base_layer(fig)
        fig.add_trace(go.Heatmap(z=g, colorscale='Reds', showscale=False,
                                 hovertemplate='%{z:,.0f} unserved<extra></extra>'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Unserved before Stage 2',
                                 marker=dict(size=10, color='#DC2626', symbol='square')))
        facility_layer(fig, plan[plan.stage == 0], 'Facilities')
        fc, fr = grid_xy(plan.x_m, plan.y_m)
        s1, up, stf = (plan.stage == 1).values, (plan.upgrade == 1).values, plan.extra_staff.values
        fig.add_trace(go.Scatter(x=fc[s1], y=fr[s1], mode='markers', name='Built in Stage 1',
                                 text=plan.name[s1], hovertemplate='%{text}<extra></extra>',
                                 marker=dict(size=15, symbol='star', color='#FACC15',
                                             line=dict(color='#0F172A', width=1))))
        fig.add_trace(go.Scatter(x=fc[stf > 0], y=fr[stf > 0], mode='markers', name='Extra clinicians',
                                 text=[f'{n}: +{k} clinician(s)' for n, k in zip(plan.name[stf > 0], stf[stf > 0])],
                                 hovertemplate='%{text}<extra></extra>',
                                 marker=dict(size=14 + 4 * stf[stf > 0], symbol='circle-open', color='#2563EB',
                                             line=dict(width=2.5))))
        fig.add_trace(go.Scatter(x=fc[up], y=fr[up], mode='markers', name='Upgraded to health centre',
                                 text=plan.name[up], hovertemplate='%{text}: upgrade<extra></extra>',
                                 marker=dict(size=20, symbol='square-open', color='#7C3AED', line=dict(width=2.5))))
        show(map_layout(fig))
    with right:
        panel(f'Actions chosen (solver status: {status})')
        showt = acts[['name', 'stage', 'capacity', 'served', 'extra_staff', 'upgrade']].copy()
        showt['stage'] = showt.stage.map({0: 'Existing', 1: 'Stage 1'})
        showt['upgrade'] = showt.upgrade.map({0: '', 1: 'Yes'})
        showt['served'] = showt.served.round(0)
        showt.columns = ['Facility', 'Built in', 'Base capacity', 'People served', 'Extra clinicians', 'Upgrade']
        table(showt.sort_values('People served', ascending=False))
    n_s1 = int((acts.stage == 1).sum())
    if n_s1:
        st.markdown(f'<div class="note"><b>The stages are linked.</b> {n_s1} dispensary built in Stage 1 needs extra '
                    'capacity in Stage 2. Year-1 building decisions create Year-2 needs, so the stages must be '
                    'planned together.</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="assume"><b>Model assumptions.</b> Capacity: dispensary 10,000 people, health centre 50,000, '
        'hospital 150,000. One extra clinician costs 1 unit and adds 5,000 capacity (at most 4 per facility). '
        'Upgrading a dispensary to a health centre costs 6 units and adds 40,000. People may use any of their five '
        f'nearest facilities within {limit_km:g} km. These are planning assumptions, not official figures.</p>',
        unsafe_allow_html=True)

st.divider()
st.caption('Data: Sentinel-2 (Copernicus) via Google Earth Engine · EuroSAT · Google Open Buildings · '
           'OpenStreetMap health facilities (HOT/HDX) · 2022 Tanzania Census household size. '
           'Population, capacities and costs are estimates and assumptions.')
