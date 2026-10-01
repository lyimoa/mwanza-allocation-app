# Mwanza multi-stage healthcare allocation - interactive demo
# DSAI 6220 Advanced Machine Learning, Group 1 (NM-AIST)
#
# Reads the tables produced by the Colab notebook and re-runs the two
# optimization stages live, so the budgets can be changed with sliders.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp
import streamlit as st
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle
from scipy.spatial import cKDTree

TILE_M = 640                         # one tile = 640 m x 640 m
HH_SIZE = 4.3                        # people per building (2022 Census)
CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
CLASS_COLOURS = ['#e9c46a', '#1b4332', '#95d5b2', '#6c757d', '#9d4edd',
                 '#b7e4c7', '#f4a261', '#d62828', '#48cae4', '#023e8a']
NAVY, TEAL = '#1F3A5F', '#1A7F6E'

st.set_page_config(page_title='Mwanza healthcare allocation', page_icon='🛰️', layout='wide')


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    tiles = pd.read_csv('data/mwanza_tiles_access.csv')
    fac = pd.read_csv('data/mwanza_health_facilities_osm.csv')
    fac['name'] = fac['name'].fillna('Unnamed')
    if 'x_m' not in fac.columns:                       # fallback: GPS -> UTM 36S
        from pyproj import Transformer
        t = Transformer.from_crs('EPSG:4326', 'EPSG:32736', always_xy=True)
        fac['x_m'], fac['y_m'] = t.transform(fac.lon.values, fac.lat.values)
    tiles['population'] = tiles['population'].fillna(0)
    return tiles, fac.reset_index(drop=True)


tiles, fac = load_data()
N_ROWS, N_COLS = int(tiles.row.max()) + 1, int(tiles.col.max()) + 1
# top-left corner of the grid in metres, recovered from the tile centres
X0 = float(np.median(tiles.x_m - (tiles.col + 0.5) * TILE_M))
Y0 = float(np.median(tiles.y_m + (tiles.row + 0.5) * TILE_M))


def to_grid(values, frame=tiles):
    g = np.full((N_ROWS, N_COLS), np.nan)
    g[frame.row.values, frame.col.values] = values
    return g


def grid_xy(x_m, y_m):
    """metres -> position on the tile grid (for drawing points on the maps)"""
    return (np.asarray(x_m) - X0) / TILE_M - 0.5, (Y0 - np.asarray(y_m)) / TILE_M - 0.5


def base_map(ax, alpha=0.55):
    ax.imshow(np.log10(to_grid(tiles.population.values) + 1), cmap='Greys', alpha=alpha)
    ax.axis('off')


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
def solve_stage2(K, limit_km, B, caps, cost_staff, cap_staff, max_staff, cost_upg, cap_upg, choices):
    chosen, _ = solve_stage1(K, limit_km)
    _, _, cand, _ = stage1_inputs(limit_km)
    new = cand.loc[chosen, ['x_m', 'y_m', 'lat', 'lon']].copy()
    new['name'] = [f'New Dispensary S1-{i + 1}' for i in range(len(new))]
    new['level'] = 'Dispensary/Clinic'
    new['stage'] = 1
    fac2 = pd.concat([fac.assign(stage=0)[['name', 'level', 'lat', 'lon', 'x_m', 'y_m', 'stage']],
                      new[['name', 'level', 'lat', 'lon', 'x_m', 'y_m', 'stage']]], ignore_index=True)
    cap_map = dict(zip(['Dispensary/Clinic', 'Health Centre', 'Hospital'], caps))
    fac2['capacity'] = fac2.level.map(cap_map).fillna(caps[0])

    pt = tiles[tiles.population > 0].reset_index(drop=True)
    k = min(choices, len(fac2))
    d, ix = cKDTree(fac2[['x_m', 'y_m']].values).query(pt[['x_m', 'y_m']].values, k=k)
    d, ix = np.atleast_2d(d.T).T, np.atleast_2d(ix.T).T
    options = []
    for dr, fr in zip(d, ix):
        ok = [int(f) for dd, f in zip(dr, fr) if dd <= limit_km * 1000]
        options.append(ok if ok else [int(fr[0])])

    prob = pulp.LpProblem('Stage2_Capacity', pulp.LpMinimize)
    I, F = range(len(pt)), range(len(fac2))
    send = {(i, f): pulp.LpVariable(f's_{i}_{f}', lowBound=0) for i in I for f in options[i]}
    unmet = pulp.LpVariable.dicts('unmet', I, lowBound=0)
    staff = pulp.LpVariable.dicts('staff', F, 0, max_staff, cat='Integer')
    upg = pulp.LpVariable.dicts('upg', F, cat='Binary')
    spend = pulp.lpSum(cost_staff * staff[f] + cost_upg * upg[f] for f in F)
    prob += pulp.lpSum(unmet.values()) + 0.001 * spend       # fewest unserved, then cheapest
    pop = pt.population.values
    for i in I:
        prob += pulp.lpSum(send[i, f] for f in options[i]) + unmet[i] == float(pop[i])
    users = {f: [] for f in F}
    for (i, f) in send:
        users[f].append(i)
    for f in F:
        prob += (pulp.lpSum(send[i, f] for i in users[f])
                 <= float(fac2.capacity[f]) + cap_staff * staff[f] + cap_upg * upg[f])
        if fac2.level[f] != 'Dispensary/Clinic':
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
st.sidebar.title('🛰️ Mwanza allocation')
st.sidebar.caption('DSAI 6220 · Group 1 · NM-AIST')
with st.sidebar.form('controls'):
    st.markdown('**Stage 1 · Access (Year 1)**')
    K = st.slider('New dispensaries to build (K)', 0, 10, 6)
    limit_km = st.slider('Access target (km to nearest facility)', 3.0, 10.0, 5.0, 0.5)
    st.markdown('**Stage 2 · Capacity (Year 2)**')
    B = st.slider('Budget (units; 1 unit = 1 clinician/year)', 0, 40, 16)
    with st.expander('Stage 2 assumptions'):
        cap_d = st.number_input('Dispensary capacity', 1000, 100000, 10000, 1000)
        cap_h = st.number_input('Health centre capacity', 1000, 500000, 50000, 5000)
        cap_hosp = st.number_input('Hospital capacity', 1000, 1000000, 150000, 10000)
        cap_staff = st.number_input('Capacity added per clinician', 500, 50000, 5000, 500)
        max_staff = st.number_input('Max extra clinicians per facility', 1, 10, 4)
        cost_upg = st.number_input('Upgrade cost (units)', 1, 20, 6)
        cap_upg = st.number_input('Capacity added per upgrade', 1000, 200000, 40000, 5000)
        choices = st.number_input('Facilities a person may use', 1, 10, 5)
    st.form_submit_button('Run optimization', type='primary', width='stretch')
st.sidebar.info('Change the sliders, then press **Run optimization**.')

caps = (int(cap_d), int(cap_h), int(cap_hosp))
s2_args = (caps, 1, int(cap_staff), int(max_staff), int(cost_upg), int(cap_upg), int(choices))

# ----------------------------------------------------------------------------
# Compute
# ----------------------------------------------------------------------------
dist, dem, cand, _ = stage1_inputs(limit_km)
total_pop = float(tiles.population.sum())
under_before = float(dem.population.sum()) if len(dem) else 0.0
with st.spinner('Stage 1: choosing the best sites...'):
    chosen, covered_people = solve_stage1(K, limit_km)
new_sites = cand.loc[chosen].reset_index(drop=True)
all_xy = np.vstack([fac[['x_m', 'y_m']].values, new_sites[['x_m', 'y_m']].values]) \
    if len(new_sites) else fac[['x_m', 'y_m']].values
dist_after = cKDTree(all_xy).query(tiles[['x_m', 'y_m']].values)[0] / 1000
under_mask_before = (tiles.population > 0) & (dist > limit_km)
under_mask_after = (tiles.population > 0) & (dist_after > limit_km)
under_after = float(tiles.population[under_mask_after].sum())

st.title('Satellite-guided multi-stage allocation of primary healthcare')
st.caption('55 km × 55 km study area around Mwanza city, Tanzania · '
           'CNN land-use classification of Sentinel-2 imagery + two-stage optimization')

tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ['Overview', 'Image analysis', 'Access gap', 'Stage 1 · Access', 'Stage 2 · Capacity'])

# ---------------------------- Overview --------------------------------------
with tab0:
    c = st.columns(4)
    c[0].metric('Tiles analysed (640 m)', f'{len(tiles):,}')
    c[1].metric('Estimated population', f'{total_pop:,.0f}')
    c[2].metric('Existing health facilities', f'{len(fac):,}')
    c[3].metric(f'Underserved (> {limit_km:g} km)', f'{under_before:,.0f}',
                f'{100 * under_before / total_pop:.1f}% of people', delta_color='off')
    st.markdown(
        """
**How the pipeline works**

1. **Image analysis.** A CNN (EfficientNetB0, fine-tuned on EuroSAT, 95.9% test accuracy) labels every
   640 m tile of a cloud-free Sentinel-2 image of Mwanza.
2. **Need assessment.** Building counts give population per tile; distance to the nearest facility shows who is underserved.
3. **Stage 1 · Access.** With a Year-1 budget of *K* dispensaries, an optimization model chooses the sites
   that bring the most underserved people within reach, only on land the CNN did not label as water.
4. **Stage 2 · Capacity.** With a Year-2 budget, a second model adds clinicians or upgrades dispensaries
   where demand exceeds capacity, **taking the Stage 1 sites as given**.

Use the sliders on the left to change the budgets and see both stages update.
        """)

# ---------------------------- Image analysis --------------------------------
with tab1:
    left, right = st.columns([3, 2])
    with left:
        codes = tiles.land_class.map({n: i for i, n in enumerate(CLASSES)}).values
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(to_grid(codes), cmap=ListedColormap(CLASS_COLOURS), vmin=-0.5, vmax=9.5,
                  interpolation='nearest')
        ax.legend(handles=[Patch(color=CLASS_COLOURS[i], label=n) for i, n in enumerate(CLASSES)],
                  loc='lower left', fontsize=7, framealpha=0.9)
        ax.set_title('CNN land-use classification (one square = one 640 m tile)')
        ax.axis('off')
        st.pyplot(fig)
    with right:
        summary = tiles.land_class.value_counts().rename_axis('Class').reset_index(name='Tiles')
        summary['Area (km²)'] = (summary.Tiles * 0.4096).round(1)
        summary['Share (%)'] = (100 * summary.Tiles / len(tiles)).round(1)
        st.dataframe(summary, hide_index=True, width='stretch')
        if 'confidence' in tiles.columns:
            st.metric('Mean prediction confidence', f'{tiles.confidence.mean():.3f}')
        if 'buildings' in tiles.columns:
            b = tiles.groupby('land_class').buildings.mean().round(1)
            st.markdown('**Validation against Google Open Buildings** (mean buildings per tile)')
            st.dataframe(b.sort_values(ascending=False).rename('Mean buildings').reset_index()
                         .rename(columns={'land_class': 'Class'}),
                         hide_index=True, width='stretch')

# ---------------------------- Access gap ------------------------------------
with tab2:
    a, b_ = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(np.log10(to_grid(tiles.population.values) + 1), cmap='magma')
        fc, fr = grid_xy(fac.x_m, fac.y_m)
        for lv, (m, s) in {'Hospital': ('*', 160), 'Health Centre': ('s', 45),
                           'Dispensary/Clinic': ('o', 18)}.items():
            sel = (fac.level == lv).values
            ax.scatter(fc[sel], fr[sel], marker=m, s=s, c='cyan', edgecolors='k', linewidths=0.5,
                       label=f'{lv} ({sel.sum()})')
        ax.legend(loc='lower left', fontsize=8)
        ax.set_title('Population (log scale) and existing facilities')
        ax.axis('off')
        st.pyplot(fig)
    with b_:
        dgrid = to_grid(dist)
        dgrid[to_grid(tiles.population.values) == 0] = np.nan
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(dgrid, cmap='RdYlGn_r', vmin=0, vmax=12)
        ax.contour(np.nan_to_num(dgrid, nan=0), levels=[limit_km], colors='k', linewidths=0.8)
        plt.colorbar(im, ax=ax, fraction=0.046, label='km to nearest facility')
        ax.set_title(f'Distance to nearest facility (black line = {limit_km:g} km)')
        ax.axis('off')
        st.pyplot(fig)
    st.markdown(f'**{under_before:,.0f} people** in **{int(under_mask_before.sum())} tiles** live more than '
                f'{limit_km:g} km from any facility. They are the demand for Stage 1.')

# ---------------------------- Stage 1 ---------------------------------------
with tab3:
    c = st.columns(4)
    c[0].metric('New dispensaries', K)
    c[1].metric('Underserved people covered', f'{covered_people:,.0f}',
                f'{100 * covered_people / under_before:.1f}% of underserved' if under_before else None,
                delta_color='off')
    c[2].metric('Still underserved', f'{under_after:,.0f}')
    c[3].metric(f'Access within {limit_km:g} km', f'{100 * (1 - under_after / total_pop):.1f}%',
                f'{100 * (under_before - under_after) / total_pop:+.1f} pts')
    left, right = st.columns([3, 2])
    with left:
        status = np.where(under_mask_after, 2, np.where(under_mask_before, 1, np.nan)).astype(float)
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        base_map(ax)
        ax.imshow(to_grid(status), cmap=ListedColormap(['orange', 'red']), vmin=0.5, vmax=2.5)
        fc, fr = grid_xy(fac.x_m, fac.y_m)
        ax.scatter(fc, fr, s=10, c='cyan', edgecolors='k', linewidths=0.3)
        if len(new_sites):
            nc, nr = grid_xy(new_sites.x_m, new_sites.y_m)
            ax.scatter(nc, nr, marker='*', s=320, c='yellow', edgecolors='k', zorder=5)
            for x, y in zip(nc, nr):
                ax.add_patch(Circle((x, y), limit_km * 1000 / TILE_M, fill=False, ls='--', color='k', lw=1))
        ax.legend(handles=[Patch(color='orange', label='Underserved, now covered'),
                           Patch(color='red', label='Still underserved'),
                           Line2D([], [], marker='o', ls='', color='cyan', markeredgecolor='k',
                                  label='Existing facilities'),
                           Line2D([], [], marker='*', ls='', color='yellow', markersize=14,
                                  markeredgecolor='k', label='New dispensaries')],
                  loc='lower left', fontsize=8)
        ax.set_title(f'Stage 1 plan: {K} new dispensaries with {limit_km:g} km coverage circles')
        st.pyplot(fig)
    with right:
        if len(new_sites):
            tbl = new_sites[['lat', 'lon', 'land_class']].copy()
            tbl.insert(0, 'Site', [f'S1-{i + 1}' for i in range(len(tbl))])
            tbl['Underserved within reach'] = [
                covered_pop(new_sites[['x_m', 'y_m']].values[i:i + 1], dem, limit_km)
                for i in range(len(new_sites))]
            tbl = tbl.rename(columns={'lat': 'Latitude', 'lon': 'Longitude', 'land_class': 'CNN land class'})
            st.markdown('**Chosen sites**')
            st.dataframe(tbl.round({'Latitude': 4, 'Longitude': 4, 'Underserved within reach': 0}),
                         hide_index=True, width='stretch')
        st.markdown('**Is optimization better than simple rules?** (same number of dispensaries)')
        if K > 0 and len(dem):
            top = dem.nlargest(K, 'population')[['x_m', 'y_m']].values
            rng = np.random.default_rng(42)
            rnd = np.mean([covered_pop(cand.sample(min(K, len(cand)), random_state=int(s))[['x_m', 'y_m']].values,
                                       dem, limit_km) for s in rng.integers(0, 10 ** 6, 100)])
            comp = pd.DataFrame({'Strategy': ['Optimized (ours)', 'Most-populated tiles first', 'Random sites'],
                                 'People covered': [covered_people, covered_pop(top, dem, limit_km), rnd]})
            fig, ax = plt.subplots(figsize=(5.5, 2.4))
            ax.barh(comp.Strategy, comp['People covered'], color=[TEAL, '#9bbfb8', '#cccccc'])
            for y, v in enumerate(comp['People covered']):
                ax.text(v, y, f' {v:,.0f}', va='center', fontsize=9)
            ax.invert_yaxis()
            ax.set_xlim(0, comp['People covered'].max() * 1.25)
            ax.spines[['top', 'right']].set_visible(False)
            st.pyplot(fig)
        if st.checkbox('Show the budget curve (K = 1 to 10)'):
            with st.spinner('Solving for each budget...'):
                curve = [solve_stage1(k, limit_km)[1] for k in range(1, 11)]
            fig, ax = plt.subplots(figsize=(5.5, 3))
            ax.plot(range(1, 11), curve, 'o-', color=NAVY)
            if K >= 1:
                ax.axvline(K, color='grey', ls='--')
            ax.set_xlabel('New dispensaries'); ax.set_ylabel('People covered'); ax.grid(alpha=0.3)
            st.pyplot(fig)
            extra = np.diff([0] + curve)
            st.dataframe(pd.DataFrame({'Dispensaries': range(1, 11), 'Total covered': np.round(curve),
                                       'Extra from this one': np.round(extra)}),
                         hide_index=True, width='stretch')

# ---------------------------- Stage 2 ---------------------------------------
with tab4:
    with st.spinner('Stage 2: allocating clinicians and upgrades (can take up to a minute)...'):
        plan0, pt0, _ = solve_stage2(K, limit_km, 0, *s2_args)
        plan, pt, status = solve_stage2(K, limit_km, B, *s2_args)
    gap0, gap = float(pt0.unmet.sum()), float(pt.unmet.sum())
    acts = plan[(plan.extra_staff > 0) | (plan.upgrade > 0)].copy()
    spent = int(acts.extra_staff.sum() + int(cost_upg) * acts.upgrade.sum())
    c = st.columns(4)
    c[0].metric('Capacity gap before Stage 2', f'{gap0:,.0f}')
    c[1].metric('Still unserved after Stage 2', f'{gap:,.0f}', f'{gap - gap0:,.0f}', delta_color='inverse')
    c[2].metric('Budget used', f'{spent} of {B} units')
    c[3].metric('Upgrades · clinicians', f'{int(acts.upgrade.sum())} · {int(acts.extra_staff.sum())}')
    left, right = st.columns([3, 2])
    with left:
        g = to_grid(pt0.unmet.values, pt0)
        g[g <= 0.5] = np.nan
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        base_map(ax, 0.5)
        ax.imshow(g, cmap='Reds', alpha=0.9)
        fc, fr = grid_xy(plan.x_m, plan.y_m)
        s1 = (plan.stage == 1).values
        up = (plan.upgrade == 1).values
        stf = plan.extra_staff.values
        ax.scatter(fc, fr, s=10, c='cyan', edgecolors='k', linewidths=0.3)
        ax.scatter(fc[s1], fr[s1], marker='*', s=240, c='yellow', edgecolors='k')
        ax.scatter(fc[stf > 0], fr[stf > 0], s=70 * stf[stf > 0], facecolors='none', edgecolors='blue', linewidths=2)
        ax.scatter(fc[up], fr[up], marker='s', s=230, facecolors='none', edgecolors='purple', linewidths=2.5)
        ax.legend(handles=[
            Patch(color='red', label='Unserved before Stage 2 (capacity gap)'),
            Line2D([], [], marker='*', ls='', color='yellow', markersize=13, markeredgecolor='k', label='Built in Stage 1'),
            Line2D([], [], marker='o', ls='', markerfacecolor='none', markeredgecolor='blue', markersize=11,
                   label='Extra clinicians (size = number)'),
            Line2D([], [], marker='s', ls='', markerfacecolor='none', markeredgecolor='purple', markersize=11,
                   label='Upgraded to health centre')], loc='lower left', fontsize=8)
        ax.set_title(f'Stage 2 plan with {B} units')
        st.pyplot(fig)
    with right:
        st.markdown(f'**Actions chosen** (solver status: {status})')
        if len(acts):
            show = acts[['name', 'stage', 'capacity', 'served', 'extra_staff', 'upgrade']].copy()
            show['stage'] = show.stage.map({0: 'Existing', 1: 'Stage 1'})
            show['upgrade'] = show.upgrade.map({0: '', 1: 'Yes'})
            show.columns = ['Facility', 'Built in', 'Base capacity', 'People served', 'Extra clinicians', 'Upgrade']
            st.dataframe(show.round(0).sort_values('People served', ascending=False),
                         hide_index=True, width='stretch')
            n_s1 = int(((acts.stage == 1)).sum())
            if n_s1:
                st.success(f'{n_s1} dispensary built in Stage 1 needs extra capacity in Stage 2: '
                           'Year-1 building decisions create Year-2 needs, so the stages must be planned together.')
        else:
            st.info('No actions: either the budget is 0 or there is no capacity gap.')
        st.caption('Capacities and costs are assumptions (see the sidebar). A facility is "full" when the people '
                   'depending on it exceed its assumed capacity; people may use any of their nearest facilities '
                   f'within {limit_km:g} km.')

st.divider()
st.caption('Data: Sentinel-2 (Copernicus) via Google Earth Engine · EuroSAT · Google Open Buildings · '
           'OpenStreetMap health facilities (HOT/HDX) · 2022 Tanzania Census household size. '
           'Population, capacities and costs are estimates and assumptions.')
