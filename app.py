# Mwanza multi-stage healthcare allocation - interactive dashboard
# DSAI 6220 Advanced Machine Learning, Group 1 (NM-AIST)
#
# Reads the tables produced by the Colab notebook and re-runs the two
# optimization stages live, so the budgets can be changed from the sidebar.
import base64
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
CLASS_COLOURS = ['#E9C46A', '#2D6A4F', '#95D5B2', '#6C757D', '#9D4EDD',
                 '#B7E4C7', '#F4A261', '#E76F6F', '#48CAE4', '#3B6FD4']
NAVY, INK, GREY, LINE = '#0B2545', '#1F2937', '#64748B', '#E2E8F0'
TEAL, MINT = '#1A7F6E', '#DDF3EC'
GREEN, BLUE, ORANGE, RED, GOLD = '#3FA86B', '#3B6FD4', '#E08A3C', '#E5534B', '#B8962E'
H = 440                              # every chart and table shares this height

# Stage 2 assumptions (fixed; listed in the app for transparency)
CAPS = {'Dispensary/Clinic': 10_000, 'Health Centre': 50_000, 'Hospital': 150_000}
COST_STAFF, CAP_STAFF, MAX_STAFF = 1, 5_000, 4
COST_UPG, CAP_UPG = 6, 40_000
CHOICES = 5

st.set_page_config(page_title='Mwanza healthcare allocation', page_icon='🛰️', layout='wide')

ICONS = {
    'grid': '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
    'users': '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    'plus': '<rect x="3" y="3" width="18" height="18" rx="3"/><path d="M12 8v8M8 12h8"/>',
    'check': '<circle cx="12" cy="12" r="9"/><path d="m8.5 12 2.5 2.5 4.5-5"/>',
    'alert': '<path d="M12 3 2 20h20L12 3z"/><path d="M12 10v4M12 17h.01"/>',
    'shield': '<path d="M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6l-8-3z"/><path d="m9 12 2 2 4-4"/>',
    'pin': '<path d="M12 21s7-6.2 7-11a7 7 0 1 0-14 0c0 4.800 7 11 7 11z"/><circle cx="12" cy="10" r="2.5"/>',
    'target': '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    'layers': '<path d="m12 3 9 5-9 5-9-5 9-5z"/><path d="m3 13 9 5 9-5"/>',
    'wallet': '<rect x="3" y="6" width="18" height="13" rx="2"/><path d="M3 10h18M16 15h2"/>',
    'chart': '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    'map': '<path d="m3 6 6-2 6 2 6-2v14l-6 2-6-2-6 2V6z"/><path d="M9 4v14M15 6v14"/>',
    'search': '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    'gauge': '<path d="M4 18a9 9 0 1 1 16 0"/><path d="m12 14 4-5"/>',
}


def svg(name, color, size=18):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'{ICONS[name]}</svg>')


st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
  html, body, .stApp, [class*="css"], .stMarkdown, p, label, span, div {{ font-family: 'Plus Jakarta Sans', -apple-system, 'Segoe UI', sans-serif; }}
  .stApp {{ background: #FFFFFF; }}
  [data-testid="stHeader"] {{ background: transparent; }}
  [data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
  [data-testid="stDecoration"], [data-testid="stStatusWidget"], #MainMenu, footer,
  .stDeployButton, [class*="viewerBadge"] {{ display: none !important; }}
  .block-container {{ padding: 1.2rem 2.4rem 2.5rem 2.4rem; max-width: 1500px; }}
  [data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {LINE}; }}
  [data-testid="stSidebarContent"] {{ padding-top: 0.6rem; }}

  /* hero banner */
  .hero {{ display: flex; align-items: center; gap: 22px; padding: 16px 24px; border-radius: 16px; color: #FFFFFF;
          background: linear-gradient(110deg, {NAVY} 0%, #14506A 55%, {TEAL} 100%);
          box-shadow: 0 10px 28px rgba(11,37,69,0.18); margin-bottom: 16px; position: relative; overflow: hidden; }}
  .hero:after {{ content: ''; position: absolute; right: -70px; top: -90px; width: 300px; height: 300px; border-radius: 50%;
                background: rgba(255,255,255,0.07); }}
  .hero .logo {{ background: #FFFFFF; border-radius: 14px; padding: 8px; display: flex; flex: none; }}
  .hero .logo img {{ height: 54px; }}
  .hero .txt {{ flex: 1; z-index: 1; }}
  .hero .eyebrow {{ font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase; color: {MINT}; font-weight: 600; }}
  .hero .title {{ font-size: 23px; font-weight: 800; letter-spacing: -0.02em; margin: 4px 0 4px 0; line-height: 1.15; color: #FFFFFF; }}
  .hero .subtitle {{ font-size: 13px; color: #D6E6EE; }}
  .hero .badge {{ z-index: 1; text-align: right; background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.25);
                 border-radius: 12px; padding: 8px 14px; font-size: 12px; line-height: 1.5; color: #FFFFFF; }}
  .hero .badge b {{ font-size: 13.5px; }}

  /* page tabs */
  .st-key-nav div[role="radiogroup"] {{ gap: 8px; background: #F1F6F5; padding: 6px; border-radius: 14px; width: fit-content; }}
  .st-key-nav div[role="radiogroup"] label {{ padding: 7px 16px; border-radius: 10px; cursor: pointer; margin: 0; }}
  .st-key-nav label[data-testid="stRadioOption"] > div > div:first-child {{ display: none !important; }}
  .st-key-nav label[data-testid="stRadioOption"] > div {{ gap: 0; }}
  .st-key-nav div[role="radiogroup"] label p {{ font-size: 13px; color: #475569; font-weight: 600; }}
  .st-key-nav div[role="radiogroup"] label:hover {{ background: #E2EFEC; }}
  .st-key-nav div[role="radiogroup"] label:has(input:checked), .st-key-nav label[data-selected="true"] {{ background: {NAVY}; box-shadow: 0 2px 8px rgba(11,37,69,0.25); }}
  .st-key-nav div[role="radiogroup"] label:has(input:checked) p, .st-key-nav label[data-selected="true"] p {{ color: #FFFFFF; }}

  /* sidebar */
  .side-title {{ font-size: 16px; font-weight: 800; color: {NAVY}; margin: 6px 0 2px 2px; }}
  .side-label {{ font-size: 12.5px; color: {GREY}; margin: 0 0 12px 2px; }}
  .side-foot {{ color: {GREY}; font-size: 12.5px; line-height: 1.6; margin-top: 12px; }}
  .live {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: {TEAL}; margin-right: 6px; }}

  /* section heading */
  .sec {{ display: flex; align-items: baseline; justify-content: space-between; margin: 14px 0 12px 0; }}
  .sec .l {{ display: flex; align-items: center; gap: 10px; font-size: 16.5px; font-weight: 800; color: {NAVY}; letter-spacing: -0.01em; }}
  .sec .l svg {{ display: none; }}
  .sec .l:before {{ content: ''; width: 5px; height: 19px; border-radius: 3px; background: {TEAL}; }}
  .sec .r {{ font-size: 12px; color: {TEAL}; font-weight: 600; background: {MINT}; padding: 4px 12px; border-radius: 999px; }}
  .sec .r:empty {{ display: none; }}

  /* KPI cards */
  .kpi {{ background: linear-gradient(180deg, #FFFFFF 0%, #F7FBFA 100%); border: 1px solid {LINE}; border-radius: 16px;
         padding: 14px 16px; height: 118px; overflow: hidden; box-shadow: 0 4px 14px rgba(11,37,69,0.06);
         margin-bottom: 14px; position: relative; }}
  .kpi:before {{ content: ''; position: absolute; left: 0; right: 0; top: 0; height: 4px; background: var(--tone); }}
  .kpi .h {{ display: flex; align-items: center; justify-content: space-between; flex-direction: row-reverse; gap: 10px;
            color: #475569; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; }}
  .kpi .chip {{ width: 30px; height: 30px; border-radius: 50%; background: var(--tint); display: flex;
               align-items: center; justify-content: center; flex: none; }}
  .kpi .v {{ font-size: 25px; font-weight: 800; color: {NAVY}; margin-top: 4px; letter-spacing: -0.03em; line-height: 1.1; }}
  .kpi .s {{ color: {GREY}; font-size: 12px; margin-top: 5px; }}
  .kpi .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--tone); margin-right: 7px; }}

  /* chart cards */
  [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 16px !important; border-color: {LINE} !important;
        background: #FFFFFF; box-shadow: 0 4px 14px rgba(11,37,69,0.06); }}
  .card-h {{ display: flex; align-items: center; justify-content: space-between; padding-bottom: 8px; margin-bottom: 4px; }}
  .card-h .l {{ display: flex; align-items: center; gap: 9px; font-weight: 700; color: {NAVY}; font-size: 14px; }}
  .card-h .r {{ color: {GREY}; font-size: 11.5px; }}
  .tblwrap {{ height: {H}px; overflow: auto; border: 1px solid {LINE}; border-radius: 12px; }}
  table.tbl {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  table.tbl th {{ position: sticky; top: 0; background: {NAVY}; color: #FFFFFF; text-align: left; font-weight: 600;
                 font-size: 11.5px; letter-spacing: 0.05em; text-transform: uppercase; padding: 11px 14px; }}
  table.tbl td {{ padding: 11px 14px; border-bottom: 1px solid #EEF2F4; color: {INK}; }}
  table.tbl tr:nth-child(even) td {{ background: #F7FBFA; }}
  table.tbl tr:hover td {{ background: {MINT}; }}
  table.tbl .num {{ text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }}
  .note {{ background: {MINT}; border-left: 5px solid {TEAL}; border-radius: 12px; padding: 13px 16px; color: {INK}; font-size: 14px; }}
  .assume {{ color: {GREY}; font-size: 13px; line-height: 1.6; }}
  .assume b {{ color: {INK}; }}
  div[data-testid="stForm"] {{ border: 1px solid {LINE}; border-radius: 14px; background: #F7FBFA; padding: 12px 12px 6px 12px; }}
</style>
""", unsafe_allow_html=True)

TONES = {'green': (GREEN, '#E9F6EE'), 'blue': (BLUE, '#E8EEFB'), 'orange': (ORANGE, '#FCEFE2'),
         'red': (RED, '#FDECEA'), 'gold': (GOLD, '#F7F1DF')}


def kpi(col, label, value, sub, tone='blue', icon='chart', dot=False):
    c, tint = TONES[tone]
    d = '<span class="dot"></span>' if dot else ''
    col.markdown(f'<div class="kpi" style="--tone:{c};--tint:{tint}"><div class="h"><div class="chip">'
                 f'{svg(icon, c)}</div>{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{d}{sub}</div></div>', unsafe_allow_html=True)


def section(title, right='', icon='gauge'):
    st.markdown(f'<div class="sec"><div class="l">{svg(icon, BLUE, 20)}{title}</div>'
                f'<div class="r">{right}</div></div>', unsafe_allow_html=True)


def card(title, hint='', icon='chart'):
    box = st.container(border=True, height=H + 84)
    box.markdown(f'<div class="card-h"><div class="l">{svg(icon, TEAL, 17)}{title}</div>'
                 f'<div class="r">{hint}</div></div>', unsafe_allow_html=True)
    return box


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
POP_LOG = np.where(POP_GRID > 0, np.log10(np.where(POP_GRID > 0, POP_GRID, 1)), np.nan)
CODES = to_grid(tiles.land_class.map({n: i for i, n in enumerate(CLASSES)}).values)
CLASS_SCALE = [[i / 10 if j == 0 else (i + 1) / 10, CLASS_COLOURS[i]] for i in range(10) for j in (0, 1)]
CLASS_NAMES = np.vectorize(lambda v: CLASSES[int(v)] if v == v else '')(CODES)
POP_SCALE = [[0, '#EAF7F3'], [0.35, '#8FD9C4'], [0.7, '#1A7F6E'], [1, '#0B2545']]


# ----------------------------------------------------------------------------
# Chart helpers (every figure has the same fixed height)
# ----------------------------------------------------------------------------
def map_layout(fig, legend=True):
    fig.update_layout(
        height=H, margin=dict(l=4, r=4, t=4, b=4), paper_bgcolor='white', plot_bgcolor='#F8FAFC',
        showlegend=legend, font=dict(family="'Plus Jakarta Sans', 'Segoe UI', Arial, sans-serif", color=INK, size=12),
        legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.01, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12, color=INK), itemsizing='constant'),
        xaxis=dict(visible=False, range=[-0.5, N_COLS - 0.5], constrain='domain'),
        yaxis=dict(visible=False, range=[N_ROWS - 0.5, -0.5], scaleanchor='x', constrain='domain'))
    return fig


def swatch(fig, name, color, symbol='square', size=11):
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=name,
                             marker=dict(size=size, color=color, symbol=symbol)))


def base_layer(fig):
    fig.add_trace(go.Heatmap(z=POP_LOG, colorscale=[[0, '#EEF1F5'], [1, '#B4BDCB']], showscale=False,
                             hoverinfo='skip'))


def facility_layer(fig, frame, name='Existing facilities'):
    fc, fr = grid_xy(frame.x_m, frame.y_m)
    fig.add_trace(go.Scatter(x=fc, y=fr, mode='markers', name=name, text=frame.name,
                             hovertemplate='%{text}<extra></extra>',
                             marker=dict(size=6, color='#0B2545', line=dict(color='#FFFFFF', width=0.8))))


FONT = "'Plus Jakarta Sans', 'Segoe UI', Arial, sans-serif"
HOVER = dict(bgcolor=NAVY, bordercolor=NAVY, font=dict(color='white', size=12, family=FONT))


def chart_layout(fig):
    fig.update_layout(height=H, margin=dict(l=24, r=28, t=18, b=46), paper_bgcolor='white',
                      plot_bgcolor='white', showlegend=False, hoverlabel=HOVER,
                      font=dict(family=FONT, color=INK, size=12))
    fig.update_xaxes(gridcolor='#EEF2F4', griddash='dot', zeroline=False, showline=False, ticks='',
                     tickfont=dict(color=GREY, size=11), title_font=dict(color=GREY, size=11.5))
    fig.update_yaxes(gridcolor='#EEF2F4', griddash='dot', zeroline=False, showline=False, ticks='', automargin=True,
                     tickfont=dict(color=INK, size=12), title_font=dict(color=GREY, size=11.5))
    return fig


def hbar(labels, values, xtitle, colors=None):
    """Slim rounded bars on a pale track; the name sits above each bar and the value at its end."""
    values = [float(v) for v in values]
    labels = [str(l) for l in labels]
    vmax = max(values + [1])
    slot = (H - 70) / max(len(values), 1)            # pixels available per bar
    bar_px = min(16, slot * 0.3)
    w = bar_px / slot
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[vmax] * len(values), y=labels, orientation='h', hoverinfo='skip', width=w,
                         marker=dict(color='#EEF3F4', cornerradius=30, line=dict(width=0))))
    fig.add_trace(go.Bar(
        x=values, y=labels, orientation='h', hovertemplate='%{y}: %{x:,.0f}<extra></extra>', width=w,
        marker=dict(color=colors if colors else values, cornerradius=30, line=dict(width=0),
                    colorscale=None if colors else [[0, '#7FD3BD'], [1, '#1A7F6E']])))
    for lab, v in zip(labels, values):
        fig.add_annotation(x=0, y=lab, text=lab, showarrow=False, xanchor='left', yanchor='bottom',
                           yshift=bar_px / 2 + 3, font=dict(size=12.5, color=INK, family=FONT))
        fig.add_annotation(x=vmax, y=lab, text=f'<b>{v:,.0f}</b>', showarrow=False, xanchor='right',
                           yanchor='bottom', yshift=bar_px / 2 + 3, font=dict(size=13, color=NAVY, family=FONT))
    fig.update_yaxes(autorange='reversed', showgrid=False, showticklabels=False)
    fig.update_xaxes(title=xtitle, range=[0, vmax * 1.005], showgrid=False, showticklabels=False)
    fig.update_layout(barmode='overlay')
    chart_layout(fig)
    fig.update_layout(margin=dict(l=14, r=14, t=24, b=40))
    return fig


def donut(labels, values, colors, centre, caption='total'):
    keep = [i for i, v in enumerate(values) if v > 0]
    labels, values, colors = [labels[i] for i in keep], [values[i] for i in keep], [colors[i] for i in keep]
    total = sum(values) or 1
    names = [f'{l}  <b>{100 * v / total:.0f}%</b>' for l, v in zip(labels, values)]
    fig = go.Figure(go.Pie(labels=names, values=values, hole=0.72, sort=False, direction='clockwise',
                           domain=dict(x=[0.02, 0.56], y=[0.06, 0.94]),
                           marker=dict(colors=colors, line=dict(color='white', width=3)),
                           textinfo='none', hovertemplate='%{label}<br>%{value:,.0f}<extra></extra>'))
    fig.update_layout(height=H, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='white', showlegend=True,
                      hoverlabel=HOVER, font=dict(family=FONT, color=INK, size=12),
                      legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=0.62,
                                  font=dict(size=12.5, color=INK), itemsizing='constant', tracegroupgap=6),
                      annotations=[dict(text=f'<b>{centre}</b>', x=0.29, y=0.53, showarrow=False,
                                        font=dict(size=24, color=NAVY, family=FONT)),
                                   dict(text=caption, x=0.29, y=0.44, showarrow=False,
                                        font=dict(size=11.5, color=GREY, family=FONT))])
    return fig


def show(box, fig):
    box.plotly_chart(fig, width='stretch', theme=None, config={'displayModeBar': False})


def table(box, frame):
    """A clean HTML table: navy header, striped rows, numbers right-aligned with thousands separators."""
    head = ''.join(f'<th class="{"num" if pd.api.types.is_numeric_dtype(frame[c]) else ""}">{c}</th>'
                   for c in frame.columns)
    rows = []
    for _, r in frame.iterrows():
        cells = []
        for c in frame.columns:
            v = r[c]
            if pd.api.types.is_numeric_dtype(frame[c]):
                if pd.isna(v):
                    txt = ''
                elif float(v) == int(v) or abs(v) >= 1000:
                    txt = f'{v:,.0f}'
                elif c.lower().startswith(('lat', 'lon')):
                    txt = f'{v:.4f}'
                else:
                    txt = f'{v:,.1f}'
                cells.append(f'<td class="num">{txt}</td>')
            else:
                cells.append(f'<td>{"" if pd.isna(v) else v}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    box.markdown(f'<div class="tblwrap"><table class="tbl"><thead><tr>{head}</tr></thead>'
                 f'<tbody>{"".join(rows)}</tbody></table></div>', unsafe_allow_html=True)


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
# Sidebar: navigation + decision controls
# ----------------------------------------------------------------------------
PAGES = ['Overview', 'Image Analysis', 'Access Gap', 'Stage 1 · Access', 'Stage 2 · Capacity']
PRESETS = {'Recommended plan': (6, 16), 'Tight budget': (3, 8), 'Generous budget': (10, 30),
           'No intervention': (0, 0)}
ACTIONS = {'Clinicians and upgrades': (True, True), 'Clinicians only': (True, False),
           'Upgrades only': (False, True)}

with st.sidebar:
    st.markdown('<div class="side-title">Decision controls</div>'
                '<div class="side-label">Pick a scenario, or choose Custom to set your own budgets.</div>',
                unsafe_allow_html=True)
    preset = st.selectbox('Scenario', list(PRESETS) + ['Custom'], index=0, label_visibility='collapsed')
    k_def, b_def = PRESETS.get(preset, (6, 16))
    custom = preset == 'Custom'
    with st.form('controls'):
        K = st.slider('New dispensaries (Year 1)', 0, 10, k_def, disabled=not custom)
        B = st.slider('Budget units (Year 2)', 0, 40, b_def, disabled=not custom)
        limit_km = st.slider('Access target (km)', 3.0, 10.0, 5.0, 0.5)
        action = st.selectbox('Year-2 budget may buy', list(ACTIONS), index=0)
        st.form_submit_button('Run optimization', type='primary', width='stretch')
    if not custom:
        K, B = k_def, b_def
    st.markdown('<div class="side-foot"><span class="live"></span>Live · re-optimized on every run<br>'
                'Source: Sentinel-2, Open Buildings, OpenStreetMap</div>', unsafe_allow_html=True)
allow_staff, allow_upgrade = ACTIONS[action]

# ----------------------------------------------------------------------------
# Top bar
# ----------------------------------------------------------------------------
logo = next((p for p in ('logo.png', 'logo.jpg', 'logo.jpeg', 'data/logo.png', 'data/logo.jpg')
             if os.path.exists(p)), None)
logo_html = ''
if logo:
    mime = 'png' if logo.endswith('png') else 'jpeg'
    logo_html = f'<img src="data:image/{mime};base64,{base64.b64encode(open(logo, "rb").read()).decode()}"/>'
logo_block = f'<div class="logo">{logo_html}</div>' if logo_html else ''
st.markdown(
    f'<div class="hero">{logo_block}<div class="txt">'
    '<div class="eyebrow">The Nelson Mandela African Institution of Science and Technology</div>'
    '<div class="title">Mwanza Healthcare Allocation Planner</div>'
    '<div class="subtitle">Satellite imagery, deep learning and two-stage optimization to decide where to build '
    'and where to strengthen health facilities</div></div>'
    f'<div class="badge">DSAI 6220 · Group 1<br><b>{preset}</b><br>{K} dispensaries · {B} budget units</div></div>',
    unsafe_allow_html=True)
page = st.radio('Page', PAGES, key='nav', horizontal=True, label_visibility='collapsed')

# ----------------------------------------------------------------------------
# Shared results
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
acc_before = 100 * (1 - under_before / total_pop)
acc_after = 100 * (1 - under_after / total_pop)
scenario = f'{preset} · {K} dispensaries · {B} units'


def population_map():
    fig = go.Figure(go.Heatmap(z=POP_LOG, customdata=POP_GRID, colorscale=POP_SCALE, showscale=False,
                               hovertemplate='%{customdata:,.0f} people<extra></extra>'))
    return fig


def landuse_map():
    fig = go.Figure(go.Heatmap(z=CODES, zmin=-0.5, zmax=9.5, colorscale=CLASS_SCALE, showscale=False,
                               customdata=CLASS_NAMES, hovertemplate='%{customdata}<extra></extra>'))
    for n, col in zip(CLASSES, CLASS_COLOURS):
        swatch(fig, n, col)
    return map_layout(fig)


# ---------------------------- Overview --------------------------------------
if page == 'Overview':
    section('Key indicators', scenario)
    c = st.columns(4)
    kpi(c[0], 'Study area', f'{len(tiles):,}', 'tiles of 640 m × 640 m', 'blue', 'grid')
    kpi(c[1], 'Estimated population', f'{total_pop / 1e6:.2f}M', (f'from {tiles.buildings.sum():,.0f} buildings × 4.3' if 'buildings' in tiles else 'estimated from buildings'), 'blue', 'users')
    kpi(c[2], 'Health facilities', f'{len(fac):,}', 'existing, from OpenStreetMap', 'green', 'plus', True)
    kpi(c[3], 'CNN test accuracy', '95.9%', 'EfficientNetB0 on EuroSAT', 'green', 'check', True)
    c = st.columns(4)
    kpi(c[0], 'Underserved people', f'{under_before:,.0f}', f'live beyond {limit_km:g} km of care', 'orange', 'alert')
    kpi(c[1], 'Covered by Stage 1', f'{covered_people:,.0f}',
        f'{100 * covered_people / under_before:.1f}% of the underserved' if under_before else 'no gap', 'green',
        'shield', True)
    kpi(c[2], 'Still underserved', f'{under_after:,.0f}', 'after the new dispensaries', 'red', 'pin')
    kpi(c[3], f'Access within {limit_km:g} km', f'{acc_after:.1f}%', f'up from {acc_before:.1f}%', 'green',
        'target', True)
    section('Maps of the study area', '55 km × 55 km around Mwanza city', 'search')
    a, b_ = st.columns(2)
    with a:
        box = card('Where people live', 'Darker = more people', 'users')
        show(box, map_layout(population_map(), legend=False))
    with b_:
        box = card('What the CNN sees', 'Land-use class per tile', 'layers')
        show(box, landuse_map())

# ---------------------------- Image analysis --------------------------------
elif page == 'Image Analysis':
    section('Key indicators', 'CNN trained on EuroSAT, applied to Mwanza')
    c = st.columns(4)
    kpi(c[0], 'Test accuracy', '95.9%', '4,050 unseen EuroSAT tiles', 'green', 'check', True)
    kpi(c[1], 'Mean confidence', f'{tiles.confidence.mean():.3f}' if 'confidence' in tiles else 'n/a',
        'on Mwanza tiles', 'blue', 'gauge')
    kpi(c[2], 'AUC in Mwanza', '0.913', 'against Google Open Buildings', 'green', 'shield', True)
    kpi(c[3], 'Residential tiles', f'{int((tiles.land_class == "Residential").sum()):,}',
        f'{100 * (tiles.land_class == "Residential").mean():.1f}% of the study area', 'orange', 'pin')
    section('Land use and validation', 'one square = one 640 m tile', 'search')
    a, b_ = st.columns(2)
    with a:
        show(card('CNN land-use classification', 'What is on the ground?', 'map'), landuse_map())
    with b_:
        counts = tiles.land_class.value_counts().reindex(CLASSES).fillna(0)
        show(card('Share of tiles by class', 'How is the land used?', 'target'),
             donut(CLASSES, counts.values, CLASS_COLOURS, f'{len(tiles):,}', 'tiles'))
    a, b_ = st.columns(2)
    with a:
        if 'buildings' in tiles.columns:
            m = tiles.groupby('land_class').buildings.mean().sort_values(ascending=False)
            show(card('Mean buildings per tile, by CNN class', 'Do the labels match reality?', 'chart'),
                 hbar(list(m.index), list(m.values.round(1)), 'Buildings per tile (Google Open Buildings)'))
    with b_:
        s = tiles.groupby('land_class').agg(Tiles=('row', 'size')).reset_index()
        s['Area (km²)'] = (s.Tiles * 0.4096).round(1)
        s['Share (%)'] = (100 * s.Tiles / len(tiles)).round(1)
        if 'buildings' in tiles.columns:
            s['Mean buildings'] = tiles.groupby('land_class').buildings.mean().round(1).values
        table(card('Class summary', 'Details', 'grid'),
              s.rename(columns={'land_class': 'CNN class'}).sort_values('Tiles', ascending=False))

# ---------------------------- Access gap ------------------------------------
elif page == 'Access Gap':
    section('Key indicators', f'Access target: {limit_km:g} km to the nearest facility')
    c = st.columns(4)
    kpi(c[0], f'Within {limit_km:g} km of care', f'{acc_before:.1f}%', 'of the estimated population', 'green',
        'check', True)
    kpi(c[1], 'Underserved people', f'{under_before:,.0f}', f'{100 - acc_before:.1f}% of the population',
        'orange', 'alert')
    kpi(c[2], 'Underserved tiles', f'{int(mask_before.sum()):,}', 'populated tiles beyond the target', 'red', 'pin')
    kpi(c[3], 'Health facilities', f'{len(fac):,}',
        'dispensaries, health centres, hospitals', 'blue', 'plus')
    section('Where is the gap?', 'hover over the maps for details', 'search')
    a, b_ = st.columns(2)
    with a:
        fig = population_map()
        for lv, sym, size, col in [('Dispensary/Clinic', 'circle', 6, '#14213D'),
                                   ('Health Centre', 'square', 9, ORANGE), ('Hospital', 'star', 13, RED)]:
            sub = fac[fac.level == lv]
            fc, fr = grid_xy(sub.x_m, sub.y_m)
            fig.add_trace(go.Scatter(x=fc, y=fr, mode='markers', name=f'{lv} ({len(sub)})', text=sub.name,
                                     hovertemplate='%{text}<extra></extra>',
                                     marker=dict(size=size, symbol=sym, color=col,
                                                 line=dict(color='#FFFFFF', width=0.8))))
        show(card('Population and existing facilities', 'Where is care today?', 'map'), map_layout(fig))
    with b_:
        dgrid = to_grid(dist)
        fig = go.Figure(go.Heatmap(z=np.where(POP_GRID > 0, dgrid, np.nan), zmin=0, zmax=12, showscale=False,
                                   colorscale=[[0, '#3FA86B'], [0.42, '#F2E8A0'], [0.7, '#E08A3C'], [1, '#C0392B']],
                                   hovertemplate='%{z:.1f} km<extra></extra>'))
        fig.add_trace(go.Contour(z=dgrid, contours=dict(start=limit_km, end=limit_km, size=1, coloring='none'),
                                 line=dict(color=NAVY, width=1.3), showscale=False, hoverinfo='skip',
                                 showlegend=False))
        swatch(fig, 'Near a facility', '#3FA86B')
        swatch(fig, 'Around the limit', '#F2E8A0')
        swatch(fig, 'Far from care', '#C0392B')
        show(card('Distance to the nearest facility', f'Dark line = {limit_km:g} km', 'pin'), map_layout(fig))
    a, b_ = st.columns(2)
    with a:
        bins = [0, 1, 2, 3, 4, 5, 6, 8, 10, 100]
        labels = ['0–1', '1–2', '2–3', '3–4', '4–5', '5–6', '6–8', '8–10', '10+']
        grp = pd.cut(dist, bins=bins, labels=labels, right=False)
        by = tiles.population.groupby(grp, observed=False).sum()
        cols = [GREEN if float(l.split('–')[0].replace('+', '')) < limit_km else ORANGE for l in labels]
        fig = go.Figure(go.Bar(x=labels, y=by.values, marker=dict(color=cols, cornerradius=5),
                               hovertemplate='%{x} km: %{y:,.0f} people<extra></extra>'))
        fig.update_xaxes(title='Distance to nearest facility (km)', showgrid=False)
        fig.update_yaxes(title='People')
        show(card('People by distance to care', 'How far do people live?', 'chart'), chart_layout(fig))
    with b_:
        lv = fac.level.value_counts()
        show(card('Facilities by level', 'What kind of care exists?', 'target'),
             donut(list(lv.index), list(lv.values), [TEAL, NAVY, ORANGE][:len(lv)], f'{len(fac)}', 'facilities'))

# ---------------------------- Stage 1 ---------------------------------------
elif page == 'Stage 1 · Access':
    section('Key indicators', scenario)
    c = st.columns(4)
    kpi(c[0], 'New dispensaries', f'{K}', 'Year-1 budget', 'blue', 'plus')
    kpi(c[1], 'People covered', f'{covered_people:,.0f}',
        f'{100 * covered_people / under_before:.1f}% of the underserved' if under_before else 'no gap', 'green',
        'shield', True)
    kpi(c[2], 'Still underserved', f'{under_after:,.0f}', f'beyond {limit_km:g} km after Stage 1', 'red', 'alert')
    kpi(c[3], f'Access within {limit_km:g} km', f'{acc_after:.1f}%', f'up from {acc_before:.1f}%', 'green',
        'target', True)
    section('The plan', 'chosen by the maximal covering location model', 'search')
    a, b_ = st.columns(2)
    with a:
        status = np.where(mask_after, 1.0, np.where(mask_before, 0.0, np.nan))
        fig = go.Figure()
        base_layer(fig)
        fig.add_trace(go.Heatmap(z=to_grid(status), zmin=0, zmax=1, showscale=False, hoverinfo='skip',
                                 colorscale=[[0, ORANGE], [0.5, ORANGE], [0.5, RED], [1, RED]]))
        swatch(fig, 'Now covered', ORANGE)
        swatch(fig, 'Still underserved', RED)
        facility_layer(fig, fac)
        if len(new_sites):
            nc, nr = grid_xy(new_sites.x_m, new_sites.y_m)
            r = limit_km * 1000 / TILE_M
            for x, y in zip(nc, nr):
                fig.add_shape(type='circle', x0=x - r, x1=x + r, y0=y - r, y1=y + r,
                              line=dict(color=BLUE, width=1.4, dash='dash'))
            fig.add_trace(go.Scatter(x=nc, y=nr, mode='markers', name='New dispensaries',
                                     text=[f'S1-{i + 1}' for i in range(len(nc))],
                                     hovertemplate='%{text}<extra></extra>',
                                     marker=dict(size=17, symbol='star', color='#F5C518',
                                                 line=dict(color=NAVY, width=1))))
        show(card('Where to build', f'Circles = {limit_km:g} km reach', 'map'), map_layout(fig))
    with b_:
        tbl = pd.DataFrame({
            'Site': [f'S1-{i + 1}' for i in range(len(new_sites))],
            'Latitude': new_sites.lat.round(4) if len(new_sites) else [],
            'Longitude': new_sites.lon.round(4) if len(new_sites) else [],
            'CNN land class': new_sites.land_class if len(new_sites) else [],
            'People within reach': [round(covered_pop(new_sites[['x_m', 'y_m']].values[i:i + 1], dem, limit_km))
                                    for i in range(len(new_sites))]})
        table(card('Sites chosen', 'On CNN-verified land', 'grid'),
              tbl.sort_values('People within reach', ascending=False))
    section('Is the plan any good?', 'same number of dispensaries for every strategy', 'chart')
    a, b_ = st.columns(2)
    with a:
        if K > 0 and len(dem):
            top = dem.nlargest(K, 'population')[['x_m', 'y_m']].values
            rng = np.random.default_rng(42)
            rnd = np.mean([covered_pop(cand.sample(min(K, len(cand)), random_state=int(sd))[['x_m', 'y_m']].values,
                                       dem, limit_km) for sd in rng.integers(0, 10 ** 6, 100)])
            vals = [covered_people, covered_pop(top, dem, limit_km), rnd]
        else:
            vals = [0, 0, 0]
        show(card('Optimization vs simple rules', 'Who covers more people?', 'chart'),
             hbar(['Optimized (ours)', 'Most-populated tiles first', 'Random sites'], vals,
                  'Underserved people covered', [GREEN, '#9DB7E8', '#CBD5E1']))
    with b_:
        with st.spinner('Solving for each budget...'):
            curve = [solve_stage1(k, limit_km)[1] for k in range(1, 11)]
        extra = np.diff([0] + curve)
        fig = go.Figure(go.Scatter(x=list(range(1, 11)), y=curve, mode='lines+markers', customdata=extra,
                                   line=dict(color=TEAL, width=3.5, shape='spline'), fill='tozeroy',
                                   fillgradient=dict(type='vertical', colorscale=[[0, 'rgba(26,127,110,0.0)'],
                                                                                  [1, 'rgba(26,127,110,0.30)']]),
                                   marker=dict(size=9, color='white', line=dict(color=TEAL, width=2.5)),
                                   hovertemplate='%{x} dispensaries<br>%{y:,.0f} covered'
                                                 '<br>+%{customdata:,.0f} from the last one<extra></extra>'))
        if K >= 1:
            fig.add_vline(x=K, line_dash='dot', line_color=NAVY, line_width=1.5)
            fig.add_trace(go.Scatter(x=[K], y=[curve[K - 1]], mode='markers+text', hoverinfo='skip',
                                     text=[f'<b>{curve[K - 1]:,.0f}</b>'], textposition='top left',
                                     textfont=dict(color=NAVY, size=12.5),
                                     marker=dict(size=14, color=NAVY, line=dict(color='white', width=3))))
        fig.update_xaxes(title='New dispensaries', dtick=1, showgrid=False)
        fig.update_yaxes(title='People covered', rangemode='tozero', tickformat=',.0f')
        show(card('Coverage for each budget', 'Diminishing returns', 'gauge'), chart_layout(fig))

# ---------------------------- Stage 2 ---------------------------------------
else:
    with st.spinner('Stage 2: allocating clinicians and upgrades (can take up to a minute)...'):
        plan0, pt0, _ = solve_stage2(K, limit_km, 0)
        plan, pt, status = solve_stage2(K, limit_km, B, allow_staff, allow_upgrade)
    gap0, gap = float(pt0.unmet.sum()), float(pt.unmet.sum())
    acts = plan[(plan.extra_staff > 0) | (plan.upgrade > 0)].copy()
    n_up, n_staff = int(acts.upgrade.sum()), int(acts.extra_staff.sum())
    spent = COST_STAFF * n_staff + COST_UPG * n_up
    section('Key indicators', f'{scenario} · {action.lower()}')
    c = st.columns(4)
    kpi(c[0], 'Capacity gap', f'{gap0:,.0f}', 'people with every nearby facility full', 'orange', 'alert')
    kpi(c[1], 'Still unserved', f'{gap:,.0f}', 'after Stage 2' if gap > 0.5 else 'gap fully closed',
        'red' if gap > 0.5 else 'green', 'shield' if gap <= 0.5 else 'pin', gap <= 0.5)
    kpi(c[2], 'Budget used', f'{spent} / {B}', 'units · 1 unit = 1 clinician per year', 'blue', 'wallet')
    kpi(c[3], 'Actions', f'{n_up} + {n_staff}', 'upgrades + extra clinicians', 'gold', 'layers')
    section('The plan', f'solver status: {status}', 'search')
    a, b_ = st.columns(2)
    with a:
        g = to_grid(pt0.unmet.values, pt0)
        g[g <= 0.5] = np.nan
        fig = go.Figure()
        base_layer(fig)
        fig.add_trace(go.Heatmap(z=g, colorscale=[[0, '#F7B7B2'], [1, '#B42318']], showscale=False,
                                 hovertemplate='%{z:,.0f} unserved<extra></extra>'))
        swatch(fig, 'Unserved before Stage 2', RED)
        facility_layer(fig, plan[plan.stage == 0], 'Facilities')
        fc, fr = grid_xy(plan.x_m, plan.y_m)
        s1, up, stf = (plan.stage == 1).values, (plan.upgrade == 1).values, plan.extra_staff.values
        fig.add_trace(go.Scatter(x=fc[s1], y=fr[s1], mode='markers', name='Built in Stage 1',
                                 text=plan.name[s1], hovertemplate='%{text}<extra></extra>',
                                 marker=dict(size=15, symbol='star', color='#F5C518', line=dict(color=NAVY, width=1))))
        fig.add_trace(go.Scatter(x=fc[stf > 0], y=fr[stf > 0], mode='markers', name='Extra clinicians',
                                 text=[f'{n}: +{k} clinician(s)' for n, k in zip(plan.name[stf > 0], stf[stf > 0])],
                                 hovertemplate='%{text}<extra></extra>',
                                 marker=dict(size=14 + 4 * stf[stf > 0], symbol='circle-open', color=BLUE,
                                             line=dict(width=2.5))))
        fig.add_trace(go.Scatter(x=fc[up], y=fr[up], mode='markers', name='Upgraded',
                                 text=plan.name[up], hovertemplate='%{text}: upgrade<extra></extra>',
                                 marker=dict(size=20, symbol='square-open', color='#7C3AED', line=dict(width=2.5))))
        show(card('Where to add capacity', 'Red = the gap before Stage 2', 'map'), map_layout(fig))
    with b_:
        showt = acts[['name', 'stage', 'capacity', 'served', 'extra_staff', 'upgrade']].copy()
        showt['stage'] = showt.stage.map({0: 'Existing', 1: 'Stage 1'})
        showt['upgrade'] = showt.upgrade.map({0: '', 1: 'Yes'})
        showt['served'] = showt.served.round(0)
        showt.columns = ['Facility', 'Built in', 'Base capacity', 'People served', 'Extra clinicians', 'Upgrade']
        table(card('Actions chosen', 'Staff or upgrade?', 'grid'),
              showt.sort_values('People served', ascending=False))
    section('How the budget is used', 'assumptions are listed on the right', 'chart')
    a, b_ = st.columns(2)
    with a:
        parts = [COST_UPG * n_up, COST_STAFF * n_staff, max(B - spent, 0)]
        show(card('Budget split', 'Where do the units go?', 'wallet'),
             donut(['Upgrades', 'Clinicians', 'Unused'], parts, [NAVY, TEAL, '#D5DEE3'], f'{spent} / {B}', 'units used')
             if B > 0 else donut(['No budget'], [1], ['#CBD5E1'], '0'))
    with b_:
        box = card('Reading the result', 'Assumptions', 'layers')
        n_s1 = int((acts.stage == 1).sum())
        if n_s1:
            box.markdown(f'<div class="note"><b>The stages are linked.</b> {n_s1} dispensary built in Stage 1 needs '
                         'extra capacity in Stage 2. Year-1 building decisions create Year-2 needs, so the stages '
                         'must be planned together.</div>', unsafe_allow_html=True)
        box.markdown(
            '<p class="assume"><b>Capacity.</b> Dispensary 10,000 people · health centre 50,000 · hospital 150,000.<br>'
            '<b>Clinician.</b> Costs 1 unit and adds 5,000 capacity (at most 4 per facility).<br>'
            '<b>Upgrade.</b> Dispensary to health centre: costs 6 units and adds 40,000 capacity.<br>'
            f'<b>Choice.</b> People may use any of their five nearest facilities within {limit_km:g} km.<br>'
            '<b>Full facility.</b> The people depending on it exceed its assumed capacity.<br><br>'
            'These are planning assumptions, not official figures. With official data the same model can be rerun.</p>',
            unsafe_allow_html=True)

st.markdown(f'<p class="assume" style="margin-top:18px;border-top:1px solid {LINE};padding-top:12px">'
            'Data: Sentinel-2 (Copernicus) via Google Earth Engine · EuroSAT · Google Open Buildings · OpenStreetMap '
            'health facilities (HOT/HDX) · 2022 Tanzania Census household size. Population, capacities and costs are '
            'estimates and assumptions.</p>', unsafe_allow_html=True)
