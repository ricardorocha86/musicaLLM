import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np
import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Literal, List
from collections import Counter
from scipy import stats

st.set_page_config(page_title="Experimento Musical Gemini", layout="wide", initial_sidebar_state="expanded")

# === ESCALAS SEM BRANCO ===
SCALE_BLUE = [[0, '#6baed6'], [0.5, '#3182bd'], [1, '#08519c']]
SCALE_ORANGE = [[0, '#fdae6b'], [0.5, '#e6550d'], [1, '#a63603']]
SCALE_GREEN = [[0, '#74c476'], [0.5, '#31a354'], [1, '#006d2c']]
ACCENT_COLORS = ['#3182bd', '#e6550d', '#31a354', '#756bb1', '#f39c12']

import matplotlib.colors as mcolors
CMAP_BLUE = mcolors.LinearSegmentedColormap.from_list('cb', ['#6baed6', '#08519c'])

try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
except:
    API_KEY, client = None, None

ESTILOS = Literal['rock', 'samba', 'mpb', 'funk', 'sertanejo', 'carimbo', 'forro', 'rap']
class AnaliseMusical(BaseModel):
    estilo: ESTILOS
    justificativa: str
    confianca: float = Field(..., ge=0.0, le=100.0)
    instrumentos: List[str] = Field(..., min_items=1)
    andamento_percebido: Literal['muito_lento', 'lento', 'moderado', 'rapido', 'muito_rapido']
    presenca_vocal: Literal['masculina', 'feminina', 'dueto', 'grupo_coro', 'instrumental_sem_voz']
    densidade_arranjo: Literal['minimalista', 'equilibrada', 'densa_caotica']
    clima: List[str] = Field(..., max_items=3)
    temas: List[str] = Field(..., max_items=3)
    publico_alvo: Literal['infantil', 'jovem', 'adulto', 'familia', 'nicho']
    registro_linguistico: Literal['formal', 'informal', 'giria', 'regional', 'nao_se_aplica']

PROMPT_TEXT = "Analise o áudio como musicólogo especialista."

def listar_audios(pasta):
    if not os.path.exists(pasta): return []
    return sorted([f for f in os.listdir(pasta) if f.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a'))])

def wilson_ci(p, n, z=1.96):
    """Intervalo de Confiança Wilson Score (95%)"""
    if n == 0: return (0, 0)
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    spread = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
    return (max(0, center - spread), min(1, center + spread))

def plot_confusion_matrix(df, title=""):
    labels = sorted(df['estilo_real'].dropna().unique())
    if len(labels) == 0: return None
    cm = confusion_matrix(df['estilo_real'], df['estilo_llm'], labels=labels)
    cm_pct = np.zeros_like(cm, dtype=float)
    for i in range(len(labels)):
        if cm[i].sum() > 0: cm_pct[i] = cm[i] / cm[i].sum() * 100
    fig = px.imshow(cm_pct, x=labels, y=labels, text_auto='.1f', color_continuous_scale=SCALE_BLUE,
                    labels={'x': 'Predito', 'y': 'Real', 'color': '%'}, aspect='auto')
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def get_metrics_table(df):
    labels = sorted(df['estilo_real'].dropna().unique())
    if len(labels) == 0: return pd.DataFrame()
    prec, rec, f1, sup = precision_recall_fscore_support(df['estilo_real'], df['estilo_llm'], labels=labels, zero_division=0)
    rows = [{'Estilo': l, 'Precision': prec[i], 'Recall': rec[i], 'F1': f1[i], 'N': int(sup[i])} for i, l in enumerate(labels)]
    rows.append({'Estilo': '📊 MÉDIA', 'Precision': prec.mean(), 'Recall': rec.mean(), 'F1': f1.mean(), 'N': int(sup.sum())})
    return pd.DataFrame(rows)

def style_metrics_df(df):
    return df.style.format({'Precision': '{:.1%}', 'Recall': '{:.1%}', 'F1': '{:.1%}'}).background_gradient(subset=['F1'], cmap=CMAP_BLUE)

def get_top_confusions(df, n=5):
    erros = df[df['estilo_real'] != df['estilo_llm']]
    if erros.empty: return pd.DataFrame()
    counts = erros.groupby(['estilo_real', 'estilo_llm']).size().reset_index(name='N').sort_values('N', ascending=False).head(n)
    counts['Confusão'] = counts['estilo_real'] + ' → ' + counts['estilo_llm']
    return counts[['Confusão', 'N']]

def plot_ci_comparison(data, x_col, title, color):
    """Cria gráfico de Forest Plot para comparar ICs"""
    fig = go.Figure()
    for i, row in data.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['IC_low'], row['IC_high']], y=[row[x_col], row[x_col]],
            mode='lines', line=dict(color=color, width=3), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[row['Acurácia']], y=[row[x_col]],
            mode='markers+text', marker=dict(color=color, size=12),
            text=[f"{row['Acurácia']:.1%}"], textposition='middle right', showlegend=False
        ))
    fig.update_layout(title=title, xaxis_title="Acurácia", yaxis_title="", xaxis_range=[0, 1],
                      height=200 + len(data) * 40, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def calc_acc_with_ci(df, group_col):
    """Calcula acurácia com IC por grupo"""
    result = df.groupby(group_col).apply(lambda x: pd.Series({
        'Acurácia': (x['estilo_real'] == x['estilo_llm']).mean(),
        'N': len(x),
        'Acertos': (x['estilo_real'] == x['estilo_llm']).sum()
    })).reset_index()
    result['IC_low'] = result.apply(lambda r: wilson_ci(r['Acurácia'], r['N'])[0], axis=1)
    result['IC_high'] = result.apply(lambda r: wilson_ci(r['Acurácia'], r['N'])[1], axis=1)
    return result

FILE_DATA = 'Experimento_Completo_Gemini.xlsx'
PASTA_MUSICAS = 'musicas_IA'

@st.cache_data
def load_data():
    if not os.path.exists(FILE_DATA): return pd.DataFrame()
    try:
        all_dfs = []
        for sheet_df in pd.read_excel(FILE_DATA, sheet_name=None).values():
            for col in ['custo_input', 'custo_output', 'custo_total', 'tempo_execucao', 'confianca']:
                if col in sheet_df.columns and sheet_df[col].dtype == 'object':
                    sheet_df[col] = sheet_df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.').astype(float)
            all_dfs.append(sheet_df)
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    except Exception as e:
        st.error(f"Erro: {e}")
        return pd.DataFrame()

df = load_data()

st.title("🎵 Experimento: Classificação Musical com Gemini")

if df.empty:
    st.warning(f"Aguardando `{FILE_DATA}`. Execute `python musica.py` primeiro.")
    st.stop()

# === SIDEBAR MELHORADA ===
st.sidebar.header("🎛️ Configuração da Análise")

st.sidebar.subheader("Filtros de Dados")
col1, col2 = st.sidebar.columns(2)
with col1:
    sel_modelos = st.multiselect("Modelos", df['modelo'].unique().tolist(), default=df['modelo'].unique().tolist(), label_visibility="collapsed")
with col2:
    sel_prompts = st.multiselect("Prompts", df['prompt_id'].unique().tolist(), default=df['prompt_id'].unique().tolist(), label_visibility="collapsed")

if sel_modelos:
    sel_modelos_labels = ", ".join([m.split('-')[1] for m in sel_modelos])
else:
    sel_modelos_labels = "Nenhum"

st.sidebar.caption(f"**Modelos:** {sel_modelos_labels}")
st.sidebar.caption(f"**Prompts:** {len(sel_prompts)} selecionados")

df_filtered = df[(df['modelo'].isin(sel_modelos)) & (df['prompt_id'].isin(sel_prompts))]

st.sidebar.divider()
st.sidebar.metric("📊 Amostras Ativas", f"{len(df_filtered):,}")
if len(df_filtered) > 0:
    acc_global = (df_filtered['estilo_real'] == df_filtered['estilo_llm']).mean()
    st.sidebar.metric("✅ Acurácia Global", f"{acc_global:.1%}")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Visão Geral", "🔬 Modelos", "📝 Prompts", "🎯 Individual", "❌ Erros", "📋 Dados", "🧪 Playground"])

# =============================================================================
# TAB 1: VISÃO GERAL - FOCO ESTATÍSTICO
# =============================================================================
with tab1:
    st.header("📊 Análise Estatística dos Resultados")
    
    if len(df_filtered) == 0:
        st.warning("Selecione pelo menos um modelo e prompt.")
        st.stop()
    
    total = len(df_filtered)
    acertos = (df_filtered['estilo_real'] == df_filtered['estilo_llm']).sum()
    acuracia = acertos / total
    ci_low, ci_high = wilson_ci(acuracia, total)
    
    # KPIs principais
    st.subheader("Resultado Principal")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.metric("✅ Acurácia Global", f"{acuracia:.1%}", help=f"IC 95%: [{ci_low:.1%}, {ci_high:.1%}]")
        st.caption(f"Intervalo de Confiança 95%: **[{ci_low:.1%} — {ci_high:.1%}]**")
        st.progress(acuracia)
    with c2:
        st.metric("📁 N Amostras", f"{total:,}")
        st.metric("✔️ Acertos", f"{acertos:,}")
    with c3:
        st.metric("❌ Erros", f"{total - acertos:,}")
        st.metric("Taxa de Erro", f"{(1-acuracia):.1%}")
    
    st.divider()
    
    # === FOREST PLOT: TODAS AS 9 CONFIGURAÇÕES ===
    st.subheader("🏆 Ranking das 9 Configurações (Forest Plot)")
    st.caption("Cada barra representa uma combinação Modelo+Prompt. Ordenado do melhor ao pior. Barras que não se sobrepõem diferem significativamente.")
    
    # Calcular acurácia por configuração
    configs_stats = df_filtered.groupby(['modelo', 'prompt_id']).apply(lambda x: pd.Series({
        'Acurácia': (x['estilo_real'] == x['estilo_llm']).mean(),
        'N': len(x),
        'Acertos': (x['estilo_real'] == x['estilo_llm']).sum()
    })).reset_index()
    configs_stats['Config'] = configs_stats['modelo'].str.replace('gemini-', '') + ' + ' + configs_stats['prompt_id'].str.replace('P_', '')
    configs_stats['IC_low'] = configs_stats.apply(lambda r: wilson_ci(r['Acurácia'], r['N'])[0], axis=1)
    configs_stats['IC_high'] = configs_stats.apply(lambda r: wilson_ci(r['Acurácia'], r['N'])[1], axis=1)
    configs_stats = configs_stats.sort_values('Acurácia', ascending=True)
    
    fig_configs = go.Figure()
    n_configs = len(configs_stats)
    colors_gradient = [f'rgb({int(100 + 155*i/n_configs)}, {int(50 + 100*i/n_configs)}, {int(180 - 80*i/n_configs)})' for i in range(n_configs)]
    
    for i, (_, row) in enumerate(configs_stats.iterrows()):
        # Linha do IC
        fig_configs.add_trace(go.Scatter(
            x=[row['IC_low'], row['IC_high']], y=[i, i],
            mode='lines', line=dict(color=colors_gradient[i], width=8), showlegend=False
        ))
        # Ponto central
        fig_configs.add_trace(go.Scatter(
            x=[row['Acurácia']], y=[i],
            mode='markers+text', marker=dict(color='white', size=12, line=dict(color=colors_gradient[i], width=2)),
            text=[f" {row['Acurácia']:.1%}"], textposition='middle right', textfont=dict(size=11), showlegend=False
        ))
    
    fig_configs.update_layout(
        yaxis=dict(tickmode='array', tickvals=list(range(n_configs)), ticktext=configs_stats['Config'].tolist()),
        xaxis_title="Acurácia", xaxis_range=[0.5, 1.05], 
        height=80 + n_configs * 35, margin=dict(l=20, r=20, t=20, b=40)
    )
    st.plotly_chart(fig_configs, use_container_width=True)
    
    # Conclusões
    best_config = configs_stats.iloc[-1]
    worst_config = configs_stats.iloc[0]
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🥇 **Melhor:** {best_config['Config']} = {best_config['Acurácia']:.1%}")
    with col2:
        st.warning(f"🥉 **Pior:** {worst_config['Config']} = {worst_config['Acurácia']:.1%}")
    
    if best_config['IC_low'] > worst_config['IC_high']:
        st.info("📈 A diferença entre a melhor e pior configuração é **estatisticamente significativa**.")
    
    st.divider()
    
    # Comparação visual de ICs por ESTILO
    st.subheader("🎯 Acurácia por Estilo Musical (com IC 95%)")
    st.caption("Barras de erro mostram o intervalo de confiança. Se não se sobrepõem, a diferença é estatisticamente significativa.")
    
    acc_by_style = calc_acc_with_ci(df_filtered, 'estilo_real')
    acc_by_style = acc_by_style.sort_values('Acurácia', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=acc_by_style['estilo_real'], x=acc_by_style['Acurácia'], orientation='h',
        error_x=dict(type='data', symmetric=False, 
                     array=acc_by_style['IC_high'] - acc_by_style['Acurácia'],
                     arrayminus=acc_by_style['Acurácia'] - acc_by_style['IC_low']),
        marker_color='#3182bd',
        text=[f"{a:.1%}" for a in acc_by_style['Acurácia']], textposition='outside'
    ))
    fig.update_layout(xaxis_range=[0, 1.15], xaxis_title="Acurácia", yaxis_title="Estilo", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights automáticos
    best_style = acc_by_style.iloc[-1]
    worst_style = acc_by_style.iloc[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🏆 **{best_style['estilo_real'].upper()}** é o mais fácil: {best_style['Acurácia']:.1%} (N={int(best_style['N'])})")
    with col2:
        st.error(f"⚠️ **{worst_style['estilo_real'].upper()}** é o mais difícil: {worst_style['Acurácia']:.1%} (N={int(worst_style['N'])})")
    
    # ICs se sobrepõem?
    if best_style['IC_low'] > worst_style['IC_high']:
        st.info("📈 A diferença entre o melhor e pior estilo é **estatisticamente significativa** (ICs não se sobrepõem).")
    else:
        st.warning("⚠️ A diferença pode não ser significativa (ICs se sobrepõem).")
    
    st.divider()
    
    # Matriz + Métricas
    col_cm, col_met = st.columns([1.2, 1])
    with col_cm:
        st.subheader("Matriz de Confusão")
        fig_cm = plot_confusion_matrix(df_filtered)
        if fig_cm: st.plotly_chart(fig_cm, use_container_width=True)
    with col_met:
        st.subheader("Métricas Detalhadas")
        met = get_metrics_table(df_filtered)
        if not met.empty: st.dataframe(style_metrics_df(met), hide_index=True, use_container_width=True)
    
    # Top confusões
    st.subheader("🔄 Principais Erros de Classificação")
    conf = get_top_confusions(df_filtered, 10)
    if not conf.empty:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(conf, hide_index=True)
        with col2:
            fig = px.bar(conf.head(5), x='N', y='Confusão', orientation='h', color='N', color_continuous_scale=SCALE_ORANGE)
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2: ANÁLISE POR MODELO
# =============================================================================
with tab2:
    st.header("🔬 Comparação Estatística entre Modelos")
    
    # Calcular stats por modelo
    acc_by_model = calc_acc_with_ci(df, 'modelo')
    acc_by_model['Custo'] = df.groupby('modelo')['custo_total'].mean().values if 'custo_total' in df.columns else 0
    acc_by_model['Tempo'] = df.groupby('modelo')['tempo_execucao'].mean().values if 'tempo_execucao' in df.columns else 0
    
    # Forest Plot de Modelos
    st.subheader("📊 Acurácia por Modelo (Forest Plot)")
    st.caption("Visualização de IC 95%. Barras que não se sobrepõem indicam diferença significativa.")
    
    fig = go.Figure()
    colors = ACCENT_COLORS[:len(acc_by_model)]
    for i, (_, row) in enumerate(acc_by_model.iterrows()):
        # Linha do IC
        fig.add_trace(go.Scatter(
            x=[row['IC_low'], row['IC_high']], y=[i, i],
            mode='lines', line=dict(color=colors[i], width=8), name=row['modelo'], showlegend=False
        ))
        # Ponto central
        fig.add_trace(go.Scatter(
            x=[row['Acurácia']], y=[i],
            mode='markers+text', marker=dict(color='white', size=14, line=dict(color=colors[i], width=3)),
            text=[f" {row['Acurácia']:.1%}"], textposition='middle right', textfont=dict(size=14),
            name=row['modelo'], showlegend=False
        ))
    
    fig.update_layout(
        yaxis=dict(tickmode='array', tickvals=list(range(len(acc_by_model))), ticktext=acc_by_model['modelo'].tolist()),
        xaxis_title="Acurácia", xaxis_range=[0.5, 1.05], height=200, margin=dict(l=20, r=20, t=20, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Conclusões automáticas
    best_model = acc_by_model.loc[acc_by_model['Acurácia'].idxmax()]
    worst_model = acc_by_model.loc[acc_by_model['Acurácia'].idxmin()]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"🏆 **Melhor:** {best_model['modelo']}\n\nAcurácia: {best_model['Acurácia']:.1%}")
    with col2:
        if best_model['IC_low'] > worst_model['IC_high']:
            st.info("📈 Diferença **SIGNIFICATIVA** entre modelos")
        else:
            st.warning("⚠️ Diferença pode não ser significativa")
    with col3:
        cheapest = acc_by_model.loc[acc_by_model['Custo'].idxmin()]
        st.info(f"💰 **Mais barato:** {cheapest['modelo']}\n\n${cheapest['Custo']:.4f}/req")
    
    st.divider()
    
    # Tabela completa
    st.subheader("📋 Tabela Comparativa")
    display = acc_by_model[['modelo', 'Acurácia', 'IC_low', 'IC_high', 'N', 'Custo', 'Tempo']].copy()
    display['IC 95%'] = display.apply(lambda r: f"[{r['IC_low']:.1%}, {r['IC_high']:.1%}]", axis=1)
    display = display[['modelo', 'Acurácia', 'IC 95%', 'N', 'Custo', 'Tempo']]
    st.dataframe(display.style.format({'Acurácia': '{:.1%}', 'Custo': '${:.4f}', 'Tempo': '{:.2f}s'}), hide_index=True)
    
    # Detalhamento por modelo
    st.divider()
    st.subheader("🔎 Detalhamento por Modelo")
    sel_modelo = st.selectbox("Selecione:", acc_by_model['modelo'].tolist())
    df_mod = df[df['modelo'] == sel_modelo]
    
    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig = plot_confusion_matrix(df_mod, f"Matriz - {sel_modelo}")
        if fig: st.plotly_chart(fig, use_container_width=True)
    with col2:
        met = get_metrics_table(df_mod)
        if not met.empty: st.dataframe(style_metrics_df(met), hide_index=True)

# =============================================================================
# TAB 3: ANÁLISE POR PROMPT
# =============================================================================
with tab3:
    st.header("📝 Impacto do Prompt na Performance")
    
    prompt_order = ['P_Basico', 'P_Intermediario', 'P_Avancado']
    
    acc_by_prompt = calc_acc_with_ci(df, 'prompt_id')
    acc_by_prompt['ordem'] = acc_by_prompt['prompt_id'].map({p: i for i, p in enumerate(prompt_order)})
    acc_by_prompt = acc_by_prompt.sort_values('ordem')
    
    # Forest Plot de Prompts
    st.subheader("📊 Acurácia por Prompt (Forest Plot)")
    
    fig = go.Figure()
    colors = ['#6baed6', '#3182bd', '#08519c']
    for i, (_, row) in enumerate(acc_by_prompt.iterrows()):
        fig.add_trace(go.Scatter(
            x=[row['IC_low'], row['IC_high']], y=[i, i],
            mode='lines', line=dict(color=colors[i], width=8), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[row['Acurácia']], y=[i],
            mode='markers+text', marker=dict(color='white', size=14, line=dict(color=colors[i], width=3)),
            text=[f" {row['Acurácia']:.1%}"], textposition='middle right', textfont=dict(size=14), showlegend=False
        ))
    
    fig.update_layout(
        yaxis=dict(tickmode='array', tickvals=list(range(len(acc_by_prompt))), ticktext=acc_by_prompt['prompt_id'].tolist()),
        xaxis_title="Acurácia", xaxis_range=[0.5, 1.05], height=180, margin=dict(l=20, r=20, t=20, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Análise de tendência
    acc_list = acc_by_prompt['Acurácia'].tolist()
    if len(acc_list) >= 2:
        delta = acc_list[-1] - acc_list[0]
        col1, col2 = st.columns(2)
        with col1:
            if delta > 0.02:
                st.success(f"✅ Prompts elaborados **MELHORAM** a acurácia em **+{delta:.1%}**")
            elif delta < -0.02:
                st.error(f"❌ Prompts elaborados **PIORAM** a acurácia em **{delta:.1%}**")
            else:
                st.info(f"≈ Impacto **MÍNIMO** (Δ = {delta:.1%})")
        with col2:
            # Teste de significância
            if acc_by_prompt.iloc[-1]['IC_low'] > acc_by_prompt.iloc[0]['IC_high']:
                st.info("📈 Diferença é **estatisticamente significativa**")
            elif acc_by_prompt.iloc[0]['IC_low'] > acc_by_prompt.iloc[-1]['IC_high']:
                st.warning("📉 Prompt avançado é significativamente **PIOR**")
            else:
                st.warning("⚠️ ICs se sobrepõem - diferença **não significativa**")
    
    st.divider()
    
    # Heatmap Modelo × Prompt
    st.subheader("🗺️ Heatmap: Modelo × Prompt")
    pivot = df.pivot_table(values='estilo_llm', index='modelo', columns='prompt_id', 
                           aggfunc=lambda x: (df.loc[x.index, 'estilo_real'] == x).mean())
    pivot = pivot[[c for c in prompt_order if c in pivot.columns]]
    fig = px.imshow(pivot * 100, text_auto='.1f', color_continuous_scale=SCALE_BLUE)
    fig.update_layout(coloraxis_colorbar_title="Acurácia %")
    st.plotly_chart(fig, use_container_width=True)
    
    # Melhor combinação
    best_combo = df.groupby(['modelo', 'prompt_id']).apply(lambda x: (x['estilo_real'] == x['estilo_llm']).mean())
    best_idx = best_combo.idxmax()
    st.success(f"🏆 **Melhor combinação:** {best_idx[0]} + {best_idx[1]} = {best_combo.max():.1%}")

# =============================================================================
# TAB 4: INDIVIDUAL
# =============================================================================
with tab4:
    st.header("🎯 Análise de Configuração Individual")
    
    configs = [f"{r['modelo']} + {r['prompt_id']}" for _, r in df.groupby(['modelo', 'prompt_id']).size().reset_index().iterrows()]
    sel = st.selectbox("Configuração:", configs)
    
    if sel:
        m, p = sel.split(' + ')
        dfc = df[(df['modelo'] == m) & (df['prompt_id'] == p)]
        acc = (dfc['estilo_real'] == dfc['estilo_llm']).mean() if len(dfc) > 0 else 0
        ci = wilson_ci(acc, len(dfc))
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Acurácia", f"{acc:.1%}", help=f"IC: [{ci[0]:.1%}, {ci[1]:.1%}]")
        c2.metric("N", len(dfc))
        c3.metric("Custo", f"${dfc['custo_total'].sum():.2f}" if 'custo_total' in dfc.columns else "N/A")
        
        st.caption(f"**IC 95%:** [{ci[0]:.1%}, {ci[1]:.1%}]")
        
        col1, col2 = st.columns([1.2, 1])
        with col1:
            fig = plot_confusion_matrix(dfc)
            if fig: st.plotly_chart(fig, use_container_width=True)
        with col2:
            met = get_metrics_table(dfc)
            if not met.empty: st.dataframe(style_metrics_df(met), hide_index=True)

# =============================================================================
# TAB 5: ERROS
# =============================================================================
with tab5:
    st.header("❌ Análise de Erros")
    erros = df_filtered[df_filtered['estilo_real'] != df_filtered['estilo_llm']]
    st.warning(f"{len(erros)} erros ({len(erros)/len(df_filtered)*100:.1f}%)" if len(df_filtered) > 0 else "Sem dados")
    
    c1, c2 = st.columns(2)
    with c1: f_real = st.multiselect("Estilo Real:", erros['estilo_real'].unique() if not erros.empty else [])
    with c2: f_pred = st.multiselect("Estilo Predito:", erros['estilo_llm'].unique() if not erros.empty else [])
    if f_real: erros = erros[erros['estilo_real'].isin(f_real)]
    if f_pred: erros = erros[erros['estilo_llm'].isin(f_pred)]
    
    for _, r in erros.head(10).iterrows():
        with st.expander(f"❌ {r['arquivo']} | {r['estilo_real']} → {r['estilo_llm']}"):
            c1, c2 = st.columns([1, 2])
            with c1:
                path = os.path.join(PASTA_MUSICAS, r['arquivo'])
                if os.path.exists(path): st.audio(path)
            with c2:
                st.write(f"**Modelo:** {r['modelo']} | **Prompt:** {r['prompt_id']}")
                st.write(f"**Confiança:** {r.get('confianca', 0):.0f}%")
                st.caption(r.get('justificativa', ''))

# TAB 6: DADOS
with tab6:
    st.dataframe(df_filtered, height=600, use_container_width=True)
    st.download_button("📥 CSV", df_filtered.to_csv(index=False), "dados.csv")

# TAB 7: PLAYGROUND
with tab7:
    st.header("🧪 Playground - Teste sua Música")
    
    if not client:
        st.error("API Key não configurada em .streamlit/secrets.toml")
    else:
        # Definir prompts exatos do experimento
        PROMPTS_EXPERIMENTO = {
            "P_Basico": """Você é um classificador musical automatizado.
Sua tarefa é ouvir o arquivo de áudio fornecido e preencher os metadados solicitados no schema JSON.

INSTRUÇÕES CRÍTICAS:
1. Analise o áudio focando na instrumentação, ritmo e voz.
2. Classifique o estilo estritamente dentro das opções permitidas.
3. Para o campo 'confianca', use uma escala percentual de 0.0 a 100.0 (ex: 95.5).
4. Seja objetivo e direto na justificativa.""",
            
            "P_Intermediario": """Atue como um Musicólogo Especialista em gêneros brasileiros e globais.
Analise o áudio com rigor técnico para extrair características acústicas e sociolinguísticas.

--- GUIA DE CLASSIFICAÇÃO (TAXONOMIA) ---
1. ROCK: Guitarras distorcidas, bateria forte 4/4, baixo elétrico.
2. SAMBA: Ritmo binário (2/4), síncope, percussão (surdo, tamborim), cavaquinho.
3. MPB: Harmonia sofisticada, foco na lírica, influências Jazz/Bossa.
4. FUNK (BR): Batida repetitiva, graves pesados, estética minimalista.
5. SERTANEJO: Violão/sanfona, duetos vocais (terças), temas amor/sofrência.
6. CARIMBÓ: Ritmo acelerado do norte, curimbó, metais, maracas.
7. FORRÓ: Sanfona, zabumba, triângulo. Ritmo baião/xote.
8. RAP: Fala rítmica (flow), beats eletrônicos, registro urbano.

--- INSTRUÇÕES ---
- Valide densidade_arranjo e registro_linguistico.
- Confiança: 0.0 a 100.0 baseada na clareza dos sinais.""",
            
            "P_Avancado": """Atue como um Musicólogo Especialista em gêneros brasileiros e globais.
Analise o áudio com rigor técnico para extrair características acústicas e sociolinguísticas.

--- GUIA DE CLASSIFICAÇÃO (TAXONOMIA) ---
1. ROCK: Guitarras distorcidas, bateria forte 4/4, baixo elétrico.
2. SAMBA: Ritmo binário (2/4), síncope, percussão (surdo, tamborim), cavaquinho.
3. MPB: Harmonia sofisticada, foco na lírica, influências Jazz/Bossa.
4. FUNK (BR): Batida repetitiva, graves pesados, estética minimalista.
5. SERTANEJO: Violão/sanfona, duetos vocais (terças), temas amor/sofrência.
6. CARIMBÓ: Ritmo acelerado do norte, curimbó, metais, maracas.
7. FORRÓ: Sanfona, zabumba, triângulo. Ritmo baião/xote.
8. RAP: Fala rítmica (flow), beats eletrônicos, registro urbano.

--- EXEMPLOS DE REFERÊNCIA ---
ROCK: Legião Urbana "Tempo Perdido", Pitty "Admirável Chip Novo"
SAMBA: Zeca Pagodinho "Deixa a Vida Me Levar", Cartola "O Mundo é um Moinho"
MPB: Elis Regina "Águas de Março", Djavan "Oceano"
FUNK: Anitta "Show das Poderosas", MC Kevinho "Olha a Explosão"
SERTANEJO: Chitãozinho "Evidências", Marília Mendonça "Infiel"
FORRÓ: Luiz Gonzaga "Asa Branca", Wesley Safadão "Camarote"
RAP: Racionais MC's "Diário de um Detento", Emicida "Levanta e Anda"

Se o áudio soar como essas referências, classifique no respectivo estilo."""
        }
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Upload de Áudio")
            uploaded_file = st.file_uploader("Escolha um arquivo de áudio:", type=['mp3', 'wav', 'ogg', 'm4a'])
            
            if uploaded_file:
                st.audio(uploaded_file)
                st.caption(f"Arquivo: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
            
            st.divider()
            
            st.subheader("⚙️ Configuração")
            modelo = st.selectbox("Modelo Gemini:", ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-3-flash-preview"])
            prompt_sel = st.selectbox("Prompt:", list(PROMPTS_EXPERIMENTO.keys()), 
                                       format_func=lambda x: {"P_Basico": "🟢 Básico", "P_Intermediario": "🟡 Intermediário", "P_Avancado": "🔴 Avançado"}[x])
            temperatura = st.slider("Temperatura:", 0.0, 2.0, 0.0, 0.1, help="0 = determinístico (como no experimento)")
            
            executar = st.button("🚀 Classificar Áudio", type="primary", use_container_width=True)
        
        with col2:
            st.subheader("📝 Prompt Selecionado")
            st.code(PROMPTS_EXPERIMENTO[prompt_sel], language="text")
        
        if executar:
            if not uploaded_file:
                st.error("Por favor, faça upload de um arquivo de áudio.")
            else:
                with st.spinner("🎵 Analisando áudio com Gemini...", show_time=True):
                    try:
                        audio_bytes = uploaded_file.read()
                        
                        # Determinar MIME type
                        ext = uploaded_file.name.split('.')[-1].lower()
                        mime_types = {'mp3': 'audio/mp3', 'wav': 'audio/wav', 'ogg': 'audio/ogg', 'm4a': 'audio/mp4'}
                        mime = mime_types.get(ext, 'audio/mp3')
                        
                        response = client.models.generate_content(
                            model=modelo,
                            contents=[PROMPTS_EXPERIMENTO[prompt_sel], types.Part.from_bytes(data=audio_bytes, mime_type=mime)],
                            config=types.GenerateContentConfig(
                                temperature=temperatura,
                                response_mime_type="application/json",
                                response_schema=AnaliseMusical
                            )
                        )
                        
                        resultado = response.parsed
                        
                        st.success("✅ Análise concluída!")
                        
                        # Resultado principal
                        c1, c2, c3 = st.columns(3)
                        c1.metric("🎵 Estilo", resultado.estilo.upper())
                        c2.metric("🎯 Confiança", f"{resultado.confianca:.0f}%")
                        c3.metric("🎸 Andamento", resultado.andamento_percebido)
                        
                        st.markdown(f"**Justificativa:** {resultado.justificativa}")
                        
                        with st.expander("📊 Detalhes Completos", expanded=True):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write(f"**Instrumentos:** {', '.join(resultado.instrumentos)}")
                                st.write(f"**Presença Vocal:** {resultado.presenca_vocal}")
                                st.write(f"**Densidade:** {resultado.densidade_arranjo}")
                            with col_b:
                                st.write(f"**Clima:** {', '.join(resultado.clima)}")
                                st.write(f"**Temas:** {', '.join(resultado.temas)}")
                                st.write(f"**Público Alvo:** {resultado.publico_alvo}")
                                st.write(f"**Registro:** {resultado.registro_linguistico}")
                        
                        with st.expander("🔧 JSON Completo"):
                            st.json(resultado.model_dump_json())
                        
                    except Exception as e:
                        st.error(f"Erro na classificação: {str(e)}")

