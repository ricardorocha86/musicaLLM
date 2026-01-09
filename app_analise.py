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
import matplotlib.colors as mcolors
from metodologia import METODOLOGIA_SUNO, METODOLOGIA_ABERTO, METODOLOGIA_REAIS

st.set_page_config(page_title="Experimento Musical Gemini", layout="wide", initial_sidebar_state="expanded")

# === ESCALAS SEM BRANCO ===
SCALE_BLUE = [[0, '#6baed6'], [0.5, '#3182bd'], [1, '#08519c']]
SCALE_ORANGE = [[0, '#fdae6b'], [0.5, '#e6550d'], [1, '#a63603']]
SCALE_GREEN = [[0, '#74c476'], [0.5, '#31a354'], [1, '#006d2c']]
ACCENT_COLORS = ['#3182bd', '#e6550d', '#31a354', '#756bb1', '#f39c12']

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

# === NAVEGAÇÃO ===
st.sidebar.title("Navegação")
page = st.sidebar.radio("Ir para:", ["Experimento IA (Suno)", "Experimento IA (Aberto)", "Músicas Reais (Gabarito)"])

if page == "Experimento IA (Suno)":
    st.title("🎵 Experimento: Classificação Musical com Gemini")
    
    if df.empty:
        st.warning(f"Aguardando `{FILE_DATA}`. Execute `python musica.py` primeiro.")
        st.stop()

    # === SIDEBAR FILTERS ===
    st.sidebar.header("🎛️ Filtros")
    
    st.sidebar.caption("**Modelos:**")
    sel_modelos = st.sidebar.pills("sel_mod", df['modelo'].unique().tolist(), selection_mode="multi", default=df['modelo'].unique().tolist(), label_visibility="collapsed")
    
    st.sidebar.caption("**Prompts:**")
    sel_prompts = st.sidebar.pills("sel_prom", df['prompt_id'].unique().tolist(), selection_mode="multi", default=df['prompt_id'].unique().tolist(), label_visibility="collapsed")
    
    df_filtered = df[(df['modelo'].isin(sel_modelos)) & (df['prompt_id'].isin(sel_prompts))]
    
    st.sidebar.divider()
    st.sidebar.metric("📊 Amostras", f"{len(df_filtered):,}")
    if len(df_filtered) > 0:
        acc_global = (df_filtered['estilo_real'] == df_filtered['estilo_llm']).mean()
        st.sidebar.metric("✅ Acurácia", f"{acc_global:.1%}")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["📊 Visão Geral", "🔬 Modelos", "📝 Prompts", "🎯 Individual", "❌ Erros", "📋 Dados", "🧪 Playground", "ℹ️ Sobre", "📖 Metodologia"])
    
    with tab1:
        st.header("📊 Análise Estatística dos Resultados")
        if len(df_filtered) == 0:
            st.warning("Selecione pelo menos um modelo e prompt.")
        else:
            total = len(df_filtered)
            acertos = (df_filtered['estilo_real'] == df_filtered['estilo_llm']).sum()
            acuracia = acertos / total
            ci_low, ci_high = wilson_ci(acuracia, total)
            
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
            
            # === FOREST PLOT ===
            st.subheader("🏆 Ranking das 9 Configurações (Forest Plot)")
            configs_stats = df_filtered.groupby(['modelo', 'prompt_id']).apply(lambda x: pd.Series({
                'Acurácia': (x['estilo_real'] == x['estilo_llm']).mean(),
                'N': len(x)
            })).reset_index()
            configs_stats['Config'] = configs_stats['modelo'].str.replace('gemini-', '') + ' + ' + configs_stats['prompt_id'].str.replace('P_', '')
            configs_stats['IC_low'] = configs_stats.apply(lambda r: wilson_ci(r['Acurácia'], r['N'])[0], axis=1)
            configs_stats['IC_high'] = configs_stats.apply(lambda r: wilson_ci(r['Acurácia'], r['N'])[1], axis=1)
            configs_stats = configs_stats.sort_values('Acurácia', ascending=True)
            
            fig_configs = go.Figure()
            n_configs = len(configs_stats)
            colors_gradient = [f'rgb({int(100 + 155*i/n_configs)}, {int(50 + 100*i/n_configs)}, {int(180 - 80*i/n_configs)})' for i in range(n_configs)]
            
            for i, (_, row) in enumerate(configs_stats.iterrows()):
                fig_configs.add_trace(go.Scatter(x=[row['IC_low'], row['IC_high']], y=[i, i], mode='lines', line=dict(color=colors_gradient[i], width=8), showlegend=False))
                fig_configs.add_trace(go.Scatter(x=[row['Acurácia']], y=[i], mode='markers+text', marker=dict(color='white', size=12, line=dict(color=colors_gradient[i], width=2)), text=[f" {row['Acurácia']:.1%}"], textposition='middle right', showlegend=False))
            
            fig_configs.update_layout(yaxis=dict(tickmode='array', tickvals=list(range(n_configs)), ticktext=configs_stats['Config'].tolist()), xaxis_title="Acurácia", xaxis_range=[0.5, 1.05], height=80 + n_configs * 35, margin=dict(l=20, r=20, t=20, b=40))
            st.plotly_chart(fig_configs, use_container_width=True)
            
            st.divider()
            
            # Comparação visual de ICs por ESTILO
            st.subheader("🎯 Acurácia por Estilo Musical (com IC 95%)")
            acc_by_style = calc_acc_with_ci(df_filtered, 'estilo_real').sort_values('Acurácia', ascending=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=acc_by_style['estilo_real'], x=acc_by_style['Acurácia'], orientation='h',
                error_x=dict(type='data', symmetric=False, array=acc_by_style['IC_high'] - acc_by_style['Acurácia'], arrayminus=acc_by_style['Acurácia'] - acc_by_style['IC_low']),
                marker_color='#3182bd', text=[f"{a:.1%}" for a in acc_by_style['Acurácia']], textposition='outside'
            ))
            fig.update_layout(xaxis_range=[0, 1.15], xaxis_title="Acurácia", yaxis_title="Estilo", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            col_cm, col_met = st.columns([1.2, 1])
            with col_cm:
                st.subheader("Matriz de Confusão")
                fig_cm = plot_confusion_matrix(df_filtered)
                if fig_cm: st.plotly_chart(fig_cm, use_container_width=True)
            with col_met:
                st.subheader("Métricas Detalhadas")
                met = get_metrics_table(df_filtered)
                if not met.empty: st.dataframe(style_metrics_df(met), hide_index=True, use_container_width=True)
    
    with tab2:
        st.header("🔬 Comparação Estatística entre Modelos")
        acc_by_model = calc_acc_with_ci(df, 'modelo')
        acc_by_model['Custo'] = df.groupby('modelo')['custo_total'].mean().values if 'custo_total' in df.columns else 0
        acc_by_model['Tempo'] = df.groupby('modelo')['tempo_execucao'].mean().values if 'tempo_execucao' in df.columns else 0
        
        fig = go.Figure()
        colors = ACCENT_COLORS[:len(acc_by_model)]
        for i, (_, row) in enumerate(acc_by_model.iterrows()):
            fig.add_trace(go.Scatter(x=[row['IC_low'], row['IC_high']], y=[i, i], mode='lines', line=dict(color=colors[i], width=8), showlegend=False))
            fig.add_trace(go.Scatter(x=[row['Acurácia']], y=[i], mode='markers+text', marker=dict(color='white', size=14, line=dict(color=colors[i], width=3)), text=[f" {row['Acurácia']:.1%}"], textposition='middle right', showlegend=False))
        fig.update_layout(yaxis=dict(tickmode='array', tickvals=list(range(len(acc_by_model))), ticktext=acc_by_model['modelo'].tolist()), xaxis_title="Acurácia", xaxis_range=[0.5, 1.05], height=200, margin=dict(l=20, r=20, t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(acc_by_model.style.format({'Acurácia': '{:.1%}', 'Custo': '${:.4f}', 'Tempo': '{:.2f}s'}), hide_index=True)

    with tab3:
        st.header("📝 Impacto do Prompt")
        acc_by_prompt = calc_acc_with_ci(df, 'prompt_id').sort_values('Acurácia')
        st.dataframe(acc_by_prompt.style.format({'Acurácia': '{:.1%}'}), hide_index=True)

    with tab4:
        st.header("🎯 Detalhe Individual")
        configs = [f"{r['modelo']} + {r['prompt_id']}" for _, r in df.groupby(['modelo', 'prompt_id']).size().reset_index().iterrows()]
        sel = st.selectbox("Configuração:", configs)
        if sel:
            m, p = sel.split(' + ')
            dfc = df[(df['modelo'] == m) & (df['prompt_id'] == p)]
            col1, col2 = st.columns([1.2, 1])
            with col1:
                fig = plot_confusion_matrix(dfc)
                if fig: st.plotly_chart(fig, use_container_width=True)
            with col2:
                met = get_metrics_table(dfc)
                if not met.empty: st.dataframe(style_metrics_df(met), hide_index=True)

    with tab5:
        st.header("❌ Análise de Erros")
        erros = df_filtered[df_filtered['estilo_real'] != df_filtered['estilo_llm']]
        if not erros.empty:
            st.dataframe(erros[['arquivo', 'estilo_real', 'estilo_llm', 'modelo', 'prompt_id', 'justificativa']], use_container_width=True)
        else:
            st.success("Nenhum erro encontrado nos filtros atuais.")

    with tab6:
        st.dataframe(df_filtered, height=600, use_container_width=True)

    with tab7:
        st.header("🧪 Playground - Teste sua Música")
        if not client:
            st.error("API Key não configurada.")
        else:
            st.info("Para usar o Playground, acesse o script original.")

    with tab8:
        st.markdown("## Sobre o Experimento\nAnálise de capacidade de modelos Gemini em classificar gêneros musicais brasileiros.")

    with tab9:
        st.markdown(METODOLOGIA_SUNO)



elif page == "Experimento IA (Aberto)":
    st.title("🔓 Experimento: Classificação Aberta")
    st.caption("Análise onde o modelo pode sugerir múltiplos gêneros livremente (List[str]).")
    
    FILE_ABERTO = "Experimento_Completo_Gemini_Aberto.xlsx"
    if not os.path.exists(FILE_ABERTO):
        st.error(f"Arquivo `{FILE_ABERTO}` não encontrado. Execute `musica2.py` primeiro.")
        st.stop()
        
    @st.cache_data
    def load_data_aberto():
        try:
            all_dfs = []
            xl = pd.ExcelFile(FILE_ABERTO)
            for sheet in xl.sheet_names:
                sheet_df = xl.parse(sheet)
                all_dfs.append(sheet_df)
            return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        except Exception as e:
            st.error(f"Erro ao carregar excel: {e}")
            return pd.DataFrame()

    df_a = load_data_aberto()
    
    # Processamento Inicial
    # Converter estilos_llm de string "A, B, C" para lista
    df_a['estilos_list'] = df_a['estilos_llm'].fillna('').apply(lambda x: [s.strip().lower() for s in str(x).split(',') if s.strip()])
    
    # Definir Critério de Acerto "Broad" (Se o estilo real está contido na lista)
    def check_hit(row):
        real = str(row['estilo_real']).strip().lower()
        pred_list = row['estilos_list']
        # Verifica se real está na lista OU se algum item da lista contem o real (ex: 'samba' in 'samba de raiz')
        for p in pred_list:
            if real == p or real in p or p in real:
                return 1
        return 0

    df_a['Hit'] = df_a.apply(check_hit, axis=1)
    df_a['Qtd_Tags'] = df_a['estilos_list'].apply(len)
    
    # === SIDEBAR ===
    st.sidebar.header("🎛️ Filtros (Aberto)")
    sel_mod_a = st.sidebar.pills("filtro_mod_a", df_a['modelo'].unique(), selection_mode="multi", default=df_a['modelo'].unique(), label_visibility="collapsed")
    sel_prom_a = st.sidebar.pills("filtro_prom_a", df_a['prompt_id'].unique(), selection_mode="multi", default=df_a['prompt_id'].unique(), label_visibility="collapsed")
    
    df_af = df_a[(df_a['modelo'].isin(sel_mod_a)) & (df_a['prompt_id'].isin(sel_prom_a))]
    
    # KPIs
    st.sidebar.divider()
    acc_broad = df_af['Hit'].mean()
    st.sidebar.metric("✅ Acurácia (Broad)", f"{acc_broad:.1%}", help="Considera acerto se o estilo real estiver na lista de sugestões.")
    st.sidebar.metric("🏷️ Tags / Música", f"{df_af['Qtd_Tags'].mean():.1f}")
    
    # TABS
    tab_ov, tab_mod, tab_gen, tab_vocab, tab_tech, tab_data, tab_metod = st.tabs(["📊 Visão Geral", "🔬 Modelos", "🎸 Gêneros", "🗣️ Vocabulário", "🎚️ Técnico", "📋 Dados", "📖 Metodologia"])
    
    with tab_ov:
        st.header("📊 Resultado Geral (Classificação Livre)")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Acurácia (Broad)", f"{acc_broad:.1%}")
        c2.metric("Confiança Média", f"{df_af['confianca'].mean():.1f}%")
        c3.metric("Custo Total", f"${df_af['custo_total'].astype(float).sum():.4f}")
        
        st.divider()
        st.subheader("🏆 Ranking de Configurações (Acurácia Broad)")
        
        # Calculate Config Rank safely
        if not df_af.empty:
            conf_rank = df_af.groupby(['modelo', 'prompt_id']).apply(lambda x: pd.Series({
                'Acurácia': x['Hit'].mean(),
                'N': len(x),
                'Tags_Media': x['Qtd_Tags'].mean()
            })).reset_index().sort_values('Acurácia', ascending=True)
            
            fig_r = px.bar(conf_rank, x='Acurácia', y='modelo', color='prompt_id', orientation='h', barmode='group',
                           text_auto='.1%', title="Acurácia por Modelo e Prompt")
            fig_r.update_layout(xaxis_range=[0, 1.1])
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.warning("Sem dados para ranking.")
        
    with tab_mod:
        st.header("🔬 Comparativo de Modelos")
        # Forest plot para modelos
        if not df_a.empty:
            mod_stats = df_a.groupby('modelo').apply(lambda x: pd.Series({'Acc': x['Hit'].mean(), 'N': len(x)})).reset_index().sort_values('Acc')
            
            fig_fp = go.Figure()
            for i, row in mod_stats.iterrows():
                low, high = wilson_ci(row['Acc'], row['N'])
                fig_fp.add_trace(go.Scatter(x=[low, high], y=[row['modelo'], row['modelo']], mode='lines', line=dict(width=5, color='#3182bd')))
                fig_fp.add_trace(go.Scatter(x=[row['Acc']], y=[row['modelo']], mode='markers', marker=dict(size=12, color='#08519c')))
                
            fig_fp.update_layout(title="Intervalo de Confiança (95%) por Modelo", xaxis_title="Acurácia (Broad)", xaxis_range=[0, 1])
            st.plotly_chart(fig_fp, use_container_width=True)
        else:
            st.warning("Sem dados.")

    with tab_gen:
        st.header("🎸 Análise por Gênero Real")
        all_real_styles = sorted(df_af['estilo_real'].unique())
        sel_style = st.selectbox("Selecione um Estilo Real:", all_real_styles)
        
        if sel_style:
            subset = df_af[df_af['estilo_real'] == sel_style]
            st.metric(f"Acurácia para {sel_style.upper()}", f"{subset['Hit'].mean():.1%}")
            
            # Quais tags o modelo gerou para este estilo?
            all_tags_style = []
            for l in subset['estilos_list']: all_tags_style.extend(l)
            
            counts = pd.Series(all_tags_style).value_counts().head(20).reset_index()
            counts.columns = ['Tag Gerada', 'Frequência']
            
            fig_bar = px.bar(counts, x='Frequência', y='Tag Gerada', orientation='h', 
                             title=f"Top 20 Tags geradas para músicas de '{sel_style}'", color='Frequência', color_continuous_scale=SCALE_ORANGE)
            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab_vocab:
        st.header("🗣️ Vocabulário do Modelo")
        st.markdown("Quais termos os modelos mais usam na classificação livre?")
        
        all_tags_global = []
        for l in df_af['estilos_list']: all_tags_global.extend(l)
        
        if all_tags_global:
            vocab_counts = pd.Series(all_tags_global).value_counts().head(30).reset_index()
            vocab_counts.columns = ['Tag', 'Total']
            
            fig_v = px.treemap(vocab_counts, path=['Tag'], values='Total', title="Mapa de Tags Mais Frequentes (Treemap)")
            st.plotly_chart(fig_v, use_container_width=True)
            
            st.subheader("Tags Únicas Geradas")
            st.write(f"Total de tags únicas diferentes: **{len(set(all_tags_global))}**")
        else:
            st.warning("Sem tags para exibir.")

    with tab_tech:
        st.header("🎚️ Análise Técnica e Descritiva")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Densidade do Arranjo")
            if 'densidade' in df_af.columns:
                fig_d = px.pie(df_af, names='densidade', title="Distribuição de Densidade", hole=0.4)
                st.plotly_chart(fig_d, use_container_width=True)
                
        with col2:
            st.subheader("Presença Vocal")
            if 'presenca_vocal' in df_af.columns:
                fig_v = px.pie(df_af, names='presenca_vocal', title="Distribuição Vocal", hole=0.4)
                st.plotly_chart(fig_v, use_container_width=True)
                
        st.subheader("Instrumentos Mais Citados")
        if 'instrumentos' in df_af.columns:
            all_inst = []
            for i_str in df_af['instrumentos'].dropna():
                parts = [p.strip() for p in str(i_str).split(',')]
                all_inst.extend(parts)
            
            if all_inst:
                inst_counts = pd.Series(all_inst).value_counts().head(20).reset_index()
                inst_counts.columns = ['Instrumento', 'Contagem']
                st.plotly_chart(px.bar(inst_counts, x='Contagem', y='Instrumento', orientation='h'), use_container_width=True)

    with tab_data:
        st.dataframe(df_af, use_container_width=True)

    with tab_metod:
        st.markdown(METODOLOGIA_ABERTO)

elif page == "Músicas Reais (Gabarito)":
    st.title("🎸 Análise de Músicas Reais (Gabarito vs Modelo)")
    st.caption("Comparação entre a classificação do modelo e o gabarito oficial de músicas reais.")

    FILE_REAIS = "resultado_musicas_reais_duplo.csv"
    if not os.path.exists(FILE_REAIS):
        st.error(f"Arquivo `{FILE_REAIS}` não encontrado. Rode `musica3.py` primeiro.")
        st.stop()
    
    df_reais = pd.read_csv(FILE_REAIS)
    
    # KPIs da Pagina
    total_musicas = len(df_reais)
    # Comparar ignorando case e espaços
    df_reais['Real'] = df_reais['genero1'].fillna('').astype(str).str.lower().str.strip()
    df_reais['Predito'] = df_reais['fechado_estilo'].fillna('').astype(str).str.lower().str.strip()
    
    acertos_fechado = (df_reais['Real'] == df_reais['Predito']).sum()
    acc_fechado = acertos_fechado / total_musicas if total_musicas > 0 else 0
    confianca_media = df_reais['fechado_confianca'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎵 Músicas Analisadas", total_musicas)
    col2.metric("✅ Acurácia (Fechada)", f"{acc_fechado:.1%}")
    col3.metric("🧠 Confiança Média", f"{confianca_media:.1f}%")
    col4.metric("❌ Erros Totais", total_musicas - acertos_fechado)

    st.divider()
    
    tab_overview, tab_years, tab_open, tab_data, tab_metod_reais = st.tabs(["📊 Visão Geral", "📅 Por Ano de Lançamento", "🔓 Análise Aberta", "📋 Dados Brutos", "📖 Metodologia"])

    with tab_overview:
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.subheader("Matriz de Confusão (Prompt Fechado)")
            labels = sorted(list(set(df_reais['Real'].unique()) | set(df_reais['Predito'].dropna().unique())))
            if len(labels) > 0:
                cm = confusion_matrix(df_reais['Real'], df_reais['Predito'], labels=labels)
                fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale=SCALE_BLUE,
                                   labels=dict(x="Predito (Gemini)", y="Real (Gabarito)", color="Qtd"), aspect="auto")
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.warning("Sem dados suficientes para Matriz.")
            
        with col2:
            st.subheader("Precisão por Gênero")
            acc_by_genre = df_reais.groupby('Real').apply(lambda x: (x['Real'] == x['Predito']).mean()).reset_index(name='Acurácia')
            acc_by_genre['N'] = df_reais.groupby('Real').size().values
            acc_by_genre = acc_by_genre.sort_values('Acurácia', ascending=True)
            
            fig_bar = px.bar(acc_by_genre, x='Acurácia', y='Real', orientation='h', 
                             text=[f"{v:.1%} (N={n})" for v, n in zip(acc_by_genre['Acurácia'], acc_by_genre['N'])],
                             color='Acurácia', color_continuous_scale=SCALE_GREEN)
            fig_bar.update_layout(xaxis_range=[0, 1.1])
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab_years:
        st.header("📅 Análise Temporal: Acurácia por Ano")
        st.markdown("Verifique se o modelo tem desempenho melhor em músicas mais recentes ou antigas.")
        
        if 'ano' in df_reais.columns:
            df_reais['ano_int'] = pd.to_numeric(df_reais['ano'], errors='coerce')
            df_valid_years = df_reais.dropna(subset=['ano_int'])
            df_valid_years['acertou'] = (df_valid_years['Real'] == df_valid_years['Predito']).astype(int)
            
            acc_por_ano = df_valid_years.groupby('ano_int').agg(
                Acurácia=('acertou', 'mean'),
                Total=('acertou', 'count')
            ).reset_index()
            
            fig_line = px.scatter(acc_por_ano, x='ano_int', y='Acurácia', size='Total', 
                                  title="Acurácia por Ano de Lançamento (Tamanho da bolha = Qtd músicas)",
                                  trendline="lowess", trendline_color_override="red")
            fig_line.update_traces(mode='lines+markers')
            fig_line.update_layout(xaxis_title="Ano", yaxis_title="Acurácia Média", yaxis_tickformat=".0%")
            st.plotly_chart(fig_line, use_container_width=True)
            
            st.subheader("Tabela Detalhada por Ano")
            st.dataframe(acc_por_ano.style.format({'Acurácia': '{:.1%}'}), hide_index=True, use_container_width=True)
        else:
            st.error("Coluna 'ano' não encontrada no CSV.")

    with tab_open:
        st.header("🔓 Classificação Aberta vs Fechada")
        st.markdown("Comparação entre o que o modelo diz quando é **livre** vs quando é **restrito** ao gabarito.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Estilos Abertos Gerados")
            if 'aberto_estilo' in df_reais.columns:
                top_open = df_reais['aberto_estilo'].value_counts().head(10).reset_index()
                top_open.columns = ['Estilo Aberto', 'Contagem']
                st.dataframe(top_open, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("Subgêneros Mais Citados")
            if 'aberto_subgeneros' in df_reais.columns:
                all_subs = []
                for s in df_reais['aberto_subgeneros'].dropna():
                    parts = [p.strip() for p in str(s).split(',')]
                    all_subs.extend(parts)
                
                if all_subs:
                    top_subs = pd.Series(all_subs).value_counts().head(15).reset_index()
                    top_subs.columns = ['Subgênero', 'Contagem']
                    fig_subs = px.bar(top_subs, x='Contagem', y='Subgênero', orientation='h', color='Contagem')
                    fig_subs.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_subs, use_container_width=True)

        st.divider()
        st.subheader("Explorer: Real vs Aberto")
        cols_explore = ['arquivo', 'Real', 'Predito', 'aberto_estilo', 'aberto_subgeneros', 'aberto_justificativa']
        valid_cols = [c for c in cols_explore if c in df_reais.columns]
        st.dataframe(df_reais[valid_cols], height=500, use_container_width=True)

    with tab_data:
        st.dataframe(df_reais, use_container_width=True)

    with tab_metod_reais:
        st.markdown(METODOLOGIA_REAIS)

