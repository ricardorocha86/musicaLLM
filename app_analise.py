import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, accuracy_score
import os

st.set_page_config(page_title="Dashboard Análise Musical Gemini", layout="wide")

st.title("🎵 Análise de Classificação Musical com IA")

# Caminho do arquivo
CSV_FILE = 'classificacao_suno.csv'

# Carregar dados
if not os.path.exists(CSV_FILE):
    st.warning(f"Arquivo de resultados '{CSV_FILE}' não encontrado. Execute o script 'musica.py' primeiro.")
else:
    df = pd.read_csv(CSV_FILE)
    
    # Métricas Gerais
    total = len(df)
    acertos = df[df['estilo_real'] == df['estilo_llm']].shape[0]
    acuracia = acertos / total if total > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Arquivos", total)
    col1.metric("Acertos", acertos)
    col1.metric("Acurácia Global", f"{acuracia:.1%}")

    st.divider()

    # Abas
    tab1, tab2, tab3 = st.tabs(["📊 Visão Geral", "❌ Análise de Erros", "📋 Dados Brutos"])

    with tab1:
        st.subheader("Matriz de Confusão")
        
        # Ordem dos estilos para matriz consistente
        labels_unicos = sorted(list(set(df['estilo_real'].unique()) | set(df['estilo_llm'].unique())))
        
        cm = confusion_matrix(df['estilo_real'], df['estilo_llm'], labels=labels_unicos)
        
        fig = px.imshow(cm, 
                        text_auto=True, 
                        x=labels_unicos, 
                        y=labels_unicos,
                        labels=dict(x="Predito (LLM)", y="Real", color="Quantidade"),
                        color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Acurácia por Estilo")
        # Calcular acurácia por grupo
        acc_por_estilo = []
        for estilo in labels_unicos:
            subset = df[df['estilo_real'] == estilo]
            if len(subset) > 0:
                corretos = subset[subset['estilo_llm'] == estilo].shape[0]
                acc_por_estilo.append({'Estilo': estilo, 'Acurácia': corretos/len(subset), 'Total': len(subset)})
        
        df_acc = pd.DataFrame(acc_por_estilo)
        fig_bar = px.bar(df_acc, x='Estilo', y='Acurácia', color='Acurácia', hover_data=['Total'], range_y=[0,1])
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.subheader("Erros de Classificação")
        df_erros = df[df['estilo_real'] != df['estilo_llm']].copy()
        
        if df_erros.empty:
            st.success("Nenhum erro encontrado! O modelo acertou tudo.")
        else:
            estilo_filtro = st.multiselect("Filtrar por Estilo Real:", df_erros['estilo_real'].unique())
            
            if estilo_filtro:
                df_erros = df_erros[df_erros['estilo_real'].isin(estilo_filtro)]

            st.write(f"Mostrando {len(df_erros)} erros:")
            
            for index, row in df_erros.iterrows():
                with st.expander(f"{row['arquivo']} (Real: {row['estilo_real']} ➔ Predito: {row['estilo_llm']})"):
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        st.markdown(f"**Justificativa do Modelo:**\n> {row.get('justificativa', 'N/A')}")
                    with col_b:
                        st.markdown(f"**Elementos Detectados:**\n`{row.get('elementos', 'N/A')}`")
                        
                        # Tentar reproduzir áudio se existir
                        caminho_audio = os.path.join("musicas_IA", "musicas_suno", row['arquivo'])
                        if os.path.exists(caminho_audio):
                            st.audio(caminho_audio)
                        else:
                            st.warning("Arquivo de áudio não encontrado localmente.")

    with tab3:
        st.dataframe(df, use_container_width=True)
