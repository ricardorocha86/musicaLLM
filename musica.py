import os
import asyncio
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Literal
import pandas as pd

# Configuração mínima
client = genai.Client(api_key='kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')  # usa Client(api_key=...)

# Schema estruturado
# Schema estruturado com Chain of Thought
class AnaliseMusical(BaseModel):
    estilo: Literal['rock', 'pagode', 'samba', 'mpb', 'funk', 'gospel', 'sertanejo', 'axé', 'clássica', 'carimbo', 'forró', 'rap']
    justificativa: str
    elementos_auditivos: list[str]

EXT_MIME = {'.mp3': 'audio/mp3', '.webm': 'audio/webm', '.mp4': 'video/mp4'}

# Limitador de Concorrência Segura (Windows Proactor Friendly)
LIMIT_CONCURRENCY = 50

async def _classificar_audio_caminho(caminho, sem):
    async with sem:
        with open(caminho, 'rb') as f:
            audio_bytes = f.read()
        ext = os.path.splitext(caminho)[1].lower()
        mime = EXT_MIME.get(ext, 'audio/mp3')
        
        prompt = """
        Atue como um especialista musical. Ouça o áudio com atenção.
        1. Identifique os instrumentos principais, o ritmo e a "vibe" da música.
        2. Com base nisso, determine o estilo musical mais adequado dentre as opções permitidas.
        3. Forneça uma justificativa concisa explicando sua escolha.
        """
        
        tentativas = 5
        atraso = 1.0
        for i in range(tentativas):
            try:
                resp = await client.aio.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[prompt, types.Part.from_bytes(data=audio_bytes, mime_type=mime)],
                    config={"response_mime_type": "application/json", "response_schema": AnaliseMusical},
                )
                print(f"✅ {os.path.basename(caminho)} => {resp.parsed.estilo}")
                return {
                    'arquivo': os.path.basename(caminho),
                    'estilo_llm': resp.parsed.estilo,
                    'justificativa': resp.parsed.justificativa,
                    'elementos': ", ".join(resp.parsed.elementos_auditivos)
                }
            except Exception as e:
                if i == tentativas - 1:
                    print(f"❌ Falha final em {os.path.basename(caminho)}")
                    return None
                print(f"⚠️ Erro em {os.path.basename(caminho)} (tentativa {i+1}): {e}. Retentando...")
                await asyncio.sleep(min(atraso, 10.0))
                atraso = min(atraso * 2, 10.0)

async def analisar_audios_em_paralelo(pasta='musicas_suno', caminho_csv='classificacao_musicas.csv'):
    caminhos = []
    if not os.path.exists(pasta):
        print(f"Pasta não encontrada: {pasta}")
        return pd.DataFrame()
        
    for nome in os.listdir(pasta):
        ext = os.path.splitext(nome)[1].lower()
        if ext in EXT_MIME:
            caminhos.append(os.path.join(pasta, nome))
            
    total_arquivos = len(caminhos)
    print(f"Iniciando processamento de {total_arquivos} arquivos com concorrência de {LIMIT_CONCURRENCY}...")
    
    if total_arquivos == 0:
        return pd.DataFrame()
        
    sem = asyncio.Semaphore(LIMIT_CONCURRENCY)
    
    # Lista para acumular resultados processados
    resultados_processados = []
    
    # Processamento e Salvamento Progressivo
    # Criamos as tarefas mas aguardamos em lotes ou usamos as_completed para salvar incrementalmente
    tarefas = [_classificar_audio_caminho(c, sem) for c in caminhos]
    
    cont = 0
    # Processa conforme vão ficando prontos
    for tarefa_concluida in asyncio.as_completed(tarefas):
        try:
            resultado = await tarefa_concluida
            cont += 1
            if resultado:
                # Extrair estilo real
                nome_arquivo = resultado['arquivo']
                if '_' in nome_arquivo:
                    estilo_real = nome_arquivo.split('_')[0].strip().lower()
                else:
                    estilo_real = "desconhecido"
                
                resultado['estilo_real'] = estilo_real
                resultados_processados.append(resultado)
            
            print(f"Progresso: {cont}/{total_arquivos} ({(cont/total_arquivos):.1%})")

            # Salva checkpoint a cada 10 arquivos
            if cont % 10 == 0:
                df_parcial = pd.DataFrame(resultados_processados)
                # Reordenar colunas
                cols = ['arquivo', 'estilo_real', 'estilo_llm', 'justificativa', 'elementos']
                # Garante que as colunas existam
                for c in cols:
                    if c not in df_parcial.columns:
                        df_parcial[c] = ""
                
                df_parcial = df_parcial[cols]
                df_parcial.to_csv(caminho_csv, index=False, encoding='utf-8')
                print(f"💾 Checkpoint salvo com {len(df_parcial)} registros.")
                
        except Exception as e:
            print(f"Erro crítico ao processar tarefa: {e}")

    print("Processamento finalizado.")
    
    df_final = pd.DataFrame(resultados_processados)
    if not df_final.empty:
        cols = ['arquivo', 'estilo_real', 'estilo_llm', 'justificativa', 'elementos']
        # Fallback para colunas faltantes se houver
        for c in cols:
             if c not in df_final.columns:
                 df_final[c] = ""
        df_final = df_final[cols]
        df_final.to_csv(caminho_csv, index=False, encoding='utf-8')
        
    return df_final

def executar_classificacao(pasta='musicas_suno', caminho_csv='classificacao_musicas.csv'):
    return asyncio.run(analisar_audios_em_paralelo(pasta=pasta, caminho_csv=caminho_csv))

# Executa apenas para as músicas Suno (IA)
executar_classificacao(pasta='musicas_IA/musicas_suno', caminho_csv='classificacao_suno.csv')

