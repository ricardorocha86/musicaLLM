import os
import asyncio
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Literal
import pandas as pd

# Configuração mínima
client = genai.Client(api_key='AIzaSyB0XRRuJvU7ZAkETNGKyptLDQhpfhIyFfE')  # usa Client(api_key=...)

# Schema estruturado
class Estilo(BaseModel):
    estilo: Literal['rock', 'pagode', 'samba', 'mpb', 'funk', 'gospel', 'sertanejo', 'axé', 'clássica']

EXT_MIME = {'.mp3': 'audio/mp3', '.webm': 'audio/webm', '.mp4': 'video/mp4'}

async def _classificar_audio_caminho(caminho):
    # carrega bytes e pede somente o estilo
    with open(caminho, 'rb') as f:
        audio_bytes = f.read()
    ext = os.path.splitext(caminho)[1].lower()
    mime = EXT_MIME.get(ext, 'audio/mp3')
    prompt = "Classifique o estilo musical deste áudio. Responda apenas com o campo 'estilo'."
    tentativas = 5
    atraso = 1.0
    for i in range(tentativas):
        try:
            resp = await client.aio.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt, types.Part.from_bytes(data=audio_bytes, mime_type=mime)],
                config={"response_mime_type": "application/json", "response_schema": Estilo},
            )
            print(f"Inferência LLM: {os.path.basename(caminho)} => {resp.text}")  # mostra resposta bruta do LLM
            return os.path.basename(caminho), resp.parsed.estilo.strip()
        except Exception as e:
            if i == tentativas - 1:
                raise e
            await asyncio.sleep(min(atraso, 5.0))
            atraso = min(atraso * 2, 5.0)

async def analisar_audios_em_paralelo(pasta='musicas_suno', caminho_csv='classificacao_musicas.csv'):
    # coleta mp3s e processa em paralelo
    caminhos = []
    for nome in os.listdir(pasta):
        ext = os.path.splitext(nome)[1].lower()
        if ext in EXT_MIME:
            caminhos.append(os.path.join(pasta, nome))
    print(f"Iniciando processamento paralelo de {len(caminhos)} arquivo(s)...")  # início do paralelo
    if len(caminhos) == 0:
        print(f"Nenhum arquivo suportado encontrado em: {pasta}")
        return pd.DataFrame(columns=['arquivo', 'estilo_real', 'estilo_llm'])
    tarefas = [_classificar_audio_caminho(c) for c in caminhos]
    resultados = await asyncio.gather(*tarefas, return_exceptions=True)
    print("Processamento paralelo finalizado.")  # fim do paralelo
    saidas = []
    for r, caminho in zip(resultados, caminhos):
        if isinstance(r, Exception):
            print(f"Erro ao processar {os.path.basename(caminho)}: {repr(r)}")
            continue
        saidas.append(r)
    linhas = []
    for nome, estilo_llm in saidas:
        estilo_real = nome.split(' - ')[0].strip()
        linhas.append({
            'arquivo': nome,
            'estilo_real': estilo_real,
            'estilo_llm': estilo_llm,
        })
    colunas = ['arquivo', 'estilo_real', 'estilo_llm']
    df = pd.DataFrame(linhas, columns=colunas)
    if not df.empty:
        df.to_csv(caminho_csv, index=False, encoding='utf-8')  # salva CSV
    print(df)  # mostra resumo
    return df

def executar_classificacao(pasta='musicas_suno', caminho_csv='classificacao_musicas.csv'):
    # wrapper síncrono para rodar facilmente
    return asyncio.run(analisar_audios_em_paralelo(pasta=pasta, caminho_csv=caminho_csv))

executar_classificacao()
executar_classificacao(pasta='downloads_musicas', caminho_csv='classificacao_downloads.csv')

