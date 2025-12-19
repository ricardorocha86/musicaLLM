import os
import asyncio
import time
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Literal, List
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("❌ Erro: Chave GEMINI_API_KEY não encontrada no arquivo .env")
    exit(1)

client = genai.Client(api_key=API_KEY)

# --- CONFIGURAÇÕES DE PREÇO (por 1 milhão de tokens) ---
PRECOS = {
    "gemini-2.0-flash": {"audio_in": 0.70, "text_in": 0.10, "out": 0.40},
    "gemini-2.5-flash": {"audio_in": 1.00, "text_in": 0.30, "out": 2.50},
    "gemini-3-flash-preview": {"audio_in": 1.00, "text_in": 0.50, "out": 3.00}
}

# --- CONFIGURAÇÕES DO EXPERIMENTO ---
LIMIT_CONCURRENCY = 50
TEMPERATURA = 0
MODELOS = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-3-flash-preview"]

# --- VOCABULÁRIO CONTROLADO ---
ESTILOS = Literal['rock', 'samba', 'mpb', 'funk', 'sertanejo', 'carimbo', 'forro', 'rap']
CLIMAS = Literal['alegre_festivo', 'melancolico', 'agressivo', 'nostalgico', 'romantico', 'calmo', 'tenso', 'ironico']
TEMAS = Literal['amor', 'traicao', 'festa', 'social', 'ostentacao', 'cotidiano', 'fe', 'superacao', 'instrumental']
PUBLICOS = Literal['infantil', 'jovem', 'adulto', 'familia', 'nicho']
VOZES = Literal['masculina', 'feminina', 'dueto', 'grupo_coro', 'instrumental_sem_voz']
ANDAMENTOS = Literal['muito_lento', 'lento', 'moderado', 'rapido', 'muito_rapido']
DENSIDADES = Literal['minimalista', 'equilibrada', 'densa_caotica']

# --- SCHEMA PYDANTIC ---
class AnaliseMusical(BaseModel):
    # 1. Classificação Principal
    estilo: ESTILOS = Field(..., description="Estilo musical principal identificado na faixa.")
    justificativa: str = Field(..., description="Explicação técnica concisa da razão da classificação do estilo musical.")
    confianca: float = Field(..., ge=0.0, le=100.0, description="Nível de certeza da classificação (0.0 a 100.0).")
    
    # 2. Análise Técnica
    instrumentos: List[str] = Field(..., min_items=1, description="Lista de instrumentos principais identificados (mínimo 3).")
    andamento_percebido: ANDAMENTOS = Field(..., description="Percepção da velocidade/BPM da música.")
    presenca_vocal: VOZES = Field(..., description="Tipo de presença vocal ou instrumental.")
    densidade_arranjo: DENSIDADES = Field(..., description="Nível de preenchimento sonoro e complexidade do arranjo.")
    
    # 3. Análise Semântica/Emocional
    clima: List[CLIMAS] = Field(..., max_items=3, description="Atmosfera emocional predominante (máx 3 tags).")
    temas: List[TEMAS] = Field(..., max_items=3, description="Temas líricos ou conceituais abordados (máx 3 tags).")
    
    # 4. Sociolinguística
    publico_alvo: PUBLICOS = Field(..., description="Público-alvo demográfico mais provável.")
    registro_linguistico: Literal['formal', 'informal', 'giria', 'regional', 'nao_se_aplica'] = Field(..., description="Nível de formalidade e registro da linguagem.")
    
  

# 2. Definição dos Prompts
PROMPT_BASICO = """
Você é um classificador musical automatizado.
Sua tarefa é ouvir o arquivo de áudio fornecido e preencher os metadados solicitados no schema JSON.

INSTRUÇÕES CRÍTICAS:
1. Analise o áudio focando na instrumentação, ritmo e voz.
2. Classifique o estilo estritamente dentro das opções permitidas.
3. Para o campo 'confianca', use uma escala percentual de 0.0 a 100.0 (ex: 95.5).
4. Seja objetivo e direto na justificativa.
"""

PROMPT_INTERMEDIARIO = """
Atue como um Musicólogo Especialista em gêneros brasileiros e globais.
Analise o áudio com rigor técnico para extrair características acústicas e sociolinguísticas.

--- GUIA DE CLASSIFICAÇÃO (TAXONOMIA) ---
Analise os seguintes critérios para decidir o estilo:

1. ROCK: Presença dominante de guitarras distorcidas, bateria forte em 4/4, baixo elétrico marcante. Energia intensa ou balada pesada.
2. SAMBA: Ritmo binário (2/4), síncope característica, percussão (surdo, tamborim, pandeiro), cavaquinho e violão. Cadência dançante e "brasileira".
3. MPB: Fusão de elementos. Harmonia sofisticada (violão complexo), foco na lírica/poesia, influências de Jazz ou Bossa Nova. Voz em destaque.
4. FUNK (BR): Batida repetitiva (loop de bateria eletrônica/beatbox), graves pesados (sub-bass), estética minimalista e percussiva.
5. SERTANEJO: Destaque para violão (acústico ou elétrico) e sanfona. Uso frequente de duetos vocais (terças). Temas de amor ou sofrência.
6. CARIMBÓ: Ritmo acelerado do norte, percussão de curimbó (tambor grande), metais (saxofone/trompete), maracas. Influência indígena/africana.
7. FORRÓ: Trio clássico (sanfona, zabumba, triângulo) ou variações eletrônicas. Ritmo baião, xote ou arrasta-pé. Dançante e regional.
8. RAP: Foco total no ritmo e na fala rítmica (flow). Beats eletrônicos ou samples (boom bap/trap). Registro linguístico urbano e crítico.

--- INSTRUÇÕES DE ANÁLISE ---
- Valide a 'densidade_arranjo': Diferencie músicas minimalistas (voz e violão) de arranjos densos.
- Escute o 'registro_linguistico': Gírias sugerem Funk/Rap; Formalidade sugere MPB clássica.
- Confiança: Dê uma nota de 0.0 a 100.0 baseada na clareza dos sinais (ex: 85.0).
"""

PROMPT_AVANCADO = """
Atue como um Musicólogo Especialista em gêneros brasileiros e globais.
Analise o áudio com rigor técnico para extrair características acústicas e sociolinguísticas.

--- GUIA DE CLASSIFICAÇÃO (TAXONOMIA) ---
Analise os seguintes critérios para decidir o estilo:

1. ROCK: Presença dominante de guitarras distorcidas, bateria forte em 4/4, baixo elétrico marcante. Energia intensa ou balada pesada.
2. SAMBA: Ritmo binário (2/4), síncope característica, percussão (surdo, tamborim, pandeiro), cavaquinho e violão. Cadência dançante e "brasileira".
3. MPB: Fusão de elementos. Harmonia sofisticada (violão complexo), foco na lírica/poesia, influências de Jazz ou Bossa Nova. Voz em destaque.
4. FUNK (BR): Batida repetitiva (loop de bateria eletrônica/beatbox), graves pesados (sub-bass), estética minimalista e percussiva.
5. SERTANEJO: Destaque para violão (acústico ou elétrico) e sanfona. Uso frequente de duetos vocais (terças). Temas de amor ou sofrência.
6. CARIMBÓ: Ritmo acelerado do norte, percussão de curimbó (tambor grande), metais (saxofone/trompete), maracas. Influência indígena/africana.
7. FORRÓ: Trio clássico (sanfona, zabumba, triângulo) ou variações eletrônicas. Ritmo baião, xote ou arrasta-pé. Dançante e regional.
8. RAP: Foco total no ritmo e na fala rítmica (flow). Beats eletrônicos ou samples (boom bap/trap). Registro linguístico urbano e crítico.

--- INSTRUÇÕES DE ANÁLISE ---
- Valide a 'densidade_arranjo': Diferencie músicas minimalistas (voz e violão) de arranjos densos.
- Escute o 'registro_linguistico': Gírias sugerem Funk/Rap; Formalidade sugere MPB clássica.
- Confiança: Dê uma nota de 0.0 a 100.0 baseada na clareza dos sinais (ex: 85.0).


--- EXEMPLOS DE REFERÊNCIA (10 de cada estilo) ---
Use estas faixas específicas como âncora para comparar a sonoridade e referencia do que define o estilo:

[[ ROCK ]]
1. Legião Urbana - "Tempo Perdido"
2. Rita Lee - "Ovelha Negra"
3. Sepultura - "Roots Bloody Roots" (Exemplo pesado)
4. Charlie Brown Jr. - "Zóio de Lula"
5. Pitty - "Admirável Chip Novo"
6. Titãs - "Epitáfio"
7. Barão Vermelho - "Pro Dia Nascer Feliz"
8. Raimundos - "Mulher de Fases"
9. Skank - "Vou Deixar"
10. Raul Seixas - "Maluco Beleza"

[[ SAMBA ]]
1. Zeca Pagodinho - "Deixa a Vida Me Levar"
2. Cartola - "O Mundo é um Moinho"
3. Beth Carvalho - "Vou Festejar"
4. Martinho da Vila - "Mulheres"
5. Adoniran Barbosa - "Trem das Onze"
6. Paulinho da Viola - "Foi um Rio que Passou em Minha Vida"
7. Alcione - "Não Deixe o Samba Morrer"
8. Jorge Aragão - "Malandro"
9. Fundo de Quintal - "O Show Tem Que Continuar"
10. Clara Nunes - "Conto de Areia"

[[ MPB ]]
1. Elis Regina & Tom Jobim - "Águas de Março"
2. Caetano Veloso - "Sozinho"
3. Gilberto Gil - "Aquele Abraço"
4. Chico Buarque - "A Banda"
5. Djavan - "Oceano"
6. Milton Nascimento - "Travessia"
7. Marisa Monte - "Ainda Bem"
8. Tim Maia - "Gostava Tanto de Você"
9. Gal Costa - "Baby"
10. Jorge Ben Jor - "País Tropical"

[[ FUNK ]]
1. Cidinho & Doca - "Rap da Felicidade"
2. Claudinho & Buchecha - "Só Love"
3. Bonde do Tigrão - "Cerol na Mão"
4. MC Marcinho - "Glamourosa"
5. Anitta - "Show das Poderosas"
6. Kevin o Chris - "Evoluiu"
7. Ludmilla - "Cheguei"
8. MC Kevinho - "Olha a Explosão"
9. Pedro Sampaio - "Dançarina"
10. DJ Marlboro - "Rap das Armas" (Instrumental beat)

[[ SERTANEJO ]]
1. Chitãozinho & Xororó - "Evidências"
2. Zezé Di Camargo & Luciano - "É o Amor"
3. Leandro & Leonardo - "Pense em Mim"
4. Marília Mendonça - "Infiel"
5. Jorge & Mateus - "Pode Chorar"
6. Gusttavo Lima - "Balada (Tchê Tcherere Tchê Tchê)"
7. Bruno & Marrone - "Dormi na Praça"
8. Luan Santana - "Meteoro"
9. Sérgio Reis - "O Menino da Porteira" (Raiz)
10. Michel Teló - "Ai Se Eu Te Pego"

[[ CARIMBÓ ]]
1. Pinduca - "Dona Mariana"
2. Dona Onete - "No Meio do Pitiú"
3. Mestre Cupijó - "Mingau de Açaí"
4. Lia Sophia - "Ai Menina"
5. Mestre Verequete - "Chama Verequete"
6. Banda Calypso - "Dançando Calypso" (Fusão Pop/Carimbó)
7. Nilson Chaves - "Sabor Açaí"
8. Mestre Damasceno - "O Boto Namorador"
9. Grupo de Carimbó O Uirapuru - "Carimbó do Macaco"
10. Pinduca - "Sinha Pureza"

[[ FORRÓ ]]
1. Luiz Gonzaga - "Asa Branca"
2. Dominguinhos - "Eu Só Quero um Xodó"
3. Falamansa - "Xote dos Milagres"
4. Wesley Safadão - "Camarote" (Eletrônico)
5. Mastruz com Leite - "Meu Vaqueiro, Meu Peão"
6. Calcinha Preta - "Você Não Vale Nada"
7. Aviões do Forró - "Chupa que é de Uva"
8. Elba Ramalho - "De Volta pro Aconchego"
9. Alceu Valença - "Anunciação"
10. Frank Aguiar - "Morango do Nordeste"

[[ RAP ]]
1. Racionais MC's - "Diário de um Detento"
2. Sabotage - "Um Bom Lugar"
3. Marcelo D2 - "Qual é?"
4. Emicida - "Levanta e Anda"
5. Criolo - "Não Existe Amor em SP"
6. Gabriel o Pensador - "Cachimbo da Paz"
7. Hungria Hip Hop - "Lembranças"
8. Planet Hemp - "Mantenha o Respeito"
9. Racionais MC's - "Negro Drama"
10. Black Alien - "Babylon by Gus"

Se o áudio soar como uma dessas faixas, classifique no respectivo estilo.
"""

PROMPTS = {
    "P_Basico": PROMPT_BASICO,
    "P_Intermediario": PROMPT_INTERMEDIARIO,
    "P_Avancado": PROMPT_AVANCADO
}

EXT_MIME = {'.mp3': 'audio/mp3', '.webm': 'audio/webm', '.mp4': 'video/mp4', '.wav': 'audio/wav', '.m4a': 'audio/mp4'}

async def _classificar_audio_caminho(caminho, sem, modelo, prompt_text, prompt_name):
    async with sem:
        try:
            with open(caminho, 'rb') as f:
                audio_bytes = f.read()
            ext = os.path.splitext(caminho)[1].lower()
            mime = EXT_MIME.get(ext, 'audio/mp3')
        except Exception as e:
            return {'arquivo': os.path.basename(caminho), 'status': 'erro_leitura', 'erro_msg': str(e)}

        tentativas_max = 5
        atraso = 1.0
        
        for i in range(tentativas_max):
            t_start = time.time()
            try:
                resp = await client.aio.models.generate_content(
                    model=modelo,
                    contents=[prompt_text, types.Part.from_bytes(data=audio_bytes, mime_type=mime)],
                    config=types.GenerateContentConfig(
                        temperature=TEMPERATURA,
                        response_mime_type="application/json", 
                        response_schema=AnaliseMusical
                    ),
                )
                t_end = time.time()
                tempo_execucao = t_end - t_start

                # Extrair Tokens e Custos
                tokens_prompt_audio = 0
                tokens_prompt_text = 0
                tokens_output_text = 0
                tokens_output_thoughts = 0
                tokens_total = 0
                
                if resp.usage_metadata:
                    tokens_total = resp.usage_metadata.total_token_count
                    tokens_output_text = resp.usage_metadata.candidates_token_count or 0
                    try:
                        if hasattr(resp.usage_metadata, 'thoughts_token_count') and resp.usage_metadata.thoughts_token_count is not None:
                             tokens_output_thoughts = resp.usage_metadata.thoughts_token_count
                    except: pass

                    if resp.usage_metadata.prompt_tokens_details:
                        for d in resp.usage_metadata.prompt_tokens_details:
                            if d.modality == 'AUDIO': tokens_prompt_audio = d.token_count
                            elif d.modality == 'TEXT': tokens_prompt_text = d.token_count

                preco_mod = PRECOS.get(modelo, {"audio_in": 0, "text_in": 0, "out": 0})
                custo_input = ((tokens_prompt_audio / 1e6) * preco_mod['audio_in']) + ((tokens_prompt_text / 1e6) * preco_mod['text_in'])
                custo_output = (tokens_output_text / 1e6) * preco_mod['out'] 
                custo_total = custo_input + custo_output
                
                tempo_fmt = f"{tempo_execucao:.2f}".replace('.', ',')
                print(f"✅ [{modelo}|{prompt_name}] {os.path.basename(caminho)} (${custo_total:.6f})")
                
                # Parse output
                res = resp.parsed
                
                return {
                    'arquivo': os.path.basename(caminho),
                    'modelo': modelo,
                    'prompt_id': prompt_name,
                    # Campos do Schema
                    'estilo_llm': res.estilo,
                    'confianca': res.confianca,
                    'justificativa': res.justificativa,
                    'instrumentos': ", ".join(res.instrumentos),
                    'andamento': res.andamento_percebido,
                    'presenca_vocal': res.presenca_vocal,
                    'densidade': res.densidade_arranjo,
                    'clima': ", ".join(res.clima),
                    'temas': ", ".join(res.temas),
                    'publico_alvo': res.publico_alvo,
                    'registro_linguistico': res.registro_linguistico,
                    # Metadados
                    'input_audio_tokens': tokens_prompt_audio,
                    'input_text_tokens': tokens_prompt_text,
                    'output_text_tokens': tokens_output_text,
                    'output_thoughts_tokens': tokens_output_thoughts,
                    'total_tokens': tokens_total,
                    'custo_input': f"{custo_input:.6f}",
                    'custo_output': f"{custo_output:.6f}",
                    'custo_total': f"{custo_total:.6f}",
                    'tentativas': i + 1,
                    'tempo_execucao': tempo_fmt,
                    'status': 'sucesso',
                    'erro_msg': ''
                }
                
            except Exception as e:
                if i == tentativas_max - 1:
                    print(f"❌ Falha final [{modelo}|{prompt_name}] em {os.path.basename(caminho)}: {e}")
                    return {
                        'arquivo': os.path.basename(caminho), 'modelo': modelo, 'prompt_id': prompt_name,
                        'status': 'falha_api', 'erro_msg': str(e),
                        'tentativas': i + 1, 'tempo_execucao': "0,00",
                        'custo_input': "0", 'custo_output': "0", 'custo_total': "0"
                    }
                await asyncio.sleep(atraso)
                atraso = min(atraso * 1.5, 15)

def salvar_parcial_csv(rows_antigos, rows_novos, todos_arquivos, caminho, modelo, prompt_name):
    mapa_dados = {r['arquivo']: r for r in rows_antigos}
    for r in rows_novos: mapa_dados[r['arquivo']] = r
        
    lista_final = []
    for f in todos_arquivos:
        estilo_real = f.split('_')[0].strip().lower() if '_' in f else 'desconhecido'
        dado = mapa_dados.get(f, {
            'arquivo': f, 'modelo': modelo, 'prompt_id': prompt_name, 'estilo_real': estilo_real,
            'status': 'pendente', 'tempo_execucao': "0,00", 'custo_total': "0"
        })
        dado['modelo'] = modelo
        dado['prompt_id'] = prompt_name
        dado['estilo_real'] = estilo_real
        lista_final.append(dado)
        
    df = pd.DataFrame(lista_final)
    # Definir colunas ordenadas (Atualizado para o novo Schema)
    cols = [
        'arquivo', 'estilo_real', 'estilo_llm', 'confianca', 
        'modelo', 'prompt_id', 
        'justificativa', 'instrumentos', 'andamento', 'presenca_vocal', 'densidade', 
        'clima', 'temas', 'publico_alvo', 'registro_linguistico',
        'input_audio_tokens', 'input_text_tokens', 'output_text_tokens', 'output_thoughts_tokens', 'total_tokens',
        'custo_input', 'custo_output', 'custo_total',
        'tentativas', 'tempo_execucao', 'status', 'erro_msg'
    ]
    
    # Preencher faltantes
    for c in cols:
        if c not in df.columns: df[c] = None
    
    df = df[cols]
    df.to_csv(caminho, index=False)

async def executar_configuracao(pasta_raiz, modelo, prompt_name, prompt_text, arquivo_csv_parcial):
    todos_arquivos = []
    mapa_caminhos = {}
    for root, _, files in os.walk(pasta_raiz):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXT_MIME:
                todos_arquivos.append(f)
                mapa_caminhos[f] = os.path.join(root, f)
    todos_arquivos = sorted(list(set(todos_arquivos)))
    
    concluidos = set()
    rows = []
    if os.path.exists(arquivo_csv_parcial):
        try:
            df_existente = pd.read_csv(arquivo_csv_parcial)
            if not df_existente.empty and 'status' in df_existente.columns:
                for _, row in df_existente.iterrows():
                    if row['status'] in ['sucesso', 'falha_api']: 
                        concluidos.add(row['arquivo'])
                    rows.append(row.to_dict())
        except: pass
    
    pendentes = [f for f in todos_arquivos if f not in concluidos]
    
    print(f"\n--- Config: {modelo} + {prompt_name} | Pendentes: {len(pendentes)} ---")
    if not pendentes: return pd.DataFrame(rows)

    sem = asyncio.Semaphore(LIMIT_CONCURRENCY)
    tarefas = [ _classificar_audio_caminho(mapa_caminhos[n], sem, modelo, prompt_text, prompt_name) for n in pendentes ]
    
    novos_resultados = []
    count = 0
    for task in asyncio.as_completed(tarefas):
        res = await task
        novos_resultados.append(res)
        count += 1
        if count % 10 == 0 or count == len(pendentes):
            salvar_parcial_csv(rows, novos_resultados, todos_arquivos, arquivo_csv_parcial, modelo, prompt_name)
            
    salvar_parcial_csv(rows, novos_resultados, todos_arquivos, arquivo_csv_parcial, modelo, prompt_name)
    return pd.read_csv(arquivo_csv_parcial)

async def main():
    pasta_alvo = 'musicas_IA'
    arquivo_excel_final = 'Experimento_Completo_Gemini.xlsx'
    
    configs = []
    for mod in MODELOS:
        for p_name, p_text in PROMPTS.items():
            configs.append((mod, p_name, p_text))
            
    print(f"🧪 Iniciando Experimento Fatorial: {len(configs)} Rodadas.")
    resultados_finais = {}
    
    for modelo, p_name, p_text in configs:
        csv_name = f"temp_result_{modelo}_{p_name}.csv"
        df_config = await executar_configuracao(pasta_alvo, modelo, p_name, p_text, csv_name)
        
        mod_pretty = "-".join([p.capitalize() for p in modelo.split("-")]).replace("-Preview", "")
        prompt_pretty = {"P_Basico": "PromptBásico", "P_Intermediario": "PromptIntermediário", "P_Avancado": "PromptAvançado"}.get(p_name, p_name)
        sheet_name = f"{mod_pretty}-{prompt_pretty}"[:31]
        
        resultados_finais[sheet_name] = df_config
        await asyncio.sleep(2)

    print("\n📦 Gerando Excel Consolidado...")
    with pd.ExcelWriter(arquivo_excel_final, engine='xlsxwriter') as writer:
        for sheet, df in resultados_finais.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
    print(f"🏆 Arquivo gerado: {arquivo_excel_final}")
    await asyncio.sleep(1.0)

if __name__ == "__main__":
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try: asyncio.run(main())
    except RuntimeError as e: 
        if str(e) != "Event loop is closed": raise
