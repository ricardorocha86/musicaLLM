# Metodologia dos Experimentos - Textos para exibição no Streamlit

METODOLOGIA_SUNO = """
## 📋 Relatório Técnico: Experimento de Classificação (Fechado)

### 1. Objetivo
Avaliar a capacidade de classificação de gêneros musicais brasileiros por modelos Gemini, 
utilizando **taxonomia controlada** (8 gêneros fixos) em músicas geradas por IA (Suno).

---

### 2. Configuração do Experimento

| Parâmetro | Valor |
|-----------|-------|
| **Pasta de Áudios** | `musicas_IA/` |
| **Formato de Saída** | `Experimento_Completo_Gemini.xlsx` |
| **Temperatura** | `0` (determinístico) |
| **Concorrência** | 50 requisições simultâneas |
| **Retentativas** | 5 (com backoff exponencial) |

#### Modelos Testados:
- `gemini-2.0-flash`
- `gemini-2.5-flash`
- `gemini-3-flash-preview`

---

### 3. Prompts Utilizados

#### 3.1 Prompt Básico (`P_Basico`)
```text
Você é um classificador musical automatizado.
Sua tarefa é ouvir o arquivo de áudio fornecido e preencher os metadados solicitados no schema JSON.

INSTRUÇÕES CRÍTICAS:
1. Analise o áudio focando na instrumentação, ritmo e voz.
2. Classifique o estilo estritamente dentro das opções permitidas.
3. Para o campo 'confianca', use uma escala percentual de 0.0 a 100.0 (ex: 95.5).
4. Seja objetivo e direto na justificativa.
```

#### 3.2 Prompt Intermediário (`P_Intermediario`)
```text
Atue como um Musicólogo Especialista em gêneros brasileiros e globais.
Analise o áudio com rigor técnico para extrair características acústicas e sociolinguísticas.

--- GUIA DE CLASSIFICAÇÃO (TAXONOMIA) ---
Analise os seguintes critérios para decidir o estilo:

1. ROCK: Presença dominante de guitarras distorcidas, bateria forte em 4/4, baixo elétrico marcante.
2. SAMBA: Ritmo binário (2/4), síncope característica, percussão (surdo, tamborim, pandeiro).
3. MPB: Fusão de elementos. Harmonia sofisticada (violão complexo), foco na lírica/poesia.
4. FUNK (BR): Batida repetitiva (loop de bateria eletrônica/beatbox), graves pesados (sub-bass).
5. SERTANEJO: Destaque para violão e sanfona. Uso frequente de duetos vocais (terças).
6. CARIMBÓ: Ritmo acelerado do norte, percussão de curimbó, metais.
7. FORRÓ: Trio clássico (sanfona, zabumba, triângulo). Ritmo baião, xote ou arrasta-pé.
8. RAP: Foco total no ritmo e na fala rítmica (flow). Beats eletrônicos ou samples.

--- INSTRUÇÕES ---
- Valide a 'densidade_arranjo': Diferencie músicas minimalistas de arranjos densos.
- Escute o 'registro_linguistico': Gírias sugerem Funk/Rap; Formalidade sugere MPB.
- Confiança: Dê uma nota de 0.0 a 100.0 baseada na clareza dos sinais.
```

#### 3.3 Prompt Avançado (`P_Avancado`)
O Prompt Avançado é idêntico ao Intermediário, mas adiciona **80 exemplos de referência** 
(10 músicas por gênero) para ancorar a classificação. Exemplos incluem:
- Rock: Legião Urbana, Sepultura, Pitty
- Samba: Zeca Pagodinho, Cartola, Beth Carvalho
- MPB: Elis Regina, Caetano Veloso, Djavan
- Funk: Anitta, MC Kevinho, Ludmilla
- Sertanejo: Chitãozinho & Xororó, Marília Mendonça
- Forró: Luiz Gonzaga, Wesley Safadão
- Carimbó: Pinduca, Dona Onete
- Rap: Racionais MC's, Emicida, Criolo

---

### 4. Schema Pydantic (Estrutura de Saída)

```python
ESTILOS = Literal['rock', 'samba', 'mpb', 'funk', 'sertanejo', 'carimbo', 'forro', 'rap']
CLIMAS = Literal['alegre_festivo', 'melancolico', 'agressivo', 'nostalgico', 'romantico', 'calmo', 'tenso', 'ironico']
TEMAS = Literal['amor', 'traicao', 'festa', 'social', 'ostentacao', 'cotidiano', 'fe', 'superacao', 'instrumental']
PUBLICOS = Literal['infantil', 'jovem', 'adulto', 'familia', 'nicho']
VOZES = Literal['masculina', 'feminina', 'dueto', 'grupo_coro', 'instrumental_sem_voz']
ANDAMENTOS = Literal['muito_lento', 'lento', 'moderado', 'rapido', 'muito_rapido']
DENSIDADES = Literal['minimalista', 'equilibrada', 'densa_caotica']

class AnaliseMusical(BaseModel):
    # 1. Classificação Principal
    estilo: ESTILOS = Field(..., description="Estilo musical principal identificado na faixa.")
    justificativa: str = Field(..., description="Explicação técnica concisa da razão da classificação.")
    confianca: float = Field(..., ge=0.0, le=100.0, description="Nível de certeza (0.0 a 100.0).")
    
    # 2. Análise Técnica
    instrumentos: List[str] = Field(..., min_items=1, description="Lista de instrumentos principais.")
    andamento_percebido: ANDAMENTOS = Field(..., description="Percepção da velocidade/BPM.")
    presenca_vocal: VOZES = Field(..., description="Tipo de presença vocal ou instrumental.")
    densidade_arranjo: DENSIDADES = Field(..., description="Nível de preenchimento sonoro.")
    
    # 3. Análise Semântica/Emocional
    clima: List[CLIMAS] = Field(..., max_items=3, description="Atmosfera emocional predominante.")
    temas: List[TEMAS] = Field(..., max_items=3, description="Temas líricos ou conceituais.")
    
    # 4. Sociolinguística
    publico_alvo: PUBLICOS = Field(..., description="Público-alvo demográfico.")
    registro_linguistico: Literal['formal', 'informal', 'giria', 'regional', 'nao_se_aplica']
```

**Explicação**: O modelo é forçado a classificar em exatamente um dos 8 estilos (`Literal`), 
garantindo compatibilidade com o gabarito. Campos adicionais capturam características técnicas.

---

### 5. Delineamento Fatorial
- **3 Modelos × 3 Prompts = 9 Configurações**
- Cada configuração processa todos os arquivos da pasta
- Resultado consolidado em Excel com uma aba por configuração
"""

METODOLOGIA_ABERTO = """
## 📋 Relatório Técnico: Experimento de Classificação (Aberto)

### 1. Objetivo
Avaliar a capacidade de classificação **livre** (sem taxonomia restrita) dos modelos Gemini, 
permitindo que identifiquem múltiplos gêneros e subgêneros para cada música.

---

### 2. Configuração do Experimento

| Parâmetro | Valor |
|-----------|-------|
| **Pasta de Áudios** | `musicas_IA/` |
| **Formato de Saída** | `Experimento_Completo_Gemini_Aberto.xlsx` |
| **Temperatura** | `0` (determinístico) |
| **Concorrência** | 250 requisições simultâneas |

#### Modelos Testados:
- `gemini-2.0-flash`
- `gemini-2.5-flash`
- `gemini-3-flash-preview`

---

### 3. Prompts Utilizados

#### 3.1 Prompt Básico (`P_Basico`)
```text
Você é um classificador musical automatizado.
Sua tarefa é ouvir o arquivo de áudio fornecido e preencher os metadados solicitados no schema JSON.

INSTRUÇÕES CRÍTICAS:
1. Analise o áudio focando na instrumentação, ritmo e voz.
2. Classifique os estilos musicais livremente. Identifique todos os gêneros e subgêneros que se aplicam à faixa.
3. Para o campo 'confianca', use uma escala percentual de 0.0 a 100.0 (ex: 95.5).
4. Seja objetivo e direto na justificativa.
```

#### 3.2 Prompt Intermediário (`P_Intermediario`)
```text
Atue como um Musicólogo Especialista em gêneros globais.
Analise o áudio com rigor técnico para extrair características acústicas e sociolinguísticas.

--- INSTRUÇÕES DE ANÁLISE ---
- Classifique os estilos musicais de forma aberta. Não se limite a uma lista pré-definida.
- Se a música for uma fusão, liste todos os gêneros contribuintes (ex: ["Jazz", "Samba", "Eletrônica"]).
- Valide a 'densidade_arranjo': Diferencie músicas minimalistas de arranjos densos.
- Escute o 'registro_linguistico': Analise o vocabulário e a entonação.
- Confiança: Dê uma nota de 0.0 a 100.0 baseada na clareza dos sinais.
```

#### 3.3 Prompt Avançado (`P_Avancado`)
```text
Atue como um Musicólogo Especialista em gêneros globais e antropologia musical.
Analise o áudio com rigor técnico para extrair características acústicas, culturais e sociolinguísticas.

--- INSTRUÇÕES DE ANÁLISE ---
- Classificação de Estilo Aberta: Identifique com precisão os estilos musicais. 
  Seja granulado se possível (ex: em vez de apenas "Rock", use "Post-Punk", "Indie Rock" se aplicável).
- Liste múltiplos gêneros se houver hibridismo ou influências claras (ex: ["Pagode Baiano", "Funk Carioca"]).
- Valide a 'densidade_arranjo' e aspectos de produção (mixagem, efeitos).
- Analise a sociolinguística e o contexto cultural sugerido pela faixa.
- Confiança: Dê uma nota de 0.0 a 100.0 baseada na clareza dos sinais.

Use seu vasto conhecimento musical para rotular corretamente a faixa sem restrições de taxonomia.
```

---

### 4. Schema Pydantic (Estrutura de Saída)

```python
class AnaliseMusical(BaseModel):
    # DIFERENÇA PRINCIPAL: estilos é uma LISTA de strings, não um Literal
    estilos: List[str] = Field(..., description="Lista de estilos musicais identificados. Começe pelo grande grupo e depois os subgêneros.")
    justificativa: str = Field(..., description="Explicação técnica concisa.")
    confianca: float = Field(..., ge=0.0, le=100.0, description="Nível de certeza (0.0 a 100.0).")
    
    # Análise Técnica (igual ao experimento fechado)
    instrumentos: List[str] = Field(..., min_items=1, description="Lista de instrumentos principais.")
    andamento_percebido: ANDAMENTOS
    presenca_vocal: VOZES
    densidade_arranjo: DENSIDADES
    
    # Análise Semântica/Emocional
    clima: List[CLIMAS] = Field(..., max_items=3)
    temas: List[TEMAS] = Field(..., max_items=3)
    
    # Sociolinguística
    publico_alvo: PUBLICOS
    registro_linguistico: Literal['formal', 'informal', 'giria', 'regional', 'nao_se_aplica']
```

**Diferença Chave**: O campo `estilos` agora é `List[str]` em vez de `Literal[...]`, permitindo:
- Múltiplos gêneros por música
- Subgêneros específicos (ex: "Bossa Nova" em vez de "MPB")
- Taxonomia livre definida pelo modelo

---

### 5. Métrica de Acurácia "Broad"
Como o modelo retorna uma lista, a acurácia é calculada verificando se o **estilo real** 
(extraído do nome do arquivo) está **contido** em algum item da lista predita.
"""

METODOLOGIA_REAIS = """
## 📋 Relatório Técnico: Experimento com Músicas Reais (Gabarito)

### 1. Objetivo
Avaliar a acurácia do modelo `gemini-3-flash-preview` em músicas reais brasileiras, 
comparando com um gabarito oficial. Utiliza **dupla classificação**: fechada e aberta.

---

### 2. Configuração do Experimento

| Parâmetro | Valor |
|-----------|-------|
| **Pasta de Áudios** | `musicas_reais/` |
| **Arquivo de Gabarito** | `gabarito_musicas_reais.csv` |
| **Formato de Saída** | `resultado_musicas_reais_duplo.csv` |
| **Modelo** | `gemini-3-flash-preview` |
| **Temperatura** | `0` (determinístico) |
| **Concorrência** | 50 requisições simultâneas |

#### Gêneros no Gabarito:
`pop`, `mpb`, `forró`, `sertanejo`, `funk`, `pagode`, `hip-hop`, `rock`

---

### 3. Prompts Utilizados

#### 3.1 Prompt Fechado (Classificação Restrita)
```text
Atue como um Musicólogo Especialista. 
Analise o áudio e classifique-o ESTRITAMENTE em um dos seguintes gêneros:
[pop, mpb, forró, sertanejo, funk, pagode, hip-hop, rock]

Instruções:
1. Ignore variações sutis, force a classificação no gênero macro mais adequado da lista.
2. Forneça uma justificativa técnica concisa.
3. Atribua uma confiança de 0 a 100.
```

#### 3.2 Prompt Aberto (Classificação Livre)
```text
Atue como um Musicólogo Especialista.
Analise o áudio e identifique o estilo musical livremente, da forma mais precisa possível.
Além do estilo principal, liste subgêneros pertinentes.

Instruções:
1. Seja preciso na taxonomia (ex: prefira 'Bossa Nova' a 'MPB' se for o caso).
2. Liste subgêneros que capturem as nuances da faixa.
3. Justifique tecnicamente.
```

---

### 4. Schemas Pydantic (Estrutura de Saída)

#### 4.1 Schema Fechado
```python
GENEROS_ACEITOS = Literal['pop', 'mpb', 'forró', 'sertanejo', 'funk', 'pagode', 'hip-hop', 'rock']

class AnaliseFechada(BaseModel):
    estilo: GENEROS_ACEITOS = Field(..., description="Estilo classificado estritamente na lista.")
    justificativa: str = Field(..., description="Razão técnica da escolha.")
    confianca: float = Field(..., ge=0.0, le=100.0, description="Nível de certeza (0-100).")
```

**Explicação**: O `Literal` força o modelo a responder apenas com um dos 8 gêneros do gabarito.

#### 4.2 Schema Aberto
```python
class AnaliseAberta(BaseModel):
    estilo: str = Field(..., description="Estilo musical principal (classificação livre).")
    subgeneros: List[str] = Field(..., description="Lista de subgêneros identificados.")
    justificativa: str = Field(..., description="Razão técnica da escolha.")
    confianca: float = Field(..., ge=0.0, le=100.0, description="Nível de certeza (0-100).")
```

**Explicação**: Permite taxonomia livre para ver o que o modelo escolheria sem restrições.

---

### 5. Fluxo de Processamento
Para cada música:
1. **Chamada 1 (Fechada)**: Classifica com restrição de gêneros
2. **Chamada 2 (Aberta)**: Classifica livremente com subgêneros

O resultado final cruza com o gabarito para calcular acurácia por gênero e por ano.
"""
