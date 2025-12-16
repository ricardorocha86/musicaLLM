# 🎵 Experimento: Classificação de Estilos Musicais com IA

## 📋 Descrição do Experimento

Este experimento utiliza o **Gemini 2.5 Flash** para classificar automaticamente estilos musicais de arquivos de áudio. O objetivo é avaliar a capacidade do modelo de IA em identificar corretamente os gêneros musicais brasileiros.

---

## 🎯 Objetivo

Comparar a classificação de estilos musicais feita por uma IA (Gemini) com:
1. **Músicas geradas por IA** (principalmente do Suno AI)
2. **Músicas reais** de artistas brasileiros e internacionais

---

## 📁 Estrutura de Pastas

```
musicas_IA/
├── musicas_suno/          # 398 músicas geradas por IA (Suno, MusicGPT, etc.)
├── musicas_reais/         # 242 músicas de artistas reais
└── README.md              # Esta documentação
```

---

## 🔬 Metodologia

### 1. Preparação dos Dados

- **Músicas Suno**: Os arquivos possuem o estilo no nome do arquivo no formato `estilo_NomeDaMusica.mp3`
  - Exemplos: `rock_Sombras Eternas.mp3`, `funk_Rebola no Grave.mp3`, `samba_Roda da Vida.mp3`
  
- **Músicas Reais**: Arquivos MP3 de artistas brasileiros e internacionais, com nomes no formato `Artista - Nome da Música.mp3`

### 2. Estilos Musicais Mapeados

O modelo classifica os áudios em uma das seguintes categorias:

| Estilo | Descrição |
|--------|-----------|
| `rock` | Rock brasileiro e internacional |
| `pagode` | Pagode brasileiro |
| `samba` | Samba tradicional |
| `mpb` | Música Popular Brasileira |
| `funk` | Funk brasileiro/carioca |
| `gospel` | Música gospel/religiosa |
| `sertanejo` | Sertanejo e sertanejo universitário |
| `axé` | Axé music |
| `clássica` | Música clássica |

### 3. Processamento

O script `musica.py` executa as seguintes operações:

1. **Carrega** os arquivos de áudio da pasta especificada
2. **Envia** o áudio para o Gemini 2.5 Flash com um prompt de classificação
3. **Recebe** a resposta estruturada (schema Pydantic) com o estilo identificado
4. **Compara** o estilo identificado pelo LLM com o estilo real (extraído do nome do arquivo)
5. **Salva** os resultados em um arquivo CSV

### 4. Execução em Paralelo

O script utiliza **asyncio** para processar múltiplos arquivos simultaneamente, com:
- Retry automático (até 5 tentativas) em caso de falha
- Backoff exponencial entre tentativas
- Tratamento de erros individual por arquivo

---

## 🚀 Como Executar

### Pré-requisitos

```bash
pip install google-genai pydantic pandas
```

### Execução

```bash
python musica.py
```

### Saídas Geradas

| Arquivo | Descrição |
|---------|-----------|
| `classificacao_suno.csv` | Resultados das músicas geradas por IA |
| `classificacao_reais.csv` | Resultados das músicas reais |

### Estrutura do CSV

| Coluna | Descrição |
|--------|-----------|
| `arquivo` | Nome do arquivo de áudio |
| `estilo_real` | Estilo extraído do nome do arquivo |
| `estilo_llm` | Estilo classificado pelo Gemini |

---

## 📊 Métricas de Avaliação

Após a execução, você pode calcular:

- **Acurácia**: % de classificações corretas
- **Matriz de Confusão**: Para entender quais estilos são mais confundidos
- **Precisão por Estilo**: Performance do modelo em cada gênero musical

---

## 📝 Observações

### Músicas Suno (Geradas por IA)
- Total: **398 arquivos**
- Estilos presentes nos arquivos:
  - `carimbo` - Carimbó (46 arquivos)
  - `forro` - Forró (52 arquivos)
  - `funk` - Funk brasileiro (48 arquivos)
  - `funkmelody` - Funk melody (1 arquivo)
  - `hiphop` - Hip Hop (1 arquivo)
  - `mpb` - Música Popular Brasileira (50 arquivos)
  - `rap` - Rap nacional (52 arquivos)
  - `rock` - Rock (60 arquivos)
  - `rockleve` - Rock leve (2 arquivos)
  - `samba` / `samba2` - Samba (60 arquivos)
  - `sertanejo` - Sertanejo (46 arquivos)

### Músicas Reais
- Total: **242 arquivos**
- Variedade: Artistas brasileiros (sertanejo, funk, pagode, MPB, rock brasileiro) e internacionais (pop, rock)

---

## ⚠️ Limitações

1. **Estilos não mapeados**: Alguns arquivos de músicas reais podem ter estilos que não estão no schema (ex: forró, arrocha, piseiro)
2. **Nomes de arquivo**: A extração do estilo real depende do formato do nome do arquivo
3. **Rate Limiting**: O Google Gemini pode limitar requisições - o script possui retry automático

---

## ⚙️ Detalhes Técnicos da Execução em Paralelo

O script foi otimizado para lidar com altos volumes de requisições sem sobrecarregar a API do Google Gemini, utilizando uma arquitetura assíncrona robusta:

### 1. Controle de Concorrência (`Semaphore`)
Para evitar erros de "Too Many Requests" (HTTP 429), implementamos um **Semáforo** (`asyncio.Semaphore`).
- **Funcionamento**: O script cria centenas de tarefas (uma para cada arquivo), mas o semáforo atua como um porteiro, permitindo que apenas **15 requisições** sejam enviadas à API simultaneamente.
- **Benefício**: Garante um fluxo constante de processamento, aproveitando ao máximo a cota disponível sem atingir os limites agressivos de rejeição da API.

### 2. Tratamento de Falhas (`Retry`)
Requisições de rede podem falhar por instabilidade momentânea. O script implementa uma lógica de **tentativa e erro**:
- **Tentativas**: Cada arquivo tem direito a até **5 tentativas** de classificação.
- **Falha Parcial**: Se ocorrer um erro em um arquivo, isso não para o script. O erro é logado e o processamento continua para os outros arquivos.

### 3. Backoff Exponencial
Quando uma requisição falha, o script não tenta novamente imediatamente (o que poderia piorar o congestionamento). Ele espera um tempo progressivamente maior:
- **Estratégia**: Espera 1s, depois 2s, 4s, até o teto de 10s.
- **Resultado**: Dá tempo para a API "respirar" antes de receber nova carga.

---

## 📅 Data do Experimento

**16 de Dezembro de 2025**

---

## 👤 Autor

Projeto de experimentação com IA para classificação de áudio usando o modelo **Gemini 2.5 Flash**.
