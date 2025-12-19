# Experimento de Classificação Musical com Gemini 2.5 Flash

Este projeto implementa um pipeline de experimentação para avaliar a capacidade multimodal do modelo **Gemini 2.5 Flash** na classificação de estilos musicais a partir de arquivos de áudio brutos (gerados via Suno AI).

O sistema é dividido em duas partes principais: o motor de processamento assíncrono (`musica.py`) e a interface de análise de dados (`app_analise.py`).

## 📂 Estrutura de Arquivos

- **`musica.py`**: O "coração" do processamento. Script Python que gerencia a leitura de arquivos, comunicação com a API e persistência de dados.
- **`app_analise.py`**: Interface **Streamlit** para visualização de métricas, matriz de confusão e auditoria de erros.
- **`musicas_IA/musicas_suno/`**: Diretório de entrada contendo os arquivos de áudio (`.mp3`, `.wav`, etc).
- **`classificacao_suno.csv`**: Consolidação dos resultados (Gerado automaticamente).

---

## ⚙️ Detalhes da Execução e Paralelismo

A arquitetura do `musica.py` foi desenhada para maximizar o *throughput* (vazão de processamento) mantendo a estabilidade do sistema e respeitando os limites da API.

### 1. Concorrência Assíncrona (`asyncio`)
Ao invés de processar um arquivo por vez (sequencial), utilizamos programação assíncrona para manter múltiplas requisições "em voo" simultaneamente.

- **Task Spawning**: Uma tarefa (`asyncio.Task`) é criada para cada arquivo de áudio encontrado na pasta.
- **Semáforo Limitador (`asyncio.Semaphore`)**: Para evitar sobrecarga da API ou do sistema operacional (erro de *Too many open files*), implementamos um limite estrito de **50 execuções simultâneas**.
    - `LIMIT_CONCURRENCY = 50`
    - O semáforo garante que a 51ª tarefa só inicie quando uma das 50 anteriores for concluída.

### 2. Cadeia de Pensamento (Chain of Thought)
O modelo não é solicitado a dar apenas o "label" final. Utilizamos um **Schema Estruturado (Pydantic)** que força o modelo a raciocinar antes de classificar:
1.  **Análise**: Identificar instrumentos, ritmo e "vibe".
2.  **Justificativa**: Escrever o porquê da escolha.
3.  **Classificação**: Só então selecionar o estilo musical.
Este processo reduz "alucinações" e melhora a acurácia.

### 3. Tolerância a Falhas e Persistência
O script é robusto a falhas de rede ou interrupções:
- **Retry com Backoff Exponencial**: Se uma requisição falhar, o script tenta novamente até 5 vezes, aumentando o tempo de espera entre cada tentativa (1s, 2s, 4s...).
- **Salvamento Incremental (Checkpointing)**:
    - Utilizamos `asyncio.as_completed` para processar os resultados na medida em que ficam prontos (não-bloqueante).
    - A cada **10 arquivos processados**, o arquivo CSV é atualizado e salvo em disco. Isso permite parar e retomar o script sem perder todo o progresso.

---

## 📊 Dashboard de Análise

O `app_analise.py` consome o CSV gerado e oferece:

1.  **Visão Geral**:
    - Acurácia Global.
    - **Matriz de Confusão** (Plotly): Para visualizar onde o modelo confunde um estilo com outro (ex: confundir Samba com Pagode).
    - Gráfico de Acurácia por Estilo.
2.  **Auditoria de Erros**:
    - Lista filtrável de todos os erros cometidos.
    - Exibe o **Áudio Real** vs **Predito**.
    - Mostra a **Justificativa do Modelo** para entender o raciocínio por trás do erro.

## 🚀 Como Executar

1. **Instalação**:
   Certifique-se de ter as bibliotecas instaladas:
   ```bash
   pip install -r requirements.txt
   ```

2. **Processamento (Backend)**:
   ```bash
   python musica.py
   ```
   *O terminal exibirá uma barra de progresso e logs de cada arquivo processado.*

3. **Visualização (Frontend)**:
   ```bash
   streamlit run app_analise.py
   ```

## ⚠️ Problemas Conhecidos (Windows)

Ao finalizar a execução do script `musica.py` no Windows, você pode ver mensagens de erro no terminal como:
- `Fatal error on SSL transport`
- `RuntimeError: Event loop is closed`

**Isso é normal e inofensivo.**
Esses erros ocorrem porque o Windows fecha o loop de eventos assíncronos antes que todas as conexões seguras (SSL) do Google Gemini tenham terminado de limpar seus buffers internos. Como o script já salvou os dados (`💾 Checkpoint salvo`) e exibiu "Processamento concluído", **seus dados estão seguros** e o experimento não foi afetado. Pode ignorar essas mensagens.
