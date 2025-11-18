# Guia de Exemplos - Sistema de Previsão Vitória/ES

Este diretório contém 3 exemplos progressivos que demonstram como usar o sistema de previsão para o mercado imobiliário de Vitória/ES.

## 📚 Índice de Exemplos

| Nível | Arquivo | Tempo | Objetivo |
|-------|---------|-------|----------|
| 🟢 **Básico** | `exemplo_basico.py` | 5 min | Executar pipeline completo com mínima configuração |
| 🟡 **Intermediário** | `exemplo_intermediario.py` | 15 min | Customizar parâmetros, diagnosticar, visualizar |
| 🔴 **Avançado** | `exemplo_avancado.py` | 30-45 min | Análise completa com validação, cenários e relatório |

---

## 🟢 Exemplo Básico

**Arquivo:** `exemplo_basico.py`

### O que faz
- Cria dados sintéticos simples
- Executa pipeline com configuração padrão
- Treina modelos ARIMA, Ridge, RandomForest
- Gera previsões 12 meses
- Salva resultados em CSV

### Quando usar
- Você está começando e quer ver o sistema funcionar rapidamente
- Quer um template simples para adaptar aos seus dados
- Precisa de previsões rápidas sem muita customização

### Como executar
```bash
python exemplo_basico.py
```

### Arquivos gerados
- `previsoes_basico.csv` - Previsões de todos os modelos
- `idci_vix_historico.csv` - Série histórica do índice

### Conceitos cobertos
- ✓ Criação de dados de exemplo
- ✓ Pipeline básico
- ✓ Pré-processamento automático
- ✓ Seleção de variáveis (Granger)
- ✓ Construção IDCI-VIX
- ✓ Treinamento de modelos
- ✓ Previsões
- ✓ Interpretação simples

### Saída esperada
```
📊 Dados carregados:
   Período: 2019-01 a 2023-12
   Observações: 60
   Variáveis: 6

🔧 Criando pipeline...
   ✓ Pipeline criado

EXECUTANDO PIPELINE
1️⃣ Pré-processamento...
   ✓ Dados tornados estacionários

2️⃣ Seleção de variáveis (Granger)...
   ✓ 5 variáveis selecionadas

3️⃣ Construção do IDCI-VIX...
   ✓ Índice criado: 60 observações

4️⃣ Treinamento de modelos...
   ✓ Modelos treinados com sucesso!

5️⃣ Gerando previsões...
   ✓ Previsões geradas: 12 meses

📈 Previsões para os próximos 12 meses:
   [Tabela com previsões]

✅ EXEMPLO BÁSICO CONCLUÍDO!
```

---

## 🟡 Exemplo Intermediário

**Arquivo:** `exemplo_intermediario.py`

### O que faz
- Carrega dados de CSV ou cria dados realistas com tendência e sazonalidade
- Customiza parâmetros do pipeline
- Executa diagnóstico básico da série
- Analisa modelos individuais (parâmetros, métricas)
- Gera múltiplas visualizações
- Compara performance entre modelos
- Cria relatório textual

### Quando usar
- Você já entende o básico e quer customizar
- Precisa diagnosticar problemas nos seus dados
- Quer visualizações para apresentações
- Deseja comparar modelos diferentes
- Precisa ajustar parâmetros (lags, critérios, etc.)

### Como executar
```bash
python exemplo_intermediario.py
```

Para usar seus próprios dados, edite a linha 40:
```python
# Descomente e ajuste:
df = pd.read_csv('seus_dados.csv', index_col=0, parse_dates=True)
```

### Arquivos gerados

**Dados:**
- `previsoes_intermediario.csv` - Previsões de todos os modelos
- `idci_vix_intermediario.csv` - IDCI-VIX histórico
- `granger_results.csv` - Resultados do teste de Granger
- `sumario_intermediario.txt` - Sumário textual

**Visualizações:**
- `idci_vix_historico.png` - Série histórica com zonas interpretativas
- `comparacao_modelos.png` - Comparação de previsões
- `previsao_intervalos.png` - Previsão com intervalos de confiança

### Conceitos cobertos
- ✓ Tudo do básico, mais:
- ✓ Customização de parâmetros
- ✓ Carregamento de dados reais
- ✓ Diagnóstico de séries
- ✓ Análise de modelos individuais
- ✓ Visualizações profissionais
- ✓ Comparação de performance
- ✓ Análise de variação e tendências
- ✓ Recomendações baseadas em cenários

### Configurações customizadas
```python
CONFIG = {
    'max_vars': 4,              # Top-4 variáveis (ao invés de 5)
    'forecast_horizon': 12,     # 12 meses à frente
    'granger_maxlag': 6,        # Testa até 6 lags
    'min_train_size': 24,       # Mínimo 24 meses para treino
    'verbose': True
}
```

### Visualizações geradas

1. **IDCI-VIX Histórico**
   - Série temporal completa
   - Zonas de interpretação (Fraco/Moderado/Forte)
   - Cores e legendas profissionais

2. **Comparação de Modelos**
   - Todas as previsões sobrepostas
   - Histórico + 12 meses futuros
   - Legenda com todos os modelos

3. **Previsão com Intervalos**
   - Ensemble central
   - Intervalo de confiança
   - Interpretação de incerteza

---

## 🔴 Exemplo Avançado

**Arquivo:** `exemplo_avancado.py`

### O que faz
- Cria dados com regimes (expansão/contração)
- Executa diagnóstico ARIMA completo
- Compara IDCI-VIX automático vs target customizado
- Treina todos os modelos disponíveis
- Analisa resíduos e propriedades estatísticas
- Gera cenários quantílicos (pessimista/base/otimista)
- Executa cross-validation temporal
- Cria ensemble customizado com pesos otimizados
- Analisa regimes (Markov-Switching)
- Gera relatório completo em Markdown
- Cria 5+ visualizações avançadas

### Quando usar
- Você é usuário avançado ou pesquisador
- Precisa de análise completa e rigorosa
- Quer validação robusta com CV
- Precisa documentar metodologia
- Vai apresentar resultados para stakeholders
- Quer explorar todos os recursos do sistema

### Como executar
```bash
python exemplo_avancado.py
```

**Atenção:** Este exemplo pode levar 30-45 minutos dependendo do hardware.

### Arquivos gerados

**Dados e Resultados (7 arquivos):**
- `resultados_avancado.csv` - Todos os resultados (histórico + in-sample)
- `previsoes_avancado.csv` - Previsões de todos os modelos
- `ensemble_customizado.csv` - Ensemble com pesos otimizados
- `cenarios_quantilicos.csv` - Cenários pessimista/base/otimista
- `granger_results_avancado.csv` - Teste de Granger detalhado
- `transformacoes.txt` - Log de transformações aplicadas
- `relatorio_completo.md` - Relatório executivo completo

**Visualizações (5 arquivos):**
- `avancado_idci_vix.png` - IDCI-VIX com zonas
- `avancado_comparacao.png` - Todos os modelos
- `avancado_ensemble.png` - Ensemble com IC 90%
- `avancado_cenarios.png` - Cenários quantílicos
- `avancado_regimes.png` - Análise de regimes Markov

### Conceitos cobertos
- ✓ Tudo do básico e intermediário, mais:
- ✓ Diagnóstico ARIMA detalhado (ADF, KPSS, ACF, PACF)
- ✓ Dados com regimes econômicos
- ✓ Target customizado vs IDCI-VIX automático
- ✓ Análise de resíduos
- ✓ Feature importance (Random Forest)
- ✓ Probabilidades de regime (Markov-Switching)
- ✓ Cenários quantílicos (10%, 50%, 90%)
- ✓ Cross-validation temporal (rolling-origin)
- ✓ Métricas por horizonte (h=1, 3, 6, 12)
- ✓ Ensemble customizado com pesos
- ✓ Relatório executivo completo
- ✓ Visualizações para publicação

### Estrutura do Relatório Markdown

O arquivo `relatorio_completo.md` contém:

1. **Resumo Executivo**
   - Período, observações, horizonte
   - IDCI-VIX atual e previsto
   - Variação esperada

2. **Seleção de Variáveis**
   - Top-K com F-statistic e p-values

3. **Modelos Treinados**
   - Lista completa

4. **Previsões por Modelo**
   - Tabela comparativa 12 meses

5. **Análise de Cenários**
   - Pessimista, Base, Otimista

6. **Interpretação**
   - Cenário (otimista/negativo/estável)
   - Análise qualitativa

7. **Visualizações**
   - Todas as imagens incorporadas

8. **Notas Metodológicas**
   - Pré-processamento
   - Seleção de variáveis
   - Descrição dos modelos
   - Ensemble

9. **Disclaimer**

### Cross-Validation

O exemplo executa rolling-origin CV:
```
Janela inicial: 36 meses
Passo: 3 meses
Horizonte: 1 a 12 meses
```

Métricas calculadas:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

Por horizonte: h=1, 3, 6, 12 meses

### Ensemble Customizado

Pesos otimizados (exemplo):
```python
weights = {
    'ARIMA': 0.25,
    'Ridge': 0.20,
    'Lasso': 0.15,
    'RandomForest': 0.25,
    'QuantileRegression': 0.15
}
```

Ajustados automaticamente para modelos disponíveis e normalizados.

---

## 🔄 Progressão Recomendada

### 1️⃣ Comece pelo Básico
```bash
python exemplo_basico.py
```
- Entenda o fluxo geral
- Veja a saída esperada
- Familiarize-se com os conceitos

### 2️⃣ Experimente o Intermediário
```bash
python exemplo_intermediario.py
```
- Troque dados sintéticos pelos seus dados reais
- Ajuste parâmetros
- Analise visualizações
- Compare modelos

### 3️⃣ Aprofunde com o Avançado
```bash
python exemplo_avancado.py
```
- Execute análise completa
- Valide com CV
- Gere relatório profissional
- Explore todos os recursos

---

## 🎨 Customizando os Exemplos

### Usar seus próprios dados

**Formato esperado:**
```csv
data,var1,var2,var3,...
2019-01-01,100,50,3000,...
2019-02-01,102,51,3050,...
...
```

**Requisitos:**
- Índice temporal (mensal recomendado)
- Mínimo 50 observações (ideal: 60+)
- Pelo menos 3 variáveis
- Sem muitos NaN (<20%)

**Código:**
```python
# Carrega seus dados
df = pd.read_csv('seus_dados.csv', index_col=0, parse_dates=True)

# O resto do código permanece igual
pipeline = VitoriaForecastPipeline(...)
pipeline.preprocess(df)
# ...
```

### Alterar horizonte de previsão

```python
# Prever 6 meses ao invés de 12
pipeline = VitoriaForecastPipeline(
    forecast_horizon=6,  # Altere aqui
    # ... outros parâmetros
)
```

### Usar target customizado

```python
# Ao invés de IDCI-VIX automático
pipeline.preprocess(df)
pipeline.select_variables()

# Use uma variável específica
target = pipeline.df_stationary_['sua_variavel']

# Treina com target customizado
pipeline.train_models(target=target)
```

### Ajustar modelos

```python
# Exemplo: ARIMA com range diferente
from forecasting.arima_models import ARIMAForecaster

arima = ARIMAForecaster()
arima.fit(
    serie,
    auto=True,
    p_range=(0, 2),  # Ao invés de (0, 3)
    d_range=(0, 1),  # Ao invés de (0, 2)
    q_range=(0, 2),  # Ao invés de (0, 3)
    criterion='bic',  # Ao invés de 'aic'
    verbose=True
)
```

---

## ⚠️ Troubleshooting

### Erro: "Nenhum modelo ARIMA convergiu"

**Solução:**
```bash
# Execute o diagnóstico
python diagnostico_serie.py

# Veja o guia completo
cat GUIA_DIAGNOSTICO_ARIMA.md
```

### Erro: "Série muito curta"

**Causa:** Menos de 30 observações após pré-processamento

**Solução:**
- Colete mais dados
- Use `forecast_horizon` menor
- Use modelos mais simples (Ridge, Lasso)

### Erro: "'str' object has no attribute 'shift'"

**Causa:** Passou nome da variável ao invés do objeto Series

**Solução:**
```python
# ❌ Errado
target = 'preco_m2'

# ✓ Correto
target = df['preco_m2']
```

### Warning: "Série tem autocorrelação baixa"

**Causa:** Série pode ser ruído branco

**Solução:**
- ARIMA pode não ser apropriado
- Use Ridge, Lasso ou Random Forest
- Considere adicionar variáveis exógenas

---

## 📖 Documentação Adicional

- **[README.md](README.md)** - Documentação principal do sistema
- **[GUIA_RAPIDO_VARIAVEIS.md](GUIA_RAPIDO_VARIAVEIS.md)** - Como definir variáveis
- **[GUIA_DIAGNOSTICO_ARIMA.md](GUIA_DIAGNOSTICO_ARIMA.md)** - Diagnóstico de convergência
- **[ERROS_COMUNS.py](ERROS_COMUNS.py)** - Erros comuns e soluções

---

## 🆘 Suporte

Se tiver dúvidas:

1. Verifique se seguiu os passos do exemplo corretamente
2. Execute o diagnóstico: `python diagnostico_serie.py`
3. Consulte os guias de troubleshooting
4. Abra uma issue no GitHub com:
   - Qual exemplo está executando
   - Mensagem de erro completa
   - Características dos seus dados (tamanho, período, variáveis)

---

## 📊 Comparação dos Exemplos

| Característica | Básico | Intermediário | Avançado |
|----------------|--------|---------------|----------|
| **Tempo de execução** | ~2 min | ~5 min | ~30 min |
| **Linhas de código** | ~150 | ~300 | ~600 |
| **Dados** | Sintéticos simples | Realistas com sazonalidade | Com regimes |
| **Diagnóstico** | ❌ | ✓ Básico | ✓ Completo |
| **Modelos** | 3-4 | 5-6 | Todos |
| **Visualizações** | 0 | 3 | 5 |
| **Relatório** | Simples | Textual | Markdown completo |
| **CV** | ❌ | ❌ | ✓ |
| **Cenários** | ❌ | ❌ | ✓ |
| **Regimes** | ❌ | ❌ | ✓ |
| **Ensemble customizado** | ❌ | ❌ | ✓ |

---

## 🎯 Casos de Uso

### Caso 1: Análise Rápida
**Situação:** Preciso de previsões rápidas para reunião amanhã

**Solução:** Use `exemplo_basico.py`
- Adapte seus dados
- Execute
- Use a tabela de previsões

### Caso 2: Apresentação Executiva
**Situação:** Vou apresentar para diretoria, preciso de gráficos

**Solução:** Use `exemplo_intermediario.py`
- Gera 3 visualizações profissionais
- Sumário textual
- Comparação de modelos
- Recomendações

### Caso 3: Artigo Científico
**Situação:** Escrevendo paper, preciso de metodologia rigorosa

**Solução:** Use `exemplo_avancado.py`
- Cross-validation
- Métricas por horizonte
- Análise de regimes
- Relatório com metodologia completa
- Todas as visualizações

### Caso 4: Monitoramento Contínuo
**Situação:** Preciso atualizar previsões mensalmente

**Solução:** Adapte `exemplo_intermediario.py`
- Automatize carregamento de dados
- Configure para rodar via cron/scheduler
- Salve resultados com timestamp

---

## 💡 Dicas e Boas Práticas

### 1. Sempre comece com diagnóstico
```python
from diagnostico_serie import diagnose_series
is_ok = diagnose_series(sua_serie, name="Sua Variável")
```

### 2. Use verbose=True durante desenvolvimento
```python
pipeline = VitoriaForecastPipeline(verbose=True)
```

### 3. Salve configurações
```python
import json

config = {
    'max_vars': 5,
    'forecast_horizon': 12,
    # ...
}

with open('config.json', 'w') as f:
    json.dump(config, f)
```

### 4. Versionamento de resultados
```python
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M')
forecasts_df.to_csv(f'previsoes_{timestamp}.csv')
```

### 5. Compare com baseline simples
```python
# Baseline: última observação
baseline = idci.iloc[-1]

# Compare com modelos
print(f"Baseline (naive): {baseline:.2f}")
print(f"ARIMA: {forecasts_df['ARIMA'].iloc[-1]:.2f}")
print(f"Ensemble: {forecasts_df['Ensemble'].iloc[-1]:.2f}")
```

---

**Boa sorte com suas previsões! 🚀**
