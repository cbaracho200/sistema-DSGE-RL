# Sistema de Previsão para Mercado Imobiliário - Vitória/ES

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Sistema avançado de previsão econométrica para o mercado imobiliário de Vitória/ES, combinando modelos de séries temporais, machine learning e análise de regimes.

## 🎯 Objetivo

Desenvolver um **índice sintético (IDCI-VIX)** que capture as condições do mercado imobiliário de Vitória e gerar **previsões 12 meses à frente** usando ensemble de múltiplos modelos.

## 🏗️ Arquitetura

O sistema implementa um pipeline completo:

```
Dados Brutos
    ↓
┌─────────────────────────────────────┐
│ 1. PRÉ-PROCESSAMENTO                │
│  - Testes de estacionaridade (ADF)  │
│  - Diferenciação automática          │
│  - Padronização                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. SELEÇÃO DE VARIÁVEIS             │
│  - Fator preliminar (PCA)           │
│  - Teste de causalidade de Granger  │
│  - Seleção top-5 variáveis          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. CONSTRUÇÃO DO IDCI-VIX           │
│  - Modelo de fator dinâmico         │
│  - Filtro de Kalman                 │
│  - Escala 0-10 via CDF normal       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. MODELOS DE PREVISÃO (12M)        │
│  ├─ ARIMA/SARIMA/SARIMAX            │
│  ├─ Markov-Switching (regimes)      │
│  ├─ Ridge/Lasso (regularização)     │
│  ├─ Random Forest                   │
│  └─ Regressão Quantílica            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 5. ENSEMBLE                         │
│  - Combinação ponderada             │
│  - Intervalos de confiança          │
│  - Análise de cenários              │
└─────────────────────────────────────┘
```

## 📊 Metodologia

### 1. Pré-processamento

#### Testes de Estacionaridade
- **ADF** (Augmented Dickey-Fuller): H₀ = raiz unitária
- **KPSS** (opcional): H₀ = estacionaridade
- Critério conservador: ambos devem concordar

#### Diferenciação Automática
Escolhe ordem mínima `d` tal que:
```
Z_t = (1-L)^d X_t  é estacionária
```

Suporta:
- Diferenciação regular: `(1-L)`
- Diferenciação sazonal: `(1-L^12)`
- Composição: `(1-L)(1-L^12)`

### 2. Seleção de Variáveis

#### Teste de Causalidade de Granger

Para cada variável candidata `Z_k`, testa se `Z_k` Granger-causa o fator preliminar:

```
f_t = α + Σ φ_j f_{t-j} + Σ β_j Z_{k,t-j} + u_t
```

**H₀**: `β_1 = ... = β_p = 0` (NÃO Granger-causa)

Seleciona as **5 variáveis** com maior F-statistic.

### 3. Modelo de Fator Dinâmico

#### Equação de Medição
```
Z_t = λ · f_t + ε_t,    ε_t ~ N(0, R)
```

Onde:
- `Z_t`: vetor (5×1) de variáveis observadas
- `f_t`: fator latente (escalar)
- `λ`: loadings
- `R`: covariância diagonal

#### Equação de Estado (AR(2))
```
f_t = φ_1·f_{t-1} + φ_2·f_{t-2} + η_t,    η_t ~ N(0, σ²_η)
```

#### Estimação
- **Filtro de Kalman** para estado latente
- **Máxima Verossimilhança** para parâmetros `(λ, φ, σ_η, R)`
- **Suavizador RTS** para estimativas finais

#### Escala 0-10
```
u_t = (f_t - μ_f) / σ_f
q_t = Φ(u_t)           # CDF normal
IDCI-VIX_t = 10 · q_t
```

**Interpretação**:
- `0-3`: Mercado resfriado
- `3-5`: Estabilidade
- `5-7`: Aquecimento moderado
- `7-10`: Aquecimento forte

### 4. Modelos de Previsão

#### 4.1. ARIMA/SARIMA/SARIMAX

**ARIMA(p,d,q)**:
```
φ(L) y_t = θ(L) ε_t
```

**SARIMA(p,d,q)(P,D,Q,s)**:
```
φ(L) Φ(L^s) y_t = θ(L) Θ(L^s) ε_t
```

**SARIMAX**: adiciona variáveis exógenas `X_t`

Seleção automática de ordem via **AIC/BIC**.

#### 4.2. Markov-Switching

Captura regimes (expansão/contração):

```
y_t = μ_{s_t} + Σ φ_{s_t,j} y_{t-j} + ε_t,    ε_t ~ N(0, σ²_{s_t})
```

Onde `s_t ∈ {0, 1}` é o regime oculto com cadeia de Markov:
```
P(s_t = j | s_{t-1} = i) = p_{ij}
```

**Algoritmo de Hamilton** para filtragem de regimes.

#### 4.3. Ridge/Lasso

Regressão regularizada para seleção de lags:

**Ridge (L2)**:
```
min_β  Σ(y_t - β'x_t)² + λ Σβ²_j
```

**Lasso (L1)**:
```
min_β  Σ(y_t - β'x_t)² + λ Σ|β_j|
```

Features: lags de `y_t` e exógenas `{Z_k,t}`.

**Time-Series Cross-Validation** para selecionar `λ`.

#### 4.4. Random Forest

Modelo não-linear baseado em árvores:
- Bootstrap + seleção aleatória de features
- Features: lags, médias móveis, estatísticas rolling
- Previsão = média de 100+ árvores

#### 4.5. Regressão Quantílica

Estima quantis condicionais:
```
Q_y(τ | X) = β_τ' X
```

Para `τ ∈ {0.1, 0.5, 0.9}` → cenários pessimista/base/otimista.

**Função de perda assimétrica**:
```
ρ_τ(u) = u(τ - 𝟙_{u<0})
```

### 5. Ensemble

Combinação de previsões:

**Média Ponderada**:
```
ŷ_t = Σ w_m · ŷ_{m,t}
```

Pesos inversamente proporcionais ao RMSE:
```
w_m = (1/RMSE_m) / Σ(1/RMSE_k)
```

**Rolling-Origin Evaluation** para estimar erros.

## 📁 Estrutura do Projeto

```
sistema-DSGE-RL/
├── src/
│   ├── preprocessing/
│   │   ├── stationarity.py      # Testes ADF/KPSS, diferenciação
│   │   └── granger.py           # Causalidade de Granger
│   ├── factor_model/
│   │   └── dynamic_factor.py    # Filtro de Kalman, IDCI-VIX
│   ├── forecasting/
│   │   ├── arima_models.py      # ARIMA/SARIMA/SARIMAX
│   │   ├── markov_switching.py  # Modelos de regime
│   │   ├── regularized_models.py # Ridge/Lasso
│   │   ├── tree_models.py       # Random Forest
│   │   └── quantile_regression.py # Regressão quantílica
│   ├── evaluation/
│   │   └── ensemble.py          # Combinação e avaliação
│   └── pipeline.py              # Pipeline principal
├── notebooks/
│   └── exemplo_vitoria_forecast.ipynb
├── data/
│   ├── raw/                     # Dados originais
│   └── processed/               # Dados processados
├── tests/
├── config/
├── requirements.txt
└── README.md
```

## 🚀 Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/sistema-DSGE-RL.git
cd sistema-DSGE-RL

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instale dependências
pip install -r requirements.txt
```

## 💻 Uso Rápido

```python
import pandas as pd
from src.pipeline import VitoriaForecastPipeline

# Carrega dados (mensais, já deflacionados e em log)
df = pd.read_csv('data/raw/vitoria_dados.csv', index_col=0, parse_dates=True)

# Inicializa pipeline
pipeline = VitoriaForecastPipeline(
    max_vars=5,              # Top-5 variáveis
    forecast_horizon=12,     # 12 meses à frente
    ar_order=2,              # AR(2) para fator
    verbose=True
)

# Executa pipeline completo
results = pipeline.run_full_pipeline(
    df,
    models_to_train=['arima', 'ridge', 'lasso', 'random_forest', 'quantile'],
    ensemble_method='weighted_avg'
)

# Resultados
idci_vix = results['idci_vix']           # Índice histórico
forecasts = results['forecasts']         # Previsões por modelo
ensemble = results['ensemble']           # Previsão combinada
selected_vars = results['selected_vars'] # Variáveis selecionadas

print(f"IDCI-VIX atual: {idci_vix.iloc[-1]:.2f}")
print(f"Previsão 12M: {ensemble['forecast'].iloc[0]:.2f}")
```

## 📈 Exemplo de Saída

```
================================================================================
PASSO 1: PRÉ-PROCESSAMENTO - ESTACIONARIDADE
================================================================================
✓ 7 séries tornadas estacionárias
  Observações: 108

================================================================================
PASSO 2: SELEÇÃO DE VARIÁVEIS VIA GRANGER
================================================================================
Variância explicada pelo 1º componente: 68%
✓ 5 variáveis selecionadas:
  - credito_imob
  - lancamentos
  - preco_m2
  - emprego_construcao
  - pib_es

================================================================================
PASSO 3: CONSTRUÇÃO DO IDCI-VIX (FATOR DINÂMICO)
================================================================================
✓ IDCI-VIX construído:
  Média: 5.12
  Desvio: 2.34
  Min: 0.87, Max: 9.23

================================================================================
PASSO 4: TREINAMENTO DE MODELOS
================================================================================
✓ 5 modelos treinados com sucesso

================================================================================
PASSO 5: GERAÇÃO DE PREVISÕES
================================================================================
✓ Previsões geradas para 5 modelos

================================================================================
PASSO 6: ENSEMBLE (WEIGHTED_AVG)
================================================================================
✓ Ensemble criado com 5 modelos

################################################################################
# PIPELINE CONCLUÍDO
################################################################################

📊 Resultados:
  - Variáveis selecionadas: 5
  - IDCI-VIX atual: 6.45
  - Modelos treinados: 5
  - Previsão 12M (ensemble): 6.78
```

## 📊 Visualizações

O notebook inclui:
- Evolução histórica do IDCI-VIX
- Comparação de previsões por modelo
- Intervalos de confiança (regressão quantílica)
- Análise de regimes (Markov-switching)
- Cenários pessimista/base/otimista

## 🔬 Validação

**Rolling-Origin Cross-Validation**:
- Janela mínima de treino: 36 meses
- Horizontes: 1 a 12 meses
- Métricas: RMSE, MAE, MAPE

**Métricas por horizonte**:
```python
from src.evaluation.ensemble import ForecastEvaluator

evaluator = ForecastEvaluator(min_train_size=36, horizon=12)
metrics = evaluator.compute_metrics(eval_df, by_horizon=True)
```

## 🎓 Fundamentação Teórica

### Referências

1. **Testes de Estacionaridade**:
   - Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root.
   - Kwiatkowski, D., et al. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root.

2. **Causalidade de Granger**:
   - Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods.

3. **Modelos de Fator Dinâmico**:
   - Stock, J. H., & Watson, M. W. (2002). Forecasting using principal components from a large number of predictors.
   - Durbin, J., & Koopman, S. J. (2012). Time Series Analysis by State Space Methods.

4. **Markov-Switching**:
   - Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle.

5. **Regularização**:
   - Tibshirani, R. (1996). Regression shrinkage and selection via the lasso.
   - Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems.

6. **Regressão Quantílica**:
   - Koenker, R., & Bassett Jr, G. (1978). Regression quantiles.

7. **Ensemble**:
   - Timmermann, A. (2006). Forecast combinations.

## 🛠️ Desenvolvimento

### Testes

```bash
pytest tests/ -v --cov=src
```

### Contribuindo

1. Fork o projeto
2. Crie branch para feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push (`git push origin feature/nova-funcionalidade`)
5. Abra Pull Request

## 📝 Licença

MIT License - veja [LICENSE](LICENSE) para detalhes.

## 👥 Autores

Desenvolvido para análise do mercado imobiliário de Vitória/ES.

## 🔮 Próximos Passos

- [ ] Integração com modelos DSGE
- [ ] Interface com Reinforcement Learning
- [ ] Dashboard interativo (Streamlit/Dash)
- [ ] API REST para previsões
- [ ] Análise de viabilidade de empreendimentos
- [ ] Otimização de portfólio imobiliário

## 📧 Contato

Para dúvidas ou sugestões, abra uma issue no GitHub.

---

**Nota**: Este é um sistema de pesquisa. As previsões não constituem recomendação de investimento.
