# Guia de Diagnóstico ARIMA

Este guia ajuda a identificar e resolver problemas quando modelos ARIMA não convergem.

## 🔍 Quando usar este guia

Use este guia se você está vendo erros como:
- "Nenhum modelo ARIMA convergiu"
- "Testando 48 combinações... AIC=inf"
- "Série é praticamente constante"
- "Série muito curta após remover NaN"

## 📊 Ferramentas de Diagnóstico

### 1. Diagnóstico Básico: `diagnostico_serie.py`

**O que faz:**
- Verifica propriedades básicas da série (tamanho, NaN, infinitos)
- Detecta séries constantes ou com baixa variabilidade
- Calcula autocorrelação básica
- Identifica problemas óbvios antes de tentar ARIMA

**Como usar:**

```python
from diagnostico_serie import diagnose_series
import pandas as pd

# Sua série temporal
serie = df['sua_variavel']

# Executa diagnóstico
is_ok = diagnose_series(serie, name="Minha Série")

if not is_ok:
    print("Série tem problemas básicos - corrija antes de continuar")
```

**Ou execute o script completo:**

```bash
python diagnostico_serie.py
```

Isso vai:
1. Criar dados de teste
2. Rodar pipeline completo
3. Construir IDCI-VIX
4. Diagnosticar série
5. Executar diagnóstico ARIMA completo

### 2. Diagnóstico ARIMA Completo: `diagnostico_arima.py`

**O que faz:**
- Testes de estacionariedade (ADF, KPSS)
- Análise ACF/PACF com interpretação automática
- Testa 9 modelos ARIMA específicos e captura erros detalhados
- Testa diferentes níveis de diferenciação
- Fornece recomendações acionáveis

**Como usar:**

```python
from diagnostico_arima import full_arima_diagnosis
import pandas as pd

# Sua série temporal
serie = df['sua_variavel']

# Executa diagnóstico completo
full_arima_diagnosis(serie, name="Minha Série")
```

**Interpretando os resultados:**

O script testa estes modelos na ordem:
1. `ARIMA(0,0,0)` - Baseline (apenas média)
2. `ARIMA(1,0,0)` - AR(1) simples
3. `ARIMA(0,0,1)` - MA(1) simples
4. `ARIMA(1,0,1)` - ARMA(1,1)
5. `ARIMA(0,1,0)` - Random walk
6. `ARIMA(1,1,0)` - Modelo com diferenciação
7. `ARIMA(0,1,1)` - Modelo com diferenciação
8. `ARIMA(1,1,1)` - Fallback padrão
9. `ARIMA(2,1,2)` - Modelo mais complexo

Se **NENHUM** converge → série tem problemas fundamentais
Se **POUCOS** convergem → série é desafiadora, use modelos que convergiram
Se **MAIORIA** converge → tudo OK, use o com menor AIC

## ⚠️ Problemas Comuns e Soluções

### Problema 1: "Série é praticamente constante (std=X.XXe-XX)"

**Causa:** Todos os valores da série são iguais ou quase iguais.

**Diagnóstico:**
```python
print(f"Desvio padrão: {serie.std()}")
print(f"Valores únicos: {serie.nunique()}")
print(f"Primeiros valores: {serie.head(10).tolist()}")
```

**Soluções:**
1. Verifique se a série de entrada tem variabilidade:
   ```python
   print(df.describe())
   ```

2. Se usar IDCI-VIX, verifique se as variáveis de entrada variam:
   ```python
   print(df_stationary.std())  # Deve ter std > 0 para todas
   ```

3. Revise o processo de normalização/escalonamento:
   ```python
   # Evite normalizar séries já normalizadas
   # Verifique se scale_to_index() está funcionando corretamente
   ```

### Problema 2: "Série muito curta (X observações)"

**Causa:** ARIMA precisa de dados suficientes para estimar parâmetros.

**Diagnóstico:**
```python
print(f"Tamanho da série: {len(serie)}")
print(f"Tamanho após remover NaN: {len(serie.dropna())}")
```

**Soluções:**
1. **Mínimo recomendado:** 50 observações
2. **Mínimo absoluto:** 30 observações
3. Se tem < 30:
   - Colete mais dados
   - Use modelos mais simples (Ridge, Lasso)
   - Reduza complexidade do modelo ARIMA (ex: apenas AR(1))

### Problema 3: "Nenhum modelo ARIMA convergiu (todos AIC=inf)"

**Causa:** Múltiplas possíveis:
- Série é ruído branco (sem autocorrelação)
- Problemas numéricos na série
- Dados de entrada com qualidade ruim

**Diagnóstico:**

Execute o diagnóstico completo:
```python
from diagnostico_arima import full_arima_diagnosis
full_arima_diagnosis(serie, name="Debug")
```

Analise:
1. **ACF/PACF:** Se todos os lags têm valores < 0.2 → ruído branco
2. **Testes de estacionariedade:** Se ambos (ADF e KPSS) falharem → problema de estacionariedade
3. **Teste de modelos específicos:** Se até ARIMA(0,0,0) falha → problema numérico grave

**Soluções:**

Se é ruído branco:
```python
# Ruído branco não pode ser previsto com ARIMA
# Use modelos alternativos:
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
```

Se tem problemas numéricos:
```python
# Verifique escala dos dados
print(f"Min: {serie.min()}, Max: {serie.max()}")
print(f"Média: {serie.mean()}, Std: {serie.std()}")

# Considere re-escalar
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
serie_scaled = pd.Series(
    scaler.fit_transform(serie.values.reshape(-1, 1)).flatten(),
    index=serie.index
)
```

### Problema 4: "Série contém valores infinitos"

**Causa:** Operações matemáticas produziram inf/-inf.

**Diagnóstico:**
```python
print(f"Infinitos: {np.isinf(serie).sum()}")
print(f"Onde: {serie[np.isinf(serie)]}")
```

**Solução:**
```python
# Remova infinitos
serie_clean = serie.replace([np.inf, -np.inf], np.nan).dropna()

# Ou investigue a causa raiz
# Exemplo: divisão por zero, log de números negativos, etc.
```

### Problema 5: "Autocorrelação muito baixa - série pode ser ruído branco"

**Causa:** Série não tem padrões temporais (cada valor é independente).

**Diagnóstico:**
```python
from diagnostico_arima import analyze_acf_pacf
analyze_acf_pacf(serie, name="Debug", lags=20)
```

**Interpretação:**
- ACF e PACF todos < 0.2 → Ruído branco
- ACF e PACF têm alguns valores > 0.3 → OK para ARIMA

**Solução se for ruído branco:**

ARIMA não é apropriado. Use:
1. **Modelos simples:**
   - Média histórica
   - Último valor observado
   - Mediana móvel

2. **Modelos com exógenas:**
   - SARIMAX com variáveis explicativas
   - Regressão (Ridge, Lasso)
   - Random Forest

3. **Revise o processo:**
   - Talvez a diferenciação removeu todo o sinal
   - Talvez as variáveis selecionadas não são preditivas

### Problema 6: "ARIMA.fit() got an unexpected keyword argument 'disp'"

**Causa:** Incompatibilidade de versão do statsmodels.

**Solução:**

Já foi corrigido no código! Atualize:
```bash
git pull origin claude/development-work-01PF6KP5jF7dfQW8SeefED9z
```

Se ainda ocorrer:
```bash
pip install --upgrade statsmodels
```

## 🎯 Fluxo de Diagnóstico Recomendado

```
1. Execute diagnóstico básico
   ↓
2. Série passou?
   Sim → Continue
   Não → Corrija problemas básicos (NaN, infinitos, constante)
   ↓
3. Execute diagnóstico ARIMA completo
   ↓
4. Quantos modelos convergiram?

   0 modelos → Série não é apropriada para ARIMA
               Use Ridge/Lasso/RandomForest

   1-4 modelos → Use um dos que convergiu
                 Considere ensemble com outros métodos

   5+ modelos → Tudo OK! Use o com menor AIC
```

## 📝 Exemplo Completo de Uso

```python
import pandas as pd
import numpy as np
from diagnostico_serie import diagnose_series
from diagnostico_arima import full_arima_diagnosis

# 1. Carregue seus dados
df = pd.read_csv('dados_vitoria.csv', index_col=0, parse_dates=True)

# 2. Execute pipeline
from pipeline import VitoriaForecastPipeline

pipeline = VitoriaForecastPipeline(max_vars=5, forecast_horizon=12)
pipeline.preprocess(df)
pipeline.select_variables()
idci = pipeline.build_index()

# 3. Diagnóstico básico
print("="*80)
print("DIAGNÓSTICO BÁSICO")
print("="*80)
is_ok = diagnose_series(idci, name="IDCI-VIX")

if not is_ok:
    print("\n⚠ Série tem problemas básicos!")
    print("Verifique:")
    print("  - Seus dados de entrada")
    print("  - O processo de construção do IDCI-VIX")
    print("  - A seleção de variáveis")
    exit(1)

# 4. Diagnóstico ARIMA completo
print("\n" + "="*80)
print("DIAGNÓSTICO ARIMA COMPLETO")
print("="*80)
full_arima_diagnosis(idci, name="IDCI-VIX")

# 5. Com base nos resultados, treine modelos apropriados
# Se ARIMA convergiu:
from forecasting.arima_models import ARIMAForecaster
model = ARIMAForecaster()
model.fit(idci, auto=True, verbose=True)

# Se ARIMA não convergiu, use alternativas:
from forecasting.regularized_models import RegularizedForecaster
model = RegularizedForecaster(method='ridge')
# ... etc
```

## 🔧 Configurações Avançadas

### Ajustar sensibilidade do auto_arima

No arquivo `src/forecasting/arima_models.py`:

```python
# Reduzir range de busca (mais rápido, menos abrangente)
order = model.auto_arima(
    serie,
    p_range=(0, 2),  # ao invés de (0, 3)
    d_range=(0, 1),  # ao invés de (0, 2)
    q_range=(0, 2),  # ao invés de (0, 3)
    verbose=True
)

# Usar BIC ao invés de AIC (penaliza mais a complexidade)
order = model.auto_arima(
    serie,
    criterion='bic',  # ao invés de 'aic'
    verbose=True
)
```

### Forçar ordem específica

Se você sabe qual ordem usar:

```python
from forecasting.arima_models import ARIMAForecaster

model = ARIMAForecaster()
model.fit(
    serie,
    order=(1, 1, 1),  # Força ARIMA(1,1,1)
    auto=False,        # Não usa auto_arima
    verbose=True
)
```

## 📞 Suporte

Se após seguir este guia você ainda tiver problemas:

1. Execute e salve a saída completa do diagnóstico:
   ```bash
   python diagnostico_arima.py > diagnostico_output.txt 2>&1
   ```

2. Compartilhe:
   - O arquivo `diagnostico_output.txt`
   - Descrição dos seus dados (fonte, frequência, período)
   - O que você está tentando prever

## 📚 Referências

- [Statsmodels ARIMA Documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [ADF Test](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test)
- [KPSS Test](https://en.wikipedia.org/wiki/KPSS_test)
- [ACF/PACF Interpretation](https://otexts.com/fpp2/non-seasonal-arima.html)
