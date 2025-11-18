# 📊 Notebooks - Sistema de Previsão de Mercado Imobiliário

Exemplos práticos de uso do **Sistema DSGE-RL** para previsões de mercado imobiliário de Vitória/ES.

## 🎯 Sobre o Sistema

Este sistema utiliza:
- **Modelos de Séries Temporais** (ARIMA, SARIMA, SARIMAX)
- **Machine Learning** (Ridge, Lasso, Random Forest)
- **Regressão Quantílica** para cenários
- **Ensemble Learning** para combinar previsões
- **Índice IDCI-VIX** (0-10) como indicador de confiança do mercado

---

## 📚 Notebooks Disponíveis

### 🚀 01_inicio_rapido.ipynb
**Nível: Básico** | **Tempo: ~10 min**

Introdução ao sistema de previsão.

**O que você aprende:**
- Carregar dados de mercado
- Executar pipeline de previsão completo
- Visualizar índice IDCI-VIX
- Interpretar previsões de 12 meses
- Gerar relatório executivo

**Ideal para:** Primeiro contato com o sistema

**Resultado:**
- IDCI-VIX histórico
- Previsões de 12 meses
- Variáveis mais importantes
- Recomendações automatizadas

---

### 📈 02_previsao_precos.ipynb
**Nível: Intermediário** | **Tempo: ~20 min**

Previsão detalhada de preços por m².

**O que você aprende:**
- Usar modelos específicos para preço
- Intervalos de confiança
- Análise de tendências
- Comparação com histórico
- Validação de previsões

**Ideal para:** Precificação e análise de tendências

---

### 🎲 03_analise_cenarios.ipynb
**Nível: Intermediário** | **Tempo: ~25 min**

Simulação de cenários econômicos.

**O que você aprende:**
- Simular diferentes cenários (otimista, base, pessimista)
- Análise de sensibilidade
- Quantis de previsão
- Stress testing
- Planejamento estratégico

**Ideal para:** Tomada de decisão sob incerteza

---

### ⚙️ 04_otimizacao_parametros.ipynb
**Nível: Avançado** | **Tempo: ~30 min**

Comparação e otimização de modelos.

**O que você aprende:**
- Avaliar performance de modelos
- Otimizar hiperparâmetros
- Validação cruzada temporal
- Seleção do melhor modelo
- Customizar ensemble

**Ideal para:** Maximizar acurácia das previsões

---

## 🚀 Instalação

### 1. Clonar repositório
```bash
git clone https://github.com/cbaracho200/sistema-DSGE-RL.git
cd sistema-DSGE-RL
```

### 2. Criar ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instalar dependências
```bash
pip install -r requirements.txt
```

### 4. Iniciar Jupyter
```bash
cd notebooks/examples
jupyter notebook
```

---

## 📊 Estrutura de Dados

O sistema espera dados em formato **CSV** ou **Parquet** com:

### Colunas Esperadas
- `index`: Data (formato YYYY-MM-DD, frequência mensal)
- `preco_m2`: Preço médio por m² (númer)
- `lancamentos`: Número de lançamentos
- `credito_imob`: Volume de crédito imobiliário
- `emprego_construcao`: Emppregos na construção
- `massa_salarial`: Massa salarial total
- `pib_es`: PIB do Espírito Santo
- `selic`: Taxa Selic
- *Outras variáveis relevantes*

### Exemplo de Formato
```csv
data,preco_m2,lancamentos,credito_imob,emprego,selic
2020-01-01,5200,120,1500000,45000,4.5
2020-02-01,5250,115,1520000,45200,4.25
...
```

### Carregar Dados
```python
# CSV
df = pd.read_csv('../data/raw/dados_mercado.csv',
                 index_col='data', parse_dates=True)

# Parquet
df = pd.read_parquet('../data/raw/dados_mercado.parquet')
```

---

## 🎨 Design dos Gráficos

Todos os notebooks usam design **minimalista em preto e branco**:
- Escalas de cinza
- Layout limpo
- Foco na informação
- Ideal para relatórios profissionais

---

## 💡 Fluxo de Trabalho Recomendado

```
1. Notebook 01 → Entender o sistema e IDCI-VIX
                  ↓
2. Notebook 02 → Prever preços específicos
                  ↓
3. Notebook 03 → Simular cenários alternativos
                  ↓
4. Notebook 04 → Otimizar para máxima acurácia
```

---

## 📦 Resultados Gerados

Os notebooks geram arquivos em `data/processed/`:

```
data/processed/
├── idci_vix.csv                    # Índice histórico
├── previsao_ensemble_12m.csv       # Previsão combinada
├── previsoes_todos_modelos.csv     # Todas as previsões
├── cenarios_quantis.csv            # Cenários (otim/base/pess)
└── metricas_modelos.csv            # Performance de cada modelo
```

---

## 🔧 Configuração do Pipeline

### Parâmetros Principais

```python
pipeline = VitoriaForecastPipeline(
    max_vars=5,              # Variáveis a selecionar (3-7)
    forecast_horizon=12,     # Meses à frente (6-24)
    ar_order=2,              # Ordem AR (1-4)
    verbose=True             # Mostrar progresso
)
```

### Modelos Disponíveis

- `arima`: ARIMA clássico
- `sarima`: SARIMA com sazonalidade
- `sarimax`: SARIMAX com variáveis exógenas
- `markov`: Markov-Switching
- `ridge`: Ridge Regression
- `lasso`: Lasso Regression
- `random_forest`: Random Forest
- `quantile`: Regressão Quantílica

### Métodos de Ensemble

- `simple_avg`: Média simples
- `weighted_avg`: Média ponderada (padrão)
- `median`: Mediana

---

## 📈 Casos de Uso

### Para Incorporadoras
- Decidir timing de lançamentos
- Definir estratégia de precificação
- Planejar investimentos
- Avaliar risco de projetos

### Para Investidores
- Timing de entrada/saída
- Alocação de capital
- Gestão de risco
- Due diligence

### Para Analistas
- Relatórios de mercado
- Inteligência competitiva
- Benchmarking
- Forecast mensal

---

## ⚠️ Notas Importantes

### Performance
- Recomendado: Mínimo 60 observações mensais
- Ideal: 100+ observações
- Atualização: Mensal

### Validação
- Compare previsões com valores realizados
- Ajuste parâmetros conforme necessário
- Monitore erro de previsão (MAPE, RMSE)

### Limitações
- Não captura eventos extremos (cisnes negros)
- Baseado em padrões históricos
- Requer atualização regular dos dados

---

## 🤝 Contribuindo

Para melhorar os notebooks:

1. Manter design minimalista (preto e branco)
2. Documentar código claramente
3. Incluir exemplos práticos
4. Testar com dados reais

---

## 📄 Licença

Sistema DSGE-RL - Vitória/ES Forecast

---

## 🆘 Suporte

**Problemas comuns:**

1. **Erro ao importar módulos**
   ```bash
   # Certifique-se de estar no ambiente virtual
   pip install -r requirements.txt
   ```

2. **Dados incompatíveis**
   - Verifique formato de datas
   - Certifique-se de índice temporal
   - Remova valores faltantes críticos

3. **Modelos não convergem**
   - Verifique estacionaridade
   - Reduza max_vars
   - Aumente período histórico

---

**Última atualização:** 2025-11-18

**Versão:** 1.0
