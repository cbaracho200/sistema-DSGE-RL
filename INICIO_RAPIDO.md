# 🚀 Início Rápido - Sistema de Previsão Vitória/ES

Bem-vindo! Este guia vai te colocar em funcionamento em menos de 5 minutos.

---

## ⚡ Quick Start (30 segundos)

```bash
# 1. Clone o repositório (se ainda não fez)
git clone <url-do-repo>
cd sistema-DSGE-RL

# 2. Instale dependências
pip install -r requirements.txt

# 3. Execute o exemplo básico
python exemplo_basico.py
```

**Pronto!** Você já tem previsões 12 meses à frente do mercado imobiliário. 🎉

---

## 📚 Escolha Seu Caminho

Escolha o exemplo baseado no seu objetivo e tempo disponível:

### 🟢 Exemplo Básico (5 minutos)
**Para:** Iniciantes, análises rápidas, templates simples

```bash
python exemplo_basico.py
```

**O que faz:**
- ✓ Cria dados de exemplo
- ✓ Executa pipeline completo automático
- ✓ Treina 3 modelos (ARIMA, Ridge, RandomForest)
- ✓ Gera previsões 12 meses
- ✓ Salva resultados em CSV

**Arquivos gerados:**
- `previsoes_basico.csv`
- `idci_vix_historico.csv`

[📖 Ver documentação completa →](EXEMPLOS_README.md#-exemplo-básico)

---

### 🟡 Exemplo Intermediário (15 minutos)
**Para:** Customização, visualizações, apresentações

```bash
python exemplo_intermediario.py
```

**O que faz:**
- ✓ Tudo do básico, MAIS:
- ✓ Customização de parâmetros
- ✓ Diagnóstico de dados
- ✓ 3 visualizações profissionais (PNG)
- ✓ Análise individual de modelos
- ✓ Comparação de performance
- ✓ Sumário executivo

**Arquivos gerados:**
- 4 arquivos CSV (previsões, IDCI-VIX, Granger, sumário)
- 3 gráficos PNG (histórico, comparação, intervalos)

[📖 Ver documentação completa →](EXEMPLOS_README.md#-exemplo-intermediário)

---

### 🔴 Exemplo Avançado (30-45 minutos)
**Para:** Análise completa, validação rigorosa, papers

```bash
python exemplo_avancado.py
```

**O que faz:**
- ✓ Tudo do intermediário, MAIS:
- ✓ Diagnóstico ARIMA completo (ADF, KPSS, ACF, PACF)
- ✓ Cenários quantílicos (pessimista/base/otimista)
- ✓ Cross-validation temporal
- ✓ Ensemble customizado
- ✓ Análise de regimes (Markov-Switching)
- ✓ Relatório executivo completo (Markdown)
- ✓ 5 visualizações avançadas

**Arquivos gerados:**
- 7 arquivos CSV/TXT (resultados, previsões, cenários, etc.)
- 5 gráficos PNG de alta qualidade
- 1 relatório completo Markdown

[📖 Ver documentação completa →](EXEMPLOS_README.md#-exemplo-avançado)

---

## 🎯 Casos de Uso Rápidos

### Caso 1: "Preciso de previsões para reunião HOJE"
```bash
python exemplo_basico.py
# Use a tabela impressa no terminal ou previsoes_basico.csv
```

### Caso 2: "Vou apresentar para diretoria, preciso de GRÁFICOS"
```bash
python exemplo_intermediario.py
# Use os 3 gráficos PNG gerados
```

### Caso 3: "Estou escrevendo um PAPER científico"
```bash
python exemplo_avancado.py
# Use o relatorio_completo.md + todas as visualizações
```

---

## 🔧 Usar Seus Próprios Dados

### Passo 1: Prepare seus dados em CSV

Formato esperado:
```csv
data,preco_m2,vendas,lancamentos,credito,taxa_juros
2019-01-01,3000,100,50,5000,10.5
2019-02-01,3050,102,51,5100,10.3
...
```

**Requisitos:**
- ✓ Índice temporal (coluna 'data')
- ✓ Frequência mensal (recomendado)
- ✓ Mínimo 50 observações (ideal: 60+)
- ✓ Pelo menos 3 variáveis

### Passo 2: Edite o exemplo

Abra `exemplo_intermediario.py` (ou outro) e modifique:

```python
# Linha ~40: Comentar dados sintéticos
# df = pd.DataFrame({...})

# Descomentar e ajustar:
df = pd.read_csv('seus_dados.csv', index_col=0, parse_dates=True)
```

### Passo 3: Execute
```bash
python exemplo_intermediario.py
```

Pronto! O sistema vai processar seus dados automaticamente.

---

## 🆘 Ajuda e Solução de Problemas

### Erro comum: "Nenhum modelo ARIMA convergiu"

**Diagnóstico:**
```bash
python diagnostico_serie.py
```

**Ver guia completo:**
```bash
cat GUIA_DIAGNOSTICO_ARIMA.md
```

### Erro: "Série muito curta"
- **Causa:** Menos de 30 observações
- **Solução:** Colete mais dados ou use modelos mais simples

### Erro: "'str' object has no attribute 'shift'"
- **Causa:** Passou nome da variável ao invés do objeto
- **Solução:** Use `df['variavel']` ao invés de `'variavel'`

### Mais ajuda
- [EXEMPLOS_README.md](EXEMPLOS_README.md) - Guia completo de exemplos
- [GUIA_DIAGNOSTICO_ARIMA.md](GUIA_DIAGNOSTICO_ARIMA.md) - Troubleshooting ARIMA
- [ERROS_COMUNS.py](ERROS_COMUNS.py) - Erros comuns e soluções
- [README.md](README.md) - Documentação técnica completa

---

## 📖 Estrutura do Projeto

```
sistema-DSGE-RL/
│
├── 🚀 INÍCIO RÁPIDO
│   ├── INICIO_RAPIDO.md          ← Você está aqui!
│   ├── exemplo_basico.py          ← Comece aqui (5 min)
│   ├── exemplo_intermediario.py   ← Depois aqui (15 min)
│   ├── exemplo_avancado.py        ← Finalmente aqui (45 min)
│   └── EXEMPLOS_README.md         ← Guia completo
│
├── 📚 DOCUMENTAÇÃO
│   ├── README.md                  ← Documentação técnica
│   ├── GUIA_RAPIDO_VARIAVEIS.md   ← Como definir variáveis
│   ├── GUIA_DIAGNOSTICO_ARIMA.md  ← Troubleshooting ARIMA
│   └── ERROS_COMUNS.py            ← Erros e soluções
│
├── 🔧 FERRAMENTAS
│   ├── diagnostico_serie.py       ← Diagnóstico básico
│   └── diagnostico_arima.py       ← Diagnóstico ARIMA completo
│
├── 📦 CÓDIGO FONTE
│   └── src/
│       ├── pipeline.py            ← Pipeline principal
│       ├── preprocessing/         ← Estacionariedade, Granger
│       ├── factor_model/          ← Kalman, IDCI-VIX
│       ├── forecasting/           ← ARIMA, Ridge, RF, etc.
│       ├── evaluation/            ← Ensemble, métricas
│       └── utils/                 ← Visualizações
│
└── 📋 CONFIGURAÇÃO
    ├── requirements.txt           ← Dependências
    └── config/                    ← Arquivos de config
```

---

## 🎓 Progressão Recomendada

### Dia 1: Familiarização (1 hora)
1. Execute `exemplo_basico.py`
2. Leia a saída no terminal
3. Abra os CSVs gerados
4. Entenda o fluxo: dados → pipeline → modelos → previsões

### Dia 2: Customização (2 horas)
1. Execute `exemplo_intermediario.py`
2. Veja os gráficos gerados
3. Experimente alterar parâmetros
4. Tente com seus próprios dados

### Dia 3: Domínio (3 horas)
1. Execute `exemplo_avancado.py`
2. Leia o relatório Markdown gerado
3. Entenda a validação (CV)
4. Experimente pesos diferentes no ensemble

### Dia 4+: Aplicação
- Use o sistema com seus dados reais
- Customize modelos conforme necessário
- Automatize execução mensal
- Compartilhe resultados

---

## 💡 Dicas Importantes

### ✅ FAÇA:
- Comece sempre pelo exemplo básico
- Use `verbose=True` para ver o que acontece
- Execute diagnóstico se tiver problemas
- Salve suas configurações
- Documente suas escolhas

### ❌ NÃO FAÇA:
- Não pule direto para o avançado
- Não ignore warnings de diagnóstico
- Não use dados com muitos NaN (>20%)
- Não use menos de 50 observações
- Não confie cegamente nas previsões

---

## 📊 Fluxo de Decisão

```
Tenho dados?
│
├─ NÃO → Execute exemplo_basico.py
│         (usa dados sintéticos)
│
└─ SIM → Quantas observações?
         │
         ├─ < 50 → ⚠️  Poucos dados!
         │         Use modelos simples (Ridge, Lasso)
         │
         └─ ≥ 50 → Qual seu objetivo?
                   │
                   ├─ Análise rápida → exemplo_basico.py
                   │
                   ├─ Apresentação → exemplo_intermediario.py
                   │
                   └─ Análise rigorosa → exemplo_avancado.py
```

---

## 🔗 Links Úteis

- **Documentação Principal:** [README.md](README.md)
- **Guia de Exemplos:** [EXEMPLOS_README.md](EXEMPLOS_README.md)
- **Troubleshooting:** [GUIA_DIAGNOSTICO_ARIMA.md](GUIA_DIAGNOSTICO_ARIMA.md)
- **Definir Variáveis:** [GUIA_RAPIDO_VARIAVEIS.md](GUIA_RAPIDO_VARIAVEIS.md)
- **Erros Comuns:** [ERROS_COMUNS.py](ERROS_COMUNS.py)

---

## ⏱️ Quanto Tempo Preciso?

| Atividade | Tempo |
|-----------|-------|
| Instalação | 2 min |
| Exemplo básico | 5 min |
| Exemplo intermediário | 15 min |
| Exemplo avançado | 30-45 min |
| Adaptar para seus dados | 30 min |
| Customizar modelos | 1-2 horas |
| Análise completa | 2-4 horas |

---

## 🎯 Objetivos de Aprendizado

Após completar os 3 exemplos, você saberá:

✅ Como preparar dados para o sistema
✅ Como executar o pipeline completo
✅ Como interpretar IDCI-VIX
✅ Como comparar modelos diferentes
✅ Como diagnosticar problemas
✅ Como gerar visualizações profissionais
✅ Como criar relatórios executivos
✅ Como validar previsões (CV)
✅ Como customizar ensemble
✅ Como usar o sistema na prática

---

## 📞 Suporte

**Problemas?**
1. Verifique [GUIA_DIAGNOSTICO_ARIMA.md](GUIA_DIAGNOSTICO_ARIMA.md)
2. Veja [ERROS_COMUNS.py](ERROS_COMUNS.py)
3. Execute `python diagnostico_serie.py`
4. Abra uma issue no GitHub

**Dúvidas sobre exemplos?**
- Consulte [EXEMPLOS_README.md](EXEMPLOS_README.md)

**Dúvidas técnicas?**
- Consulte [README.md](README.md)

---

## 🚀 Comece Agora!

```bash
# Instalação
pip install -r requirements.txt

# Seu primeiro forecast em 5 minutos
python exemplo_basico.py

# 🎉 Pronto!
```

---

**Boa sorte com suas previsões! 📈**

*Sistema de Previsão para Mercado Imobiliário - Vitória/ES*
*Desenvolvido com Econometria, Machine Learning e boas práticas*
