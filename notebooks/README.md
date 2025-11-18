# 📊 Notebooks de Análise Geoespacial Imobiliária

Este diretório contém notebooks Jupyter com exemplos práticos de análise geoespacial e viabilidade de empreendimentos imobiliários, com design minimalista em preto e branco.

## 📚 Notebooks Disponíveis

### 1️⃣ `01_basico_carregamento_visualizacao.ipynb`
**Nível: Básico**

Introdução ao carregamento e visualização de dados imobiliários.

**Conteúdo:**
- Carregamento de dados em formato Parquet
- Estatísticas descritivas
- Gráficos de distribuição (área, CA, TO, altura)
- Análise por bairro
- Análise de preços e vendas
- Performance de incorporadores

**Ideal para:** Iniciantes que querem entender os dados disponíveis.

---

### 2️⃣ `02_analise_espacial_mapas.ipynb`
**Nível: Intermediário**

Análise espacial e criação de mapas interativos.

**Conteúdo:**
- Manipulação de dados geoespaciais com GeoPandas
- Mapas estáticos com matplotlib
- Mapas interativos com Folium
- Análise de proximidade e buffers
- Densidade espacial (KDE)
- Clustering espacial (DBSCAN)
- Mapas de calor

**Ideal para:** Análise de localização e distribuição espacial de empreendimentos.

---

### 3️⃣ `03_analise_mercado_graficos.ipynb`
**Nível: Intermediário**

Análise profunda do mercado imobiliário com gráficos avançados.

**Conteúdo:**
- Dashboard de indicadores (KPIs)
- Análise comparativa de preços por bairro
- Matriz de correlação
- Performance de vendas
- Segmentação por tipologia
- Análise de VGV (Valor Geral de Vendas)
- Relatório executivo

**Ideal para:** Análise de mercado e inteligência competitiva.

---

### 4️⃣ `04_avancado_analise_viabilidade.ipynb`
**Nível: Avançado**

Análise completa de viabilidade econômica de empreendimentos.

**Conteúdo:**
- Cálculo de potencial construtivo
- Otimização de mix de produtos
- Análise de viabilidade econômica (VGV, custos, margem, ROI)
- Análise de sensibilidade
- Machine Learning para predição de preços
- Dashboard comparativo de lotes
- Ranking de oportunidades

**Ideal para:** Análise de viabilidade e tomada de decisão de investimentos.

---

## 🚀 Começando

### Instalação

1. **Criar ambiente virtual (recomendado):**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

2. **Instalar dependências:**
```bash
cd notebooks
pip install -r requirements.txt
```

3. **Iniciar Jupyter:**
```bash
jupyter notebook
```

### Estrutura de Dados

Os notebooks esperam encontrar dados nos seguintes formatos:

#### Dados de Vitória (Lotes)
```
Colunas: codLote, logradouro, numero, bairro, sigla_trat, area_terreno,
         ca, to, limite_altura, afast_frontal, limite_embasamento,
         gabarito, altura, geometry, inscricaoImobiliaria, tipoConstrucao,
         numeroPavimentos, ocupacao
```

#### Dados de Imóveis
```
Colunas: Incorporador, Empreendimento, Bairro, Endereco, Cidade, Dormitorios,
         Metragem Privativa, Vagas, Preco Total, Status, Unidades Total,
         Unidades Vendidas, Estoque Atual
```

### Localização dos Arquivos

```
data/
├── raw/
│   ├── vitoria_lotes.parquet
│   └── imoveis.parquet
└── processed/
    └── (arquivos gerados pelos notebooks)
```

**Nota:** Se os arquivos não existirem, os notebooks criarão dados de exemplo automaticamente.

---

## 📊 Fluxo de Trabalho Recomendado

```
1. Notebook 01 → Entender os dados disponíveis
                  ↓
2. Notebook 02 → Análise espacial e distribuição geográfica
                  ↓
3. Notebook 03 → Análise de mercado e precificação
                  ↓
4. Notebook 04 → Análise de viabilidade e decisão
```

---

## 🎨 Design

Todos os notebooks seguem um **design minimalista em preto e branco**:
- Gráficos em escala de cinza
- Layout limpo e profissional
- Foco na informação
- Ideal para relatórios e apresentações

---

## 💡 Casos de Uso

### Para Incorporadoras
- Identificar oportunidades de terrenos
- Otimizar mix de produtos
- Análise de viabilidade de projetos
- Precificação de unidades

### Para Investidores
- Análise de mercado
- Identificação de regiões valorizadas
- Avaliação de ROI
- Comparação de oportunidades

### Para Corretoras
- Inteligência de mercado
- Análise de competidores
- Tendências de preços
- Performance de vendas

---

## 📦 Arquivos Gerados

Os notebooks geram os seguintes arquivos processados:

```
data/processed/
├── lotes_processados.parquet
├── imoveis_processados.parquet
├── lotes_com_clusters.geojson
├── lotes_com_analise_espacial.parquet
├── mapa_lotes_interativo.html
├── mapa_calor_lotes.html
├── resumo_mercado.csv
├── resumo_mercado.json
├── analise_viabilidade_lotes.csv
├── analise_viabilidade_lotes.parquet
└── relatorio_viabilidade.json
```

---

## 🔧 Personalização

### Parâmetros Urbanísticos
Ajuste no Notebook 04:
```python
# Alterar coeficientes
area_computavel = lote['area_terreno'] * lote['ca']
area_projecao = lote['area_terreno'] * lote['to']
```

### Custos de Construção
Ajuste no Notebook 04:
```python
custo_construcao_m2 = 4500  # Ajustar valor
preco_terreno_m2 = 3000     # Ajustar valor
```

### Mix de Produtos
Ajuste no Notebook 04:
```python
# Alterar distribuição
mix['1 dorm'] = int(area_disponivel * 0.10 / tipologias['1 dorm']['area'])
mix['2 dorm'] = int(area_disponivel * 0.40 / tipologias['2 dorm']['area'])
# ...
```

---

## 🤝 Contribuindo

Para adicionar novos notebooks ou melhorias:

1. Manter o padrão de design (preto e branco)
2. Documentar bem o código
3. Incluir exemplos práticos
4. Adicionar visualizações claras

---

## 📝 Notas

- **Performance:** Para grandes volumes de dados, considere usar `Dask` ou processar em lotes
- **Memória:** Os notebooks foram otimizados para datasets de até 100k registros
- **Mapas Interativos:** Arquivos HTML podem ser grandes (>5MB) para muitos pontos

---

## 🆘 Suporte

Para dúvidas ou problemas:

1. Verifique se todas as dependências foram instaladas
2. Confirme que os dados estão no formato correto
3. Execute as células em ordem sequencial

---

## 📄 Licença

Estes notebooks fazem parte do projeto Sistema DSGE-RL.

---

**Última atualização:** 2025-11-18
