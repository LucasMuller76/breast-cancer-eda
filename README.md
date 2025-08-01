# Analise Exploratoria de Dados - Cancer de Mama

Este projeto realiza uma analise exploratoria completa do dataset de cancer de mama, incluindo limpeza de dados, visualizacoes e insights importantes para diagnostico medico.

## Descricao

O dataset contem caracteristicas computacionais de imagens de massa mamaria, incluindo raio, textura, perimetro, area, suavidade, compacidade, concavidade, pontos concavos, simetria e dimensao fractal. Cada caracteristica e calculada para tres valores: media, erro padrao e pior valor.

## Objetivos

- Realizar analise exploratoria completa dos dados
- Identificar padroes e correlacoes importantes
- Gerar visualizacoes informativas
- Fornecer insights para diagnostico medico
- Preparar dados para modelagem de machine learning

## Estrutura do Dataset

O dataset contem as seguintes colunas:
- `diagnosis`: Diagnostico (M = Maligno, B = Benigno)
- `radius_mean`: Raio medio
- `texture_mean`: Textura media
- `perimeter_mean`: Perimetro medio
- `area_mean`: Area media
- `smoothness_mean`: Suavidade media
- `compactness_mean`: Compacidade media
- `concavity_mean`: Concavidade media
- `concave points_mean`: Pontos concavos medios
- `symmetry_mean`: Simetria media
- `fractal_dimension_mean`: Dimensao fractal media
- E mais caracteristicas com sufixos `_se` e `_worst`

## Como Executar

### Pre-requisitos

```bash
pip install -r requirements.txt
```

### Executar a Analise

```bash
python eda_breast_cancer.py
```

## Funcionalidades

### 1. Carregamento e Limpeza de Dados
- Carregamento do dataset CSV
- Remocao de colunas desnecessarias
- Conversao de diagnostico para numerico (M=1, B=0)
- Verificacao de dados ausentes e duplicados

### 2. Analise da Estrutura
- Informacoes basicas do dataset
- Estatisticas descritivas
- Distribuicao dos diagnosticos

### 3. Visualizacoes
- Grafico de barras da contagem de diagnosticos
- Histogramas de caracteristicas importantes
- Grafico de dispersao entre variaveis
- Boxplots comparativos
- Mapa de calor de correlacao

### 4. Analise por Agrupamentos
- Comparacao de medias por diagnostico
- Graficos comparativos entre grupos

### 5. Insights e Recomendacoes
- Principais descobertas
- Caracteristicas mais correlacionadas
- Recomendacoes para modelagem

## Arquivos Gerados

- `visualizacoes_gerais.png`: Visualizacoes principais
- `mapa_calor_correlacao.png`: Mapa de calor de correlacao
- `comparacao_diagnosticos.png`: Comparacao entre diagnosticos

## Principais Descobertas

1. **Distribuicao dos Diagnosticos**: Dataset bem balanceado entre classes
2. **Caracteristicas Mais Importantes**: Concavidade e pontos concavos sao muito correlacionados com diagnostico
3. **Diferencas Entre Grupos**: Tumores malignos apresentam valores maiores em todas as caracteristicas
4. **Qualidade dos Dados**: Dataset limpo, sem dados ausentes ou duplicados

## Tecnologias Utilizadas

- **Python 3.8+**
- **Pandas**: Manipulacao de dados
- **NumPy**: Computacao numerica
- **Matplotlib**: Visualizacoes basicas
- **Seaborn**: Visualizacoes estatisticas
- **Scikit-learn**: Pre-processamento

## Estrutura do Codigo

```
├── eda_breast_cancer.py    # Script principal
├── requirements.txt         # Dependencias
├── README.md              # Documentacao
└── Breast_cancer_dataset.csv  # Dataset
```

## Proximos Passos

1. **Modelagem de Machine Learning**
   - Implementar algoritmos de classificacao
   - Avaliar performance dos modelos
   - Otimizar hiperparametros

2. **Validacao Cruzada**
   - Implementar k-fold cross validation
   - Avaliar robustez dos modelos

3. **Feature Engineering**
   - Criar novas caracteristicas
   - Selecao de features
   - Reducao de dimensionalidade

## Resultados Esperados

- Analise completa dos dados
- Visualizacoes informativas
- Insights para diagnostico medico
- Base solida para modelagem

## Contribuicao

Contribuicoes sao bem-vindas! Por favor, abra uma issue ou pull request para melhorias.

## Licenca

Este projeto esta sob a licenca MIT. Veja o arquivo LICENSE para mais detalhes.

---

**Desenvolvido para analise de dados medicos** 