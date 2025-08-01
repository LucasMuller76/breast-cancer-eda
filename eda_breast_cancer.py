#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise Exploratória de Dados - Dataset de Câncer de Mama
==========================================================

Este script realiza uma análise exploratória completa do dataset de câncer de mama,
incluindo limpeza de dados, visualizações e insights importantes.

Autor: Análise de Dados
Data: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings

# Configurações para melhor visualização
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configurar matplotlib para português
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

def carregar_dados():
    """
    Carrega o dataset de câncer de mama
    """
    print("=" * 60)
    print("CARREGAMENTO DOS DADOS")
    print("=" * 60)
    
    try:
        # Carregar o dataset
        df = pd.read_csv('Breast_cancer_dataset.csv')
        print(f"Dataset carregado com sucesso!")
        print(f"Dimensoes: {df.shape[0]} linhas e {df.shape[1]} colunas")
        return df
    except FileNotFoundError:
        print("Erro: Arquivo 'Breast_cancer_dataset.csv' não encontrado!")
        return None

def limpar_dados(df):
    """
    Limpa e prepara os dados para análise
    """
    print("\n" + "=" * 60)
    print("LIMPEZA E PREPARACAO DOS DADOS")
    print("=" * 60)
    
    # Fazer uma cópia para não modificar o original
    df_clean = df.copy()
    
    # Remover coluna 'id' se existir
    if 'id' in df_clean.columns:
        df_clean = df_clean.drop('id', axis=1)
        print(" Coluna 'id' removida")
    
    # Converter diagnóstico para numérico (M=1, B=0)
    le = LabelEncoder()
    df_clean['diagnosis'] = le.fit_transform(df_clean['diagnosis'])
    print(" Diagnostico convertido: M=1 (maligno), B=0 (benigno)")
    
    # Verificar dados ausentes
    missing_data = df_clean.isnull().sum()
    if missing_data.sum() > 0:
        print(f" Dados ausentes encontrados:")
        print(missing_data[missing_data > 0])
    else:
        print(" Nenhum dado ausente encontrado")
    
    # Verificar valores duplicados
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        print(f" {duplicates} linhas duplicadas encontradas")
        df_clean = df_clean.drop_duplicates()
        print(" Linhas duplicadas removidas")
    else:
        print(" Nenhuma linha duplicada encontrada")
    
    print(f"\nDimensões após limpeza: {df_clean.shape[0]} linhas e {df_clean.shape[1]} colunas")
    
    return df_clean

def analisar_estrutura(df):
    """
    Analisa a estrutura básica dos dados
    """
    print("\n" + "=" * 60)
    print("ANALISE DA ESTRUTURA DOS DADOS")
    print("=" * 60)
    
    # Informações básicas
    print("\nINFORMACOES DO DATASET:")
    print(df.info())
    
    # Estatísticas descritivas
    print("\nESTATISTICAS DESCRITIVAS:")
    print(df.describe())
    
    # Distribuição do diagnóstico
    print("\nDISTRIBUICAO DO DIAGNOSTICO:")
    diagnosis_counts = df['diagnosis'].value_counts()
    print(diagnosis_counts)
    print(f"Proporção Benigno: {diagnosis_counts[0]/len(df)*100:.1f}%")
    print(f"Proporção Maligno: {diagnosis_counts[1]/len(df)*100:.1f}%")

def criar_visualizacoes(df):
    """
    Cria visualizações importantes dos dados
    """
    print("\n" + "=" * 60)
    print("CRIANDO VISUALIZACOES")
    print("=" * 60)
    
    # Configurar figura para múltiplos subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Gráfico de barras da contagem de diagnósticos
    plt.subplot(2, 3, 1)
    diagnosis_counts = df['diagnosis'].value_counts()
    colors = ['lightblue', 'lightcoral']
    labels = ['Benigno (0)', 'Maligno (1)']
    plt.bar(labels, diagnosis_counts.values, color=colors, alpha=0.7)
    plt.title('Distribuicao dos Diagnosticos', fontweight='bold')
    plt.ylabel('Contagem')
    
    # Adicionar valores nas barras
    for i, v in enumerate(diagnosis_counts.values):
        plt.text(i, v + max(v * 0.05, 10), str(v), ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. Histograma do radius_mean
    plt.subplot(2, 3, 2)
    plt.hist(df['radius_mean'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribuicao do Raio Medio', fontweight='bold')
    plt.xlabel('Radius Mean')
    plt.ylabel('Frequencia')
    
    # 3. Histograma do area_mean
    plt.subplot(2, 3, 3)
    plt.hist(df['area_mean'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Distribuicao da Area Media', fontweight='bold')
    plt.xlabel('Area Mean')
    plt.ylabel('Frequencia')
    
    # 4. Histograma do concavity_mean
    plt.subplot(2, 3, 4)
    plt.hist(df['concavity_mean'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Distribuicao da Concavidade Media', fontweight='bold')
    plt.xlabel('Concavity Mean')
    plt.ylabel('Frequencia')
    
    # 5. Gráfico de dispersão radius_mean vs area_mean
    plt.subplot(2, 3, 5)
    colors_scatter = ['blue' if x == 0 else 'red' for x in df['diagnosis']]
    plt.scatter(df['radius_mean'], df['area_mean'], c=colors_scatter, alpha=0.6)
    plt.title('Radius Mean vs Area Mean', fontweight='bold')
    plt.xlabel('Radius Mean')
    plt.ylabel('Area Mean')
    
    # Adicionar legenda
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.6, label='Benigno'),
                      Patch(facecolor='red', alpha=0.6, label='Maligno')]
    plt.legend(handles=legend_elements)
    
    # 6. Boxplot de algumas variáveis importantes
    plt.subplot(2, 3, 6)
    features_box = ['radius_mean', 'texture_mean', 'area_mean']
    df_box = df[features_box + ['diagnosis']]
    df_melted = df_box.melt(id_vars=['diagnosis'], var_name='Feature', value_name='Value')
    
    sns.boxplot(data=df_melted, x='Feature', y='Value', hue='diagnosis')
    plt.title('Boxplot das Caracteristicas por Diagnostico', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout(pad=2.0)
    plt.savefig('visualizacoes_gerais.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Mapa de calor de correlação
    print("\nCriando mapa de calor de correlacao...")
    
    # Selecionar apenas colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calcular correlação com diagnóstico
    correlations = df[numeric_cols].corr()['diagnosis'].abs().sort_values(ascending=False)
    
    # Selecionar as 10 variáveis mais correlacionadas (excluindo diagnosis)
    top_10_features = correlations[1:11].index.tolist()
    
    # Criar mapa de calor
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[top_10_features + ['diagnosis']].corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Mapa de Calor - 10 Variaveis Mais\nCorrelacionadas com Diagnostico', 
              fontweight='bold', fontsize=12)
    plt.tight_layout(pad=2.0)
    plt.savefig('mapa_calor_correlacao.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualizacoes salvas como 'visualizacoes_gerais.png' e 'mapa_calor_correlacao.png'")

def analisar_agrupamentos(df):
    """
    Analisa os dados agrupados por diagnóstico
    """
    print("\n" + "=" * 60)
    print("ANALISE POR AGRUPAMENTOS")
    print("=" * 60)
    
    # Agrupar por diagnóstico e calcular médias
    grouped = df.groupby('diagnosis').mean()
    
    print("\nMEDIAS DAS CARACTERISTICAS POR DIAGNOSTICO:")
    print(grouped.round(3))
    
    # Selecionar algumas características importantes para comparação
    important_features = ['radius_mean', 'texture_mean', 'area_mean', 'concavity_mean', 'concave points_mean']
    
    print(f"\nCOMPARACAO DAS PRINCIPAIS CARACTERISTICAS:")
    comparison_df = grouped[important_features]
    print(comparison_df.round(3))
    
    # Criar gráfico de barras comparativo
    plt.figure(figsize=(15, 8))
    
    x = np.arange(len(important_features))
    width = 0.35
    
    plt.bar(x - width/2, comparison_df.iloc[0], width, label='Benigno', alpha=0.8, color='lightblue')
    plt.bar(x + width/2, comparison_df.iloc[1], width, label='Maligno', alpha=0.8, color='lightcoral')
    
    plt.xlabel('Caracteristicas')
    plt.ylabel('Valor Medio')
    plt.title('Comparacao das Caracteristicas por Diagnostico', fontweight='bold')
    plt.xticks(x, [col.replace('_mean', '').title() for col in important_features], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    plt.savefig('comparacao_diagnosticos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(" Grafico de comparacao salvo como 'comparacao_diagnosticos.png'")

def gerar_insights(df):
    """
    Gera insights importantes da análise
    """
    print("\n" + "=" * 60)
    print("INSIGHTS IMPORTANTES")
    print("=" * 60)
    
    # Calcular correlações com diagnóstico
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['diagnosis'].abs().sort_values(ascending=False)
    
    print("\nPRINCIPAIS DESCOBERTAS:")
    print("=" * 40)
    
    # 1. Distribuição dos diagnósticos
    diagnosis_counts = df['diagnosis'].value_counts()
    print(f"1. Distribuicao dos diagnosticos:")
    print(f"   - Benignos: {diagnosis_counts[0]} ({diagnosis_counts[0]/len(df)*100:.1f}%)")
    print(f"   - Malignos: {diagnosis_counts[1]} ({diagnosis_counts[1]/len(df)*100:.1f}%)")
    
    # 2. Top 5 características mais correlacionadas
    print(f"\n2. Top 5 caracteristicas mais correlacionadas com diagnostico:")
    top_5 = correlations[1:6]
    for i, (feature, corr) in enumerate(top_5.items(), 1):
        print(f"   {i}. {feature}: {corr:.3f}")
    
    # 3. Estatísticas descritivas por grupo
    print(f"\n3. Diferencas entre grupos (Benigno vs Maligno):")
    important_features = ['radius_mean', 'texture_mean', 'area_mean', 'concavity_mean']
    
    for feature in important_features:
        benign_mean = df[df['diagnosis'] == 0][feature].mean()
        malignant_mean = df[df['diagnosis'] == 1][feature].mean()
        diff_percent = ((malignant_mean - benign_mean) / benign_mean) * 100
        
        print(f"   - {feature.replace('_mean', '').title()}:")
        print(f"     Benigno: {benign_mean:.2f}")
        print(f"     Maligno: {malignant_mean:.2f}")
        print(f"     Diferenca: {diff_percent:+.1f}%")
    
    # 4. Insights sobre qualidade dos dados
    print(f"\n4. Qualidade dos dados:")
    print(f"   - Total de registros: {len(df)}")
    print(f"   - Total de características: {len(df.columns)}")
    print(f"   - Dados ausentes: {df.isnull().sum().sum()}")
    print(f"   - Registros duplicados: {df.duplicated().sum()}")
    
    # 5. Recomendações
    print(f"\n5. RECOMENDACOES PARA MODELAGEM:")
    print(f"   - O dataset esta bem balanceado entre classes")
    print(f"   - As caracteristicas de concavidade e pontos concavos sao muito importantes")
    print(f"   - Radius e area sao bons preditores de malignidade")
    print(f"   - Considerar normalizacao das variaveis para modelagem")
    print(f"   - Usar validacao cruzada devido ao tamanho moderado do dataset")

def main():
    """
    Função principal que executa toda a análise
    """
    print("ANALISE EXPLORATORIA DE DADOS - CANCER DE MAMA")
    print("=" * 60)
    
    # 1. Carregar dados
    df = carregar_dados()
    if df is None:
        return
    
    # 2. Limpar dados
    df_clean = limpar_dados(df)
    
    # 3. Analisar estrutura
    analisar_estrutura(df_clean)
    
    # 4. Criar visualizações
    criar_visualizacoes(df_clean)
    
    # 5. Analisar agrupamentos
    analisar_agrupamentos(df_clean)
    
    # 6. Gerar insights
    gerar_insights(df_clean)
    
    print("\n" + "=" * 60)
    print("ANALISE CONCLUIDA COM SUCESSO!")
    print("=" * 60)
    print("Arquivos gerados:")
    print("   - visualizacoes_gerais.png")
    print("   - mapa_calor_correlacao.png")
    print("   - comparacao_diagnosticos.png")

if __name__ == "__main__":
    main() 