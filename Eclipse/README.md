# [🇧🇷 PR BR] Projeto ECLIPSE

Este repositório reúne uma coleção de scripts voltados à análise de curvas de luz de estrelas hospedeiras por meio do método de trânsito, com foco na detecção e caracterização de fenômenos como manchas estelares, fáculas e possíveis assinaturas de CMEs (Coronal Mass Ejections).

O projeto é baseado no modelo **ECLIPSE**, originalmente proposto por Adriana Valio (Valio, 2003), posteriormente refatorado para Python por Beatriz Duque. Em etapas mais recentes, foram incorporadas técnicas de **MCMC (Markov Chain Monte Carlo)** e **computação paralela**, utilizando scripts em C para otimização de performance no ajuste de modelos (Pinho, Duque e Valio).

---

## 📂 Estrutura dos Scripts

Abaixo estão descritos os principais scripts do projeto e suas respectivas funcionalidades:

### 🔧 Modelagem e Simulação

**`main_para_programadores.py`**  
Script em Python para modelagem detalhada de sistemas estrela–planeta e geração de curvas de luz.  
Permite incluir efeitos como:
- manchas estelares  
- fáculas  
- luas  
- outros fenômenos  

Indicado para usuários com maior familiaridade em programação.

---

**`main-first-steps.ipynb`**  
Notebook introdutório com exemplos de modelagem de curvas de luz.  
Permite simular rapidamente o impacto de diferentes componentes:
- manchas  
- fáculas  
- luas  
- CMEs  

Também suporta modelagem a partir de arquivos `.fits`. Ideal para exploração inicial e visualização.

---

**`main-lightcurves-simulations.ipynb`**  
Geração de curvas de luz utilizando modelos estelares derivados de dados solares (`.fits`).  
Permite simular o impacto de:
- regiões ativas  
- eventos estelares  
- variações estruturais da estrela  

---

**`main-lightcurves-venus-cme.ipynb`**  
Simulação baseada no trânsito de Vênus no Sol.  
Utiliza dados reais para reproduzir cenários fisicamente consistentes, sendo um excelente benchmark para validação de modelos.

---

### 📊 Análise de Sinais

**`main-fft-pca-sun-signals.ipynb`**  
Análise de sinais solares utilizando:
- FFT (Fast Fourier Transform)  
- PCA (Principal Component Analysis)  

Focado na caracterização de CMEs observadas no Sol.

---

**`main-fft-pca-sun-and-hoststar.ipynb`**  
Extensão da análise anterior, incluindo comparação entre:
- sinais solares (referência)  
- sinais observados em estrelas hospedeiras  

Exemplo aplicado à estrela HD189733A.

---

**`main-find-spots.ipynb`**  
Detecção de manchas e fáculas em curvas de luz reais.  
Fluxo principal:
1. Download de dados via biblioteca `lightkurve`  
2. Construção de curva de luz modelo  
3. Subtração entre modelo e dados observacionais  
4. Identificação de assinaturas residuais (manchas/fáculas)

---

### 🔬 Ajuste de Modelos

**`main-mcmc.ipynb`**  
Implementação de ajuste de curvas de luz utilizando:
- algoritmo MCMC  
- paralelização via código em C  

Permite modelagem com e sem manchas, buscando melhor ajuste aos dados observacionais.

---

### 🌌 Aplicação Principal do Projeto

**`main-uv-lightcurves.ipynb`**  
Notebook central do projeto.

Integra:
- análise de curvas de luz em ultravioleta  
- modelagem de sistemas estrela–planeta  
- simulação de trânsito planetário  
- ajuste via MCMC  

Objetivo: investigar possíveis assinaturas de **CMEs em estrelas hospedeiras**.

Objeto principal de estudo:
- Estrela: HD189733A  
- Planeta: HD189733Ab  

---

## 🚀 Considerações Finais

Este conjunto de scripts permite:
- simulação física detalhada de curvas de luz  
- análise espectral e estatística de sinais  
- ajuste robusto de modelos a dados reais  

O projeto pode ser utilizado tanto para fins exploratórios quanto para pesquisa científica em detecção e caracterização de exoplanetas e atividade estelar.