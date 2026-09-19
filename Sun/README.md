# [🇧🇷 PT BR] Instruções

Este diretório reúne os dados solares (AIA/SDO) usados como modelo real da estrela no ECLIPSE, além dos notebooks de download e análise que consomem esses dados.

## Estrutura

```
Sun/
├── sdo_aia_download/          # dados .fits baixados, organizados por comprimento de onda
│   ├── 171/                   # canal 171 Å (referência principal do projeto)
│   ├── 1700/                  # canal 1700 Å
│   ├── 304/                   # canal 304 Å
│   ├── transit-venus/         # trânsito de Vênus (benchmark, não é um evento de CME)
│   └── main-sun-lightcurves-sunpy.ipynb
└── UV/                        # notebooks de download/análise multi-comprimento de onda
    ├── download-sdo-lightcurves.ipynb
    ├── main-fft-analysis.ipynb
    ├── download-sdo-data.ipynb
    └── Products/               # outputs (figuras .png e arrays .npy), por comprimento de onda
```

### `sdo_aia_download/`

Cada subpasta de comprimento de onda contém pastas nomeadas `<data>` (COM CME) e `<data>-no-cme` (SEM CME, referência), com os `.fits` baixados para os 4 eventos de CME usados no projeto:

| Data | Observação |
|---|---|
| 2011-06-05 | Halo, classe GOES B durante todo o período, sem *flares* associados (evento de referência principal) |
| 2017-04-24 | Evento de CME |
| 2017-04-30 | CME com *flare* associado, **não** halo |
| 2022-10-01 | CME com *flare* associado, halo |

**Convenção de nomes:** o canal 171 Å foi baixado antes de existir a convenção de sufixo por comprimento de onda, então suas pastas não têm sufixo numérico (`2011-06-05`, não `2011-06-05-171`). Todo canal adicionado depois usa `-<comprimento_de_onda>` como sufixo (`2011-06-05-1700`, `2011-06-05-304`). Os notebooks em `UV/` já sabem lidar com essa exceção automaticamente.

`main-sun-lightcurves-sunpy.ipynb` foi o notebook original de download (só 171 Å, editado ao longo do projeto — hoje não contém mais as células de busca de todos os eventos). Para baixar qualquer canal do AIA nos 4 eventos de referência, use `UV/download-sdo-lightcurves.ipynb`.

### `UV/`

- **`download-sdo-lightcurves.ipynb`** — baixa o intervalo completo de frames (múltiplos, a cada 10 min) para os 4 eventos de referência, em qualquer canal do AIA (94, 131, 171, 193, 211, 304, 335, 1600, 1700 ou 4500 Å — selecionável via `input()` ao rodar o notebook).
- **`main-fft-analysis.ipynb`** — simula o trânsito de HD 189733 Ab sobre os frames baixados (com/sem CME) usando o Core do ECLIPSE, e roda a análise de resíduo, FFT e PCA entre os 4 eventos. Réplica genérica por comprimento de onda do pipeline original de 171 Å (`Eclipse/02-Notebooks/main-fft-pca-sun-signals.ipynb`).
- **`download-sdo-data.ipynb`** — download rápido e exploratório de **um único frame** (não salva na estrutura `sdo_aia_download/`; pensado para conferência visual pontual, não para alimentar o pipeline de simulação).
- **`Products/`** — figuras (`.png`, 300 dpi) e arrays (`.npy`) gerados por `main-fft-analysis.ipynb`, organizados em subpastas por comprimento de onda (`Products/171A/`, `Products/1700A/`, etc.).

**Observação sobre 2310 Å:** o filtro UVM2 do XMM-Newton/OM (2310 Å), usado nas observações reais de HD 189733 A, não existe como canal do AIA — o instrumento salta de 1700 Å direto para 4500 Å. Ver histórico de discussão no próprio `main-fft-analysis.ipynb` sobre alternativas (ex.: SUIT/Aditya-L1).

---

# [🇺🇸 EN] Instructions

This directory holds the solar (AIA/SDO) data used as the real star model in ECLIPSE, plus the download and analysis notebooks that consume that data.

## Structure

```
Sun/
├── sdo_aia_download/          # downloaded .fits data, organized by wavelength
│   ├── 171/                   # 171 A channel (project's main reference)
│   ├── 1700/                  # 1700 A channel
│   ├── 304/                   # 304 A channel
│   ├── transit-venus/         # Venus transit (benchmark, not a CME event)
│   └── main-sun-lightcurves-sunpy.ipynb
└── UV/                        # multi-wavelength download/analysis notebooks
    ├── download-sdo-lightcurves.ipynb
    ├── main-fft-analysis.ipynb
    ├── download-sdo-data.ipynb
    └── Products/               # outputs (.png figures and .npy arrays), by wavelength
```

### `sdo_aia_download/`

Each wavelength subfolder contains folders named `<date>` (with CME) and `<date>-no-cme` (reference, without CME), holding the `.fits` files downloaded for the project's 4 reference CME events:

| Date | Note |
|---|---|
| 2011-06-05 | Halo CME, GOES class B throughout, no associated flares (main reference event) |
| 2017-04-24 | CME event |
| 2017-04-30 | CME with associated flare, not halo |
| 2022-10-01 | CME with associated flare, halo |

**Naming convention:** the 171 A channel was downloaded before the per-wavelength suffix convention existed, so its folders have no numeric suffix (`2011-06-05`, not `2011-06-05-171`). Every channel added afterwards uses `-<wavelength>` as a suffix (`2011-06-05-1700`, `2011-06-05-304`). The notebooks in `UV/` already handle this exception automatically.

`main-sun-lightcurves-sunpy.ipynb` was the original download notebook (171 A only, edited multiple times over the project — it no longer contains the search cells for all events). To download any AIA channel for the 4 reference events, use `UV/download-sdo-lightcurves.ipynb`.

### `UV/`

- **`download-sdo-lightcurves.ipynb`** — downloads the full frame interval (multiple frames, every 10 min) for the 4 reference events, for any AIA channel (94, 131, 171, 193, 211, 304, 335, 1600, 1700 or 4500 A — selectable via `input()` when running the notebook).
- **`main-fft-analysis.ipynb`** — simulates the HD 189733 Ab transit over the downloaded frames (with/without CME) using the ECLIPSE Core, and runs the residual, FFT and PCA analysis across the 4 events. Generic, per-wavelength replica of the original 171 A pipeline (`Eclipse/02-Notebooks/main-fft-pca-sun-signals.ipynb`).
- **`download-sdo-data.ipynb`** — quick, exploratory download of a **single frame** (does not save into the `sdo_aia_download/` structure; meant for one-off visual checks, not for feeding the simulation pipeline).
- **`Products/`** — figures (`.png`, 300 dpi) and arrays (`.npy`) produced by `main-fft-analysis.ipynb`, organized into per-wavelength subfolders (`Products/171A/`, `Products/1700A/`, etc.).

**Note on 2310 A:** the XMM-Newton/OM UVM2 filter (2310 A), used in the real HD 189733 A observations, does not exist as an AIA channel — the instrument jumps from 1700 A straight to 4500 A. See the discussion carried in `main-fft-analysis.ipynb` for alternatives (e.g., SUIT/Aditya-L1).
