# 03-Products — mapa de origem

Cada produto aqui vem de um notebook específico. Esta pasta foi reorganizada em 2026-08-31 (antes disso, tudo ficava solto na raiz sem indicação de origem).

## `xray/`
Gerado por [`../main-xray-lightcurves.ipynb`](../main-xray-lightcurves.ipynb). Curvas de luz e resíduos em raios-X (EPIC-pn), incluindo os testes Δχ² com CME Cápsula/Gota.

## `uv-lightcurves/`
Gerado por [`../main-uv-lightcurves.ipynb`](../main-uv-lightcurves.ipynb) — a análise mestra em UVM2 (10 trânsitos, detrending, ajuste MCMC Cápsula/Gota).

- Arquivos soltos (`average_lightcurve_hd189733.png`, `lightcurve-vs-model.png`, `lightcurve-obs-06-vs-model.png`, etc.) e a subpasta `residual-analysis/` vêm de células fixas do notebook, uma por gráfico.
- Os arquivos sem célula ativa correspondente no notebook atual (`detrending-modelo.png`, `result-mcmc.png`, `fig-sinal-residual.png`, `fig-sinal-residual-fit.png`, `modelo-com-sinal-vs-lightcurve.png`, e a pasta `mcmc-cme/` legada) e um arquivo com nome inconsistente em `plots-papper/model-vs-obs/` (`obs-1-adjust-result-capsule.png`) foram removidos em 2026-08-31 por não terem mais utilidade. `adjust-result-gota.png` (nomenclatura antiga) foi renomeado para `lightcurve-obs-06-vs-model.png` em vez de removido.
- **`mcmc-runs/`** — a partir da seção *"Comparando curva de luz observada com modelo gerado com sinal de CME adicionado"*, cada execução do notebook cria uma pasta própria aqui, nomeada `obs<N>_<geometria>_<timestamp>/`, em vez de sobrescrever a rodada anterior. Dentro de cada pasta:
  - as figuras daquela rodada (ajuste, resíduo, MCMC);
  - `mcmc_best_values.txt` — os melhores valores do MCMC (bloco "MELHORES VALORES (MCMC)") que geraram aquelas figuras.

  As 4 pastas já existentes ali (`obs3_drop_20260824-223446/`, `obs6_drop_20260709-234157/`, `obs6_capsule_20260627-012419/`, `obs3_capsule_20260623-004910/`) são **reconstruções retroativas**, feitas em 2026-08-31 agrupando os arquivos que estavam soltos na raiz por proximidade de `mtime` — não foram geradas pelo código novo. Cada uma tem um `_RECONSTRUCTED.md` explicando a evidência usada e o que ficou faltando (rodadas antigas sobrescreviam nomes fixos, então parte dos arquivos de cada execução real já tinha se perdido antes dessa reorganização). Nenhuma delas tem `mcmc_best_values.txt`, porque essa rodada é anterior a essa funcionalidade.

## `plots-papper/`
Gerado por um notebook `plots-paper.ipynb` que **não existe mais** no repositório (removido após o commit `450b738`, "feat: add plots for paper"). Contém `activity-signal/`, `all-observations/` e `model-vs-obs/` — figuras provavelmente usadas em uma versão de paper/apresentação. Mantidas aqui sem alteração até decisão sobre o que fazer com elas (ver nota em `.personal-setup/retomar-contexto.md`).
