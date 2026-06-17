# Projekt 6 — Problem wieloagentowy (CollectCoins)

## 1. Opis środowiska

Zaimplementowano własne środowisko wieloagentowe **CollectCoins** (API PettingZoo `ParallelEnv`):

- **Agenci:** `agent_0`, `agent_1` — poruszają się po siatce 9×9.
- **Akcje (dyskretne):** stay, up, down, left, right.
- **Cel:** zebrać 4 monety w maksymalnie 65 krokach.
- **Przeszkody:** stałe ściany (dwa słupy) + sensory ścian w obserwacji.
- **Nagroda:** +1 za monetę (wspólna), shaping za zbliżanie się do monety, kara −0.01/krok.

## 2. Algorytmy i implementacja

Użyto **Stable-Baselines3** (trening na CPU — zalecane dla MLP):

| Eksperyment | Opis |
|---|---|
| **Shared (ten sam algorytm)** | Jeden model PPO/A2C trenowany na `agent_0`; w ewaluacji **parameter sharing** — ten sam model steruje oboma agentami. |
| **Mixed (różne algorytmy)** | `agent_0` = PPO, `agent_1` = A2C w jednym epizodzie. |

**Porównane algorytmy:** PPO vs A2C. **Warianty PPO:** v0 (lr=3e-4) vs v1 (lr=1e-3, większa entropia).

## 3. Eksperymenty

Dla każdej konfiguracji uruchomiono **3 niezależne seedy**. Budżet: 250 000 kroków na run (shared) lub 125k+125k (mixed). Ewaluacja co 5 000 kroków na 20 epizodach — wyniki zapisywane do CSV (`results/`).

**E1 — porównanie algorytmów:** PPO vs A2C (shared, wariant v0).

**E2 — warianty PPO:** v0 vs v1 (shared).

**E3 — ten sam vs różny algorytm:** PPO (oba agenci) vs PPO+A2C (mieszany zespół).

## 4. Jak wygląda uczenie

1. Reset środowiska — losowe pozycje agentów i monet.
2. Polityka wybiera akcje; środowisko zwraca nagrodę i nową obserwację.
3. SB3 zbiera trajektorie i aktualizuje sieć (PPO: wiele epok na batchu; A2C: on-policy krok po kroku).
4. Co `EVAL_FREQ` kroków — ewaluacja na osobnym seedzie, zapis `mean_reward ± std` do CSV.
5. Po treningu — deterministyczna ewaluacja (`evaluate.py`) i wykresy (`plot.py`).

Początkowo nagroda jest bliska zera (losowe ruchy). W miarę uczenia agenci coraz częściej docierają do monet — krzywa rośnie i stabilizuje się.

## 5. Wyniki

Wykresy (po uruchomieniu treningu):

- `plots/algo_comparison.png` — PPO vs A2C
- `plots/ppo_variants.png` — PPO v0 vs v1
- `plots/same_vs_mixed.png` — ten sam algorytm vs mieszany zespół

**Wnioski (uzupełnij po treningu):**

- PPO zwykle daje stabilniejszą krzywą uczenia niż A2C w tym środowisku.
- Wyższy `ent_coef` (v1) może spowolnić zbieżność, ale zwiększyć eksplorację.
- Zespół mixed (PPO+A2C) często osiąga gorszą kooperację niż shared PPO, bo agenci nie byli trenowani wspólnie od początku.

## 6. Uruchomienie

```bash
cd /home/mitchellds/studia/sem6/IOwADC
source .venv2/bin/activate

# Trening shared (PPO + A2C + warianty PPO)
python -m project06.train_shared

# Trening mixed (PPO agent_0 + A2C agent_1)
python -m project06.train_mixed

# Ewaluacja i wykresy
python -m project06.evaluate
python -m project06.plot

# Wizualizacja gry (pygame) — okno z siatką, agentami i monetami
python -m project06.visualize --mode shared --algo ppo --seed 0
python -m project06.visualize --mode mixed --seed 0

# Zrzut ekranu do raportu (bez okna)
python -m project06.visualize --mode shared --algo ppo --seed 0 --export project06/plots/demo_collect_coins.png
```

Szybki test (10k kroków, 1 seed):

```bash
python -m project06.train_shared --algo ppo --variant v0 --seed 0 --total-timesteps 10000
```
