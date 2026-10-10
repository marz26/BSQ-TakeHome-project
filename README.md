# BlackSwan Quants — INTC / AMD Pairs Trading

## Fase 1: dati e warm-up

Lo script scarica i prezzi di chiusura rettificati di INTC e AMD dal
`2010-01-01`, applica il limite informativo inclusivo `as_of` e salva
checkpoint pickle per ticker in `data/`. La cache è identificata dal ticker e
dall'intervallo richiesto e riduce le richieste Yahoo Finance ripetute; non è
un meccanismo di aggiornamento automatico dei valori già memorizzati. Per
aggiornare uno specifico checkpoint occorre rimuovere quel file dalla cache
prima di rieseguire il comando.

Per eseguire il warm-up con il cutoff D-1 predefinito:

```bash
python main.py
```

Per un'esecuzione riproducibile:

```bash
python main.py --as-of 2026-10-09 --start-date 2010-01-01
```

Le opzioni `--cache-dir` e `--output-dir` consentono di specificare le
directory della cache e dei grafici. Vengono generati `prices.png`,
`price_ratio.png`, `price_difference.png` e `rolling_correlation_60d.png`
in `outputs/warmup/`.

I test del data loader sono offline e si eseguono con:

```bash
pytest tests/test_data_loader.py
```

## Fase 2: validazione IS/OOS

`src/backtest.py` esegue una ricerca esplicita nel solo periodo In-Sample,
definito da `ProjectConfig.train_test_split_date`, e valuta una sola volta la
configurazione selezionata nel periodo Out-of-Sample. Per default vengono
registrate nove combinazioni (3 soglie z-score × 3 finestre mobili); ogni
tentativo e i relativi risultati IS sono conservati in `report.trials` e
riportati nel log.

Esempio d'integrazione:

```python
import logging
from datetime import date

from src.backtest import run_backtest
from src.config import ProjectConfig
from src.data_loader import load_price_data

logging.basicConfig(level=logging.INFO)
config = ProjectConfig()
as_of = date(2026, 10, 9)
prices = load_price_data(
    (config.ticker_a, config.ticker_b),
    start_date=date.fromisoformat(config.start_date),
    as_of=as_of,
)
report = run_backtest(prices, config=config, as_of=as_of)
```

Il giorno di split è il primo giorno OOS. I coefficienti OLS sono stimati
soltanto prima dello split; i segnali sono ritardati di una sessione. I report
IS e OOS hanno PnL spread in unità di prezzo (non rendimento sul capitale) e non
includono commissioni, slippage, costi di prestito o market impact. Il win rate
considera solo trade chiusi; il PnL complessivo include l'eventuale trade ancora
aperto mark-to-market.

### Sensibilità dei parametri — run riproducibile

La ricerca registra ogni combinazione nel `BacktestReport.trials`. La
classificazione automatica confronta il parametro selezionato con il valore
immediatamente adiacente, mantenendo invariato l'altro asse; marca la selezione
come fragile se un vicino inverte un PnL IS positivo oppure lo riduce almeno
del 50%. Una baseline IS non positiva è classificata come inconclusiva (e
segnalata dal report), non come prova di robustezza. Questo criterio è una
diagnostica deterministica, non un test statistico. L'OOS è mostrato
separatamente e non entra né nella selezione né nel giudizio di sensibilità.

Sul run riproducibile con dati cache aggiornati al `2026-10-09` e split
`2024-01-01`, la griglia ha selezionato `entry_z=2.0`, `window=90`: PnL IS
`+8.0897` unità spread. Cambiando la soglia ai valori adiacenti, mantenendo la
finestra a 90, il PnL IS è stato `+7.7212` (soglia 1.5) e `+7.3419` (2.5).
Riducendo invece la finestra a 60, con soglia 2.0, il PnL IS è diventato
`-5.9064`, invertendo segno; la finestra 40 ha prodotto `-6.7910`. Il report
classifica quindi questa selezione **fragile rispetto alla finestra**: la
performance positiva IS dipende sensibilmente dalla scelta dei 90 giorni e
può riflettere rumore o specificità del campione, non necessariamente un
effetto strutturale.

Nel medesimo run, la strategia ha riportato OOS `-27.1274` unità spread (un
trade, win rate 0%). È un risultato diagnostico negativo, non una stima
generalizzabile: il PnL è non normalizzato per capitale e non include costi,
slippage o vincoli di esecuzione. Nessuna conclusione di robustezza va tratta
da un singolo split o da questa griglia limitata; risultati futuri devono
riportare cutoff dati e tutti i tentativi effettivamente svolti.

## Fase 3: stabilità della relazione

La pipeline calcola un ADF rolling a 90 osservazioni sullo spread OOS costruito
con i coefficienti OLS stimati in-sample. Ogni p-value è indicizzato alla fine
della propria finestra e utilizza solo osservazioni fino a quella data. Un
p-value superiore a 0.05 è etichettato come evidenza insufficiente contro
l'ipotesi nulla di unit root; non dimostra da solo una rottura strutturale o
una causa economica. Le prime 89 date non hanno un p-value rolling valido; la
pipeline richiede almeno 90 osservazioni OOS per poter calcolare una finestra
completa.

I valori ADF sono confrontati retrospettivamente con il PnL giornaliero OOS
salvato dal backtest. I log riportano conteggi, PnL totale e drawdown
osservato per date stabili e instabili; il confronto restituisce anche il PnL
medio per gruppo. Il drawdown è calcolato sulla curva cumulativa completa
allineata e poi attribuito al gruppo ADF della data del drawdown, non è un
drawdown autonomo ricalcolato separatamente per ogni regime. Il confronto è
retrospettivo e descrittivo: non modifica posizioni o parametri e non prova che
la non stazionarietà abbia causato le perdite.

## Fase 4: short interest Nasdaq

`src/short_interest.py` interroga l'endpoint pubblico Nasdaq per INTC e AMD e
normalizza settlement date, short interest, volume medio e days-to-cover. La
risposta viene richiesta con header HTTP browser-like. La copertura e la
profondità storica effettive dipendono dalla risposta del vendor e non sono
garantite dal codice; lo schema può cambiare e l'endpoint può rispondere con
errori temporanei o 403. Se fetch o allineamento falliscono, la pipeline
registra un warning e mantiene validi i risultati delle fasi precedenti.

La settlement date non è automaticamente la data in cui l'informazione era
pubblica. Poiché l'API non espone una publication timestamp storica, l'analisi
applica un ritardo prudenziale di 14 giorni di calendario dalla settlement date
prima di rendere un report disponibile, poi fa forward-fill solo in avanti
sulle date giornaliere OOS. È un proxy conservativo, non una data di rilascio
verificata dal vendor. I valori di short interest sono conteggi di azioni e
non vengono confrontati come se le due società avessero lo stesso float.

L'analisi confronta, alle date stimate di disponibilità dei report, la
variazione percentuale dello short interest con la variazione assoluta
giornaliera dello spread. Può anche riportare il PnL medio nelle date
classificate stabili/instabili dall'ADF. Le correlazioni sono descrittive e
possono essere non disponibili se mancano osservazioni variabili sufficienti;
non misurano pressione direzionale, posizioni istituzionali nette o causalità.

#### Risultati empirici

Non si riportano qui statistiche Nasdaq: i valori precedentemente annotati non
sono stati ricalcolati e verificati con lo stesso snapshot dei prezzi, le
stesse risposte vendor e la stessa versione dell'analisi. Una nuova esecuzione
può produrre copertura e risultati diversi; per documentarli occorre
conservare cutoff, date di settlement/availability e valori allineati.

## Fase 5: simulazione dei percorsi sintetici

`src/simulation.py` implementa un **Joint Moving-Block Bootstrap** dei
rendimenti semplici giornalieri. Ogni blocco è estratto sulle due colonne
contemporaneamente, preservando la dipendenza contemporanea INTC/AMD e la
struttura seriale locale all'interno dei blocchi (la continuità seriale non è
preservata ai confini tra blocchi). I prezzi sintetici sono ricostruiti
compostando i rendimenti campionati a partire dall'ultimo prezzo In-Sample.

La pipeline usa solo prezzi con data strettamente precedente allo split per
formare i blocchi; hedge ratio e configurazione della strategia restano quelli
stimati/selezionati dal backtest In-Sample. Ogni percorso applica questi
parametri congelati e usa segnali ritardati di una sessione. Per impostazione
predefinita vengono simulati 500 percorsi, con orizzonte pari al numero di
osservazioni OOS, blocchi di 10 sessioni e seed `42`. I log registrano seed,
orizzonte, dimensione dei blocchi e statistiche della distribuzione del PnL.

### Risultati del run riproducibile (500 percorsi)

Ricalcolando il backtest e la simulazione sulla cache locale con cutoff
`2026-10-09`, split `2024-01-01`, seed `42`, blocchi di 10 sessioni e orizzonte
di 696 sessioni, si ottengono:

* **Percorsi simulati:** 500
* **PnL Medio:** `+13.9969` (Mediana: `+1.9003`)
* **Deviazione standard:** `81.6046`
* **Intervallo percentile 5°–95° della distribuzione simulata:** `p05 = -80.2708`, `p95 = +162.2771` (minimo `-355.5313`, massimo `+541.9668`)
* **Frazione di percorsi con PnL positivo:** `51.60%` (258 su 500)

Questi numeri riassumono soltanto i percorsi generati dal bootstrap
condizionale sui rendimenti In-Sample disponibili; non sono intervalli di
confidenza per una quantità futura né probabilità calibrate di profitto. La
frazione `51.60%` è una frequenza Monte Carlo su 500 percorsi e non costituisce
da sola un test di vantaggio rispetto al 50%. La mediana positiva ma vicina a
zero, insieme alla dispersione ampia e ai percentili che includono perdite e
guadagni rilevanti, mostra una distribuzione simulata incerta e variabile: non
dimostra robustezza, equivalenza a un lancio di moneta, né che la perdita OOS
reale sia o non sia un'anomalia. Il bootstrap non modella regimi assenti dal
campione, non stabilisce la stazionarietà e non tiene conto di costi,
slippage, vincoli di esecuzione o rischio di selezione dei parametri. Il PnL
resta espresso in unità di prezzo dello spread, non in rendimento sul capitale.

I test completi si eseguono con:

```bash
pytest tests/
```