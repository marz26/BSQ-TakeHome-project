from __future__ import annotations
from dataclasses import dataclass
from datetime import date

@dataclass(frozen=True)
class ProjectConfig:
    """Configurazione centralizzata e immutabile per il progetto di Pairs Trading.

    Attributes
    ----------
    ticker_a : str
        Il ticker del primo asset della coppia (default 'INTC').
    ticker_b : str
        Il ticker del secondo asset della coppia (default 'AMD').
    start_date : str
        Data di inizio per lo storico dei prezzi ('YYYY-MM-DD').
    end_date : str
        Data di fine per lo storico dei prezzi ('YYYY-MM-DD').
    train_test_split_date : str
        Data che delimita il periodo In-Sample da quello Out-of-Sample.
    z_score_entry : float
        Soglia di z-score per l'apertura delle posizioni.
    z_score_exit : float
        Soglia di z-score per la chiusura delle posizioni.
    rolling_window : int
        Finestra mobile per il calcolo di media e deviazione standard dello spread.

    Notes
    -----
    L'utilizzo di un dataclass con `frozen=True` garantisce l'immutabilità dei parametri
    durante l'esecuzione della pipeline, prevenendo modifiche accidentali allo stato globale.
    """
    ticker_a: str = "INTC"
    ticker_b: str = "AMD"
    start_date: str = "2010-01-01"
    end_date: str = "2026-10-01"
    train_test_split_date: str = "2024-01-01"
    z_score_entry: float = 2.0
    z_score_exit: float = 0.0
    rolling_window: int = 60