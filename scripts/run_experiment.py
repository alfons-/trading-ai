"""
Script principal: ejecuta un experimento completo del pipeline XGBoost + Bybit.

Uso:
    python -m scripts.run_experiment                          # config por defecto, primer símbolo
    python -m scripts.run_experiment --symbol ETHUSDT         # símbolo concreto
    python -m scripts.run_experiment --all                    # todos los símbolos
    python -m scripts.run_experiment --config configs/mi.yaml # config alternativa
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.orchestrator import OrchestratorAgent


def main():
    parser = argparse.ArgumentParser(description="Experimento XGBoost + Bybit")
    parser.add_argument("--config", type=str, default=None, help="Ruta al YAML de configuración")
    parser.add_argument("--symbol", type=str, default=None, help="Símbolo a procesar (ej. BTCUSDT)")
    parser.add_argument("--all", action="store_true", help="Procesar todos los símbolos de la config")
    args = parser.parse_args()

    orchestrator = OrchestratorAgent(config_path=args.config)

    if args.all:
        results = orchestrator.run_all()
        print("\n=== Resumen de todos los experimentos ===")
        for r in results:
            print(f"\n{r['symbol']}:")
            print(f"  Test accuracy: {r['test_metrics']['test_accuracy']:.4f}")
            print(f"  Test AUC:      {r['test_metrics']['test_auc']:.4f}")
            print(f"  Backtest ret:  {r['backtest']['retorno_total']:.4f}")
            print(f"  Buy & hold:    {r['backtest']['retorno_buyhold']:.4f}")
    else:
        result = orchestrator.run(symbol=args.symbol)
        print(f"\nResultado final para {result['symbol']}:")
        print(f"  Test accuracy: {result['test_metrics']['test_accuracy']:.4f}")
        print(f"  Test AUC:      {result['test_metrics']['test_auc']:.4f}")
        print(f"  Backtest ret:  {result['backtest']['retorno_total']:.4f}")
        print(f"  Buy & hold:    {result['backtest']['retorno_buyhold']:.4f}")


if __name__ == "__main__":
    main()
