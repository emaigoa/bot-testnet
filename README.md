# Bot de paper trading — ETH 4H (Binance Futures Testnet)

Estrategia validada por backtest 2020-2026: EMA 21/89 + RSI + filtro ATR,
brackets 2.5/4×ATR, cortos espejados, circuit breaker de drawdown.
Corre solo, cada 4 horas, en GitHub Actions — no necesita ninguna PC prendida.

## Puesta en marcha (una sola vez)

1. Crear un repositorio **privado** en GitHub (por ejemplo `bot-testnet`) y subir esta carpeta.
2. En el repo: **Settings → Secrets and variables → Actions → New repository secret**, crear dos:
   - `BINANCE_TESTNET_KEY` → tu API Key de https://testnet.binancefuture.com
   - `BINANCE_TESTNET_SECRET` → tu Secret Key
3. Pestaña **Actions** → habilitar workflows si lo pide → elegir `bot-testnet-4h` → **Run workflow** (prueba manual).
4. Ver el log de la corrida: debe decir `TESTNET equity=...` y `señal=...`. Listo:
   a partir de ahi corre solo en cada cierre de vela 4H (00/04/08/12/16/20 UTC).

## Que hace en cada ciclo

- Lee las velas reales de ETH, evalua la señal al cierre (mismo codigo que el backtest).
- Si esta flat y hay señal: orden MARKET + stop loss y take profit como ordenes
  en el exchange (se ejecutan en tiempo real, el bot no necesita estar despierto).
- Si hay posicion abierta: no toca nada, los brackets la manejan.
- Guarda `bot_log.txt`, `trades_paper.csv` y `state_paper.json` commiteados al repo,
  asi el historial queda versionado y el circuit breaker recuerda el pico de equity.

## Avisos

- **No corras a la vez el bot local y este** sobre la misma cuenta testnet: podrian
  entrar duplicado. Uno u otro.
- GitHub desactiva los crons de repos sin actividad por ~60 dias; los commits del
  propio bot cuentan como actividad, pero si alguna vez lo ves pausado, entra a
  Actions y reactivalo.
- Las claves del testnet operan dinero ficticio. Aun asi: jamas pongas claves de
  tu cuenta real ni aca ni en el codigo.
