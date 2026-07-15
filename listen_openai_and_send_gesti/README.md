# listen_openai_and_send_gesti

Listener microfono -> OpenAI (trascrizione + risposta breve) -> TTS (voce) -> invio TCP del comando `GESTI` con secondi.

## Cosa fa

- Registra clip audio WAV dal microfono (durata configurabile).
- Invia il WAV a OpenAI (trascrizione).
- Genera una risposta breve in inglese (configurabile con `--chat-system-prompt`).
- Legge la risposta con TTS dallo speaker (disattivabile con `--no-speak-response`).
- Manda al client TCP una riga (formato custom con `--out-format`).

Formato default inviato:

`[GESTI,{chat_api_seconds:.2f}]` (es. secondi chiamata risposta AI)

Esempio:

`[GESTI,1.25]`

## Setup

1. Attiva env Python del progetto.
2. Installa dipendenze minime:

```bash
pip install sounddevice numpy openai
```

3. Imposta la API key:

```bash
export OPENAI_API_KEY="sk-..."
```

## Avvio rapido

Modalita server TCP (come il tuo comando attuale):

```bash
python3 listen_openai_and_send_gesti/listen_openai_and_send_gesti.py \
  --tcp-mode server \
  --server-host 0.0.0.0 \
  --server-port 8765 \
  --segment-seconds 5
```

## Opzioni utili

- `--save-dir runtime_chunks_live/gesti_openai` (default): dove salva i WAV.
- `--sd-device N`: scegli input microfono.
- `--out-format` per cambiare la riga TCP (placeholder: `{command}`, `{clip_seconds}`, `{api_seconds}`, `{chat_api_seconds}`, `{tts_seconds}`, `{text}`).

## Debug (senza microfono)

Invia a ogni pressione su `C` una riga fissa, utile per testare il lato client:

```bash
python3 listen_openai_and_send_gesti/listen_openai_and_send_gesti.py \
  --tcp-mode server \
  --server-host 0.0.0.0 \
  --server-port 8765 \
  --debug-mode \
  --debug-seconds 1.30
```

L'invio e` `C` -> `[GESTI,1.30]`.

## Note

- Questo script non fa mapping canzone->policy: invia comandi con prefisso `GESTI`.
- I WAV restano su disco per controllare durata e qualita.
