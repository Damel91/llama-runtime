# Code persistenti — candidata locale 0.10.0

Verifica del 21 ottobre 2026. Codice: `58ae473`. Distribuzione `runtime-llama`,
import `runtime_llama`, licenza Apache-2.0 e attribuzioni upstream preservate.
Consegna di un wheel locale; nessuna pubblicazione o installazione nei consumer.

Flower MCP è l'autorità del ciclo: `llama-runtime`, `REQ-000024`–`REQ-000026`,
`GOAL-000012`, `MILE-000014`, `CHANGE-000014`, `PACKET-000015`, `RUN-000009`.
Architettura adottata revisione 1, packet spec 6 / plan 3. Il presente documento
proietta risultati ottenuti dall'host: non costituisce accettazione umana.

## Implementazione

Un motore persistente dell'adapter gestisce ammissione, preparazione, code per
alias verificato e risultati. Il solo `EmbeddingScheduler` assegna le capacità
HTTP; i batch sono selezionati quando possono partire, includendo record
diventati pronti durante la richiesta precedente. La bridge async collega le
Future del motore all'event loop, senza un coordinatore bloccante per finestra.

Le finestre indipendenti conservano deadline e cancellazione proprie e non
vengono fuse in POST condivise. Ordine, cardinalità, identità, prefissi,
tokenizzazione esatta, ID nativi, topologia drenata e assenza di replay nativi
restano invarianti. Gli input oltre contesto sono misurabili entro i budget
di risorse, ma non ammessi all'inferenza. Nessun requisito AST/splitting viene
spostato dai consumer nella libreria.

`batch_size` è il budget totale di testi della finestra di distribuzione:
256 su due repliche utilizzabili assegna quote fino a 128; su quattro fino a 64.
Sono esempi derivati, non costanti. Resti e selezione ruotano. Il batch HTTP
può essere più piccolo per dati pronti, limiti espliciti di token o byte.
Contesto per input, budget HTTP aggregato e `n_batch`/`n_ubatch` nativi restano
distinti. Il default `embedding_batch_wait_sec=0` non aspetta artificialmente;
un valore positivo abilita coalescing opzionale nello stesso motore.

Budget configurabili: 64 MiB per il pool raw/preparato e, separatamente, per il
pool dei risultati async trattenuti; 1.048.576 token preparati; 8 MiB per payload.
Tokenizer e risposte transitorie hanno limiti finiti propri; non si dichiara un
tetto totale della RSS. Il limite HTTP derivato è `P + T + max(1,P)`.

La qualifica ha portato anche a eliminare osservazioni ridondanti sullo stesso
alias: un preflight già pronto precede l'avvio del controllo di un altro owner.
Sono verificati inoltre gli escape JSON e l'envelope del POST tokenizer prima
dell'invio. Il trasporto nativo usa connessioni distinte e policy `close` senza
dipendere dalla patch keep-alive. La stringa build remota non è un audit binario.

## Verifica deterministica e packaging

- **1.372 test passati** in 57,04 s, Python 3.11.15 / macOS arm64.
- Ruff check e format check passati; diff senza errori di whitespace.
- Build wheel e sdist con `python -m build --no-isolation` riuscita.
- Installazione del wheel finale in ambiente pulito esterno al checkout;
  versioni import/distribuzione entrambe 0.10.0, dipendenze coerenti.
- Costruzione/chiusura inerte verificata senza avviare worker o connessioni;
  tutti i moduli Python nel wheel confrontati byte per byte col sorgente.
- LICENSE, NOTICE e licenza MIT del componente derivato presenti nel wheel.

La suite copre arrivi durante HTTP, code concorrenti, due/quattro repliche,
resti dispari, meno input che repliche, limiti globali, fairness, byte/token,
Unicode, fallimenti parziali, cancellazione, deadline, topologia e drain.
Regressioni specifiche coprono due domini di memoria: risultati fuori ordine
e discovery lento non devono impedire alla prima finestra di progredire, né
propagare timeout/cancellazione a un altro stream. Quattro warning HTTPS sono
attesi nelle fixture locali con certificato autosignato; non sono errori TLS live.
Questa esecuzione non qualifica Linux/Windows o l'helper VPN su una tailnet reale.

## Qualifica live limitata

Nessuna modifica a server, modelli, profili, database, job o consumer. Due alias
EmbeddingGemma 2 BF16 caricati, contesto 2048, batch/microbatch 2048, uno slot per
alias, Flash Attention off; build osservata `b11716-797de63b2`.

Identità contenuto:
`eef0e2b1bd56536465e62b4549e1eb4d8dbaf348cc3aff8b90bfac19b005770a`.
Identità semantica:
`4d6ea47cf2f5a4ed5923374551aa9f3d681771db8c6103c2b95f67d2bb176769`.

Il probe installato usa 511 documenti sintetici univoci, finestre async 256+255,
otto canary per alias con stringhe e ID, e un input oversize. Budget per run:
180 s, massimo 600 POST embedding, 550 tokenizzazioni, 1200 richieste metadati.
Ogni HTTP ha timeout massimo 20 s e risposta JSON limitata. Nessun retry.
Il fingerprint del catalogo esclude soltanto `data[].created`, generato dal
server a ogni lettura; identità, argomenti, build e profili rimangono fissati.

Soglie **invariate**: dimensione 768, maxabs `1e-5`, coseno minimo `0.999999`,
errore norma `1e-6`, proiezione dalla stessa risposta raw `1e-12`.
Cinque run finali passano canary, ordine/cardinalità, uso di entrambi gli alias,
identità e profili invariati. Ogni run invia gli ID ottenuti dal tokenizer,
senza una seconda tokenizzazione del testo nel client.

| Run | Wheel/policy | Pipeline 511 testi | POST pipeline | Metadati totali | Tokenizzazioni totali |
| --- | --- | ---: | ---: | ---: | ---: |
| A1 | 0.9.0 / attesa 0,5 s | 3,617 s | 13 | 72 | 528 |
| B2 | 0.10.0 / default 0 | 6,620 s | 18 | 68 | 528 |
| B3 | 0.10.0 / default 0 | 5,607 s | 18 | 68 | 528 |
| A2 | 0.9.0 / attesa 0,5 s | 4,898 s | 12 | 70 | 528 |
| C1 | 0.10.0 / attesa 0,5 s | 4,893 s | 16 | 64 | 528 |

I totali includono canary, oversize e controlli; ai POST pipeline si aggiungono
quattro POST canary per run. Le 528 tokenizzazioni sono 511 documenti, 16 canary
e un oversize. L'oversize misura esattamente 3002 token contro limite 2048 ed è
rifiutato senza POST embedding. Picco embedding 2 con massimo 1 per alias;
picco tokenizer 2. Nel wheel finale picco HTTP totale 5 entro tetto derivato 6;
scheduler e crediti risultano drenati al termine.

Tutti i 511 vettori di B2/B3/A2/C1 sono identici alla baseline A1: maxabs 0,
coseno minimo circa `0.9999999999999998`, massimo errore norma circa `2.22e-16`.
Ogni output è anche correlato alla propria risposta HTTP raw: errore massimo
circa `1.11e-16`, senza riordinamenti o duplicazioni.

La mediana ABBA è 4,258 s per A e 6,114 s per B, rapporto 1,436; supera il segnale
esplorativo di regressione del 20% fissato prima delle prove. **Non dimostra una
regressione isolata del motore né un miglioramento:** il LLM esterno era inattivo
nel controllo iniziale A1 ma risultava in elaborazione nei controlli B2/B3/A2.
C1 è un solo controllo di policy e non basta a stabilire una causa. I tempi
comprendono client, strumentazione, rete, coda e runtime; non sono tempi GPU.
Le somme dei tempi worker possono sovrapporsi e i relativi hook hanno
granularità diversa fra 0.9.0 e 0.10.0. Non vengono confrontate come tempo CPU.

Un tentativo preliminare B1, con wheel diverso e prima della correzione
preflight, completava 511 output numericamente validi ma esauriva il vecchio
budget di 200 metadati prima del postflight. È conservato come **incompleto**,
escluso dai confronti della candidata finale. Il protocollo registra la revisione
del limite finito a 1200 e l'aggiunta del singolo controllo C1. Nessuna soglia
numerica è stata cambiata per ottenere il risultato.

## API ed esempio consumer

Firme pubbliche preservate:

```python
adapter.prepare_texts(texts, *, input_type="document", timeout=None, cancel=None)
adapter.prepare_query(text, *, timeout=None, cancel=None)
adapter.embed_prepared(prepared, *, timeout=None, cancel=None)
adapter.embed_texts(texts, *, input_type="document", timeout=None, cancel=None)
adapter.astream_texts(windows, *, input_type="document", max_pending_windows=2,
                      timeout=None, cancel=None, close_source=False)
adapter.astream_prepared(windows, *, max_pending_windows=2, timeout=None,
                         cancel=None, close_source=False)
```

```python
from runtime_llama import build_embedding_adapter

adapter = build_embedding_adapter({
    "base_url": "http://SERVER:8080",
    "model": "embedding.gguf",
    "embedding_replica_discovery": True,
    "batch_size": 256,
    "embedding_request_parallelism": 2,
    "token_count_parallelism": 2,
    "embedding_queue_size": 2,
    "timeout": 120,
})

async def ingest(windows):
    try:
        async with adapter.astream_texts(windows, max_pending_windows=2) as stream:
            async for vectors in stream:
                consume(vectors)
    finally:
        await adapter.aclose()
```

Il consumer indica endpoint e modello logico; discovery verificata e scheduler
gestiscono le repliche utilizzabili. Non occorre dichiarare il numero di GPU.
API sync e preparazioni task-aware restano disponibili; splitting e identità
dei frammenti rimangono del consumer. Per i dettagli dei record pubblici e dei
limiti, vedere [contratti embedding](../EMBEDDING-CONTRACTS.md).

## Artefatto e limiti della consegna

`runtime_llama-0.10.0-py3-none-any.whl`, SHA-256:

```text
717de0acfdd6f8175dd956f1acb4d2096f072db3ac0afcc64c4729d3b18874f5
```

Log completi, JSON dei run, sidecar vettori, protocollo e manifest con hash
sono conservati nel bundle locale `llama-runtime-persistent-20261021`.
I dati di deployment e i percorsi personali non vengono pubblicati nella repo.
La qualifica funzionale passa; uno speedup isolato e l'accettazione applicativa
non sono dimostrati. Consumer, servizi e modelli restano invariati.
