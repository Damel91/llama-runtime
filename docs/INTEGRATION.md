# Contratto di integrazione — runtime-llama 0.5.1.dev1

La libreria usa il `llama-server` scelto dal consumer, locale o remoto.
Il gestore del server possiede build, processo, pesi, dispositivi, contesto e
slot. La libreria non avvia processi e non modifica i profili di caricamento.
Il consumer possiede l'esecuzione dei tool e la continuazione della conversazione.

Questo documento descrive API e responsabilità. Tutti gli alias, endpoint e
limiti degli esempi sono illustrativi: sostituirli con la propria configurazione.
Il repository si chiama `llama-runtime`; distribuzione `runtime-llama` e import
`runtime_llama` restano compatibili. La versione `0.5.0` aggiunge la discovery
pubblica tipizzata, conservando le API esistenti di generazione ed embedding.
Non dichiara una nuova qualifica live.

## Connessione e disponibilità

Per la generazione `stream_idle_timeout_sec=None` elimina solo la scadenza idle
SSE, anche durante prefill prima degli header o del primo token. Il default
resta 30 secondi; un timeout numerico per chiamata ripristina il limite.
Connessione, JSON, discovery ed embedding restano finiti. Stop interrompe il
socket della singola richiesta; la chiusura viene drenata prima del riuso della
sessione. Il bridge può usare il client owned con `trust_env=False`, evitando
session injection e accessi privati; un client iniettato resta del chiamante.
Vedi [STREAM-TIMEOUTS.md](STREAM-TIMEOUTS.md) per il pattern pubblico e i limiti.

Le factory costruiscono adapter senza rete. `load()` è un controllo di
disponibilità, equivalente a `is_ready()`, e non un comando amministrativo.
`health_status()` distingue raggiungibilità del server e disponibilità del
modello: un router sano può esporre un alias assente o non caricato.
`server_healthy` descrive il server; `model_ready` descrive l'alias o l'insieme
delle repliche richieste. Per compatibilità, nella generazione `healthy` descrive
il server e `backend_ready` il modello; negli embedding `healthy` descrive la
disponibilità delle repliche. I valori del contesto
osservato e degli slot restano distinti dai limiti configurati del client.

L'auto-load resta una policy del server scelto. Nessuna chiamata amministrativa
viene introdotta dal client; una richiesta di inferenza è comunque soggetta
alla policy del server. Per evitare caricamenti involontari, la qualifica live
controlla prima il catalogo e richiede alias già caricati.

API key opzionale con `api_key_env` o `api_key`; preferire la variabile d'ambiente.
TLS verificato per default. Non inserire credenziali nell'URL. Il client non
segue redirect né ripete automaticamente una richiesta fallita.

## Catalogo pubblico read-only

`discover_models(base_url, timeout=5.0)` restituisce `ModelCatalog`: `models`
conserva il catalogo osservato, `loaded_models` include solo il caricamento
provato. Le capacità `chat` ed `embeddings` restano `None` se non dichiarate.
L'API usa solo GET al catalogo e, se serve, alle proprietà root, senza selezionare
un modello. Ha limiti di risposta e cardinalità, timeout condiviso, cancellazione
e autenticazione esplicita. Il client posseduto ignora credenziali e proxy
ambientali. Contesto caricato, contesto di training e larghezza architetturale
restano osservazioni distinte con fonte. Vedi [MODEL-DISCOVERY.md](MODEL-DISCOVERY.md)
per firma completa, esempi, limiti e stati sconosciuti.

## Embedding: discovery automatica dal modello logico

Per le nuove integrazioni il consumer sceglie un modello logico, senza elenco
o conteggio di repliche. L'endpoint configurato, inclusa la porta, serve sia
discovery sia inferenza. Non aggiungere `/v1` alla `base_url`.

```python
from runtime_llama import build_embedding_adapter

embedding = build_embedding_adapter({
    "role_id": "retrieval",
    "base_url": "http://localhost:8080",
    "model": "embeddinggemma-2-BF16.gguf",
    "embedding_replica_discovery": True,
    "dimension": 768,
    "normalize": True,
    "batch_size": 256,
    "max_input_tokens": 2048,
    "token_count_parallelism": 2,
    "embedding_prefixes": {
        "query": "task: search result | query: ",
        "document": "title: none | text: ",
    },
})
try:
    vectors = embedding.embed_texts(["documento uno", "documento due"])
    query = embedding.embed_query("come si configura il runtime?")
finally:
    embedding.close()
```

Il nome logico è il basename pubblicato da Noxi in `noxi-model-name-v1`; si può
anche usare un alias ancora o `sha256:<digest contenuto>`. Un nome che risolve
identità diverse è ambiguo e viene rifiutato. La libreria usa solo istanze
caricate con tag validi di contenuto e compatibilità semantica uguali. Non
deduce equivalenza da path, suffissi o dimensione dei vettori e non scarica
pesi per calcolare hash nel client.

Noxi deve pubblicare `noxi-content-sha256-v1`, `noxi-embedding-sha256-v1` e
`noxi-model-name-v1` nei tag nativi `/v1/models`. Il primo deriva dai byte di
tutti gli shard e degli adattatori, attraverso un manifest canonico; non è un
hash del percorso. Non servono porte aggiuntive o patch del binario. Un server
precedente senza tag può servire un alias preciso come singola istanza non
verificata; non permette di formare automaticamente un pool verificato.

In modalità automatica ogni alias ha tetto fisso di una richiesta attiva.
`embedding_request_parallelism`, se omesso, vale 4 come bound globale; il
numero di repliche disponibili limita il parallelismo effettivo. Un valore
esplicito resta un tetto, non il numero di modelli da caricare. Senza discovery
il default storico resta 1. `embedding_queue_size` mantiene la coda limitata.
Non combinare discovery con `embedding_replicas` esplicite.

L'identità verificata si fissa per la vita dell'adapter. I cambi di topologia
vengono osservati dal catalogo; una sostituzione dei pesi non viene interpretata
come normale replica. Gli ID interni servono al routing, non all'identità di
indici/cache. L'adapter non amministra il caricamento. Una modifica del pool
attende la fine delle chiamate correnti con timeout/cancellazione/chiusura
rispettati, senza replay dei lavori già eseguiti.

Condividere un embedding adapter fra i chiamanti dello stesso consumer consente
un unico scheduler e limite per replica. Adapter indipendenti hanno scheduler
indipendenti; la libreria non introduce un servizio di coordinamento fra
processi consumer diversi.

Contesto allocato, GPU e Flash Attention restano separati dall'identità dei
pesi. I limiti effettivi della replica sono osservati e includono prefissi e
token speciali. Il valore 2048 dell'esempio è un budget client, non un comando
di caricamento. Il [contratto architetturale](AUTOMATIC-EMBEDDING-REPLICAS.md)
descrive attestazione del runtime, fallback e confini della verifica.

## Embedding: batch totale e preparazione esatta

`batch_size` è il budget totale di testi per finestra. Il solo scheduler
sceglie alias distinti e divide la finestra in quote bilanciate, limitate dal
parallelismo globale: 256/2 → 128 per alias, 256/4 → 64. Le quote con resto
differiscono al massimo di uno. Più finestre condividono la stessa coda.
Mantenere `batch_size=8` limita la finestra a otto testi complessivi; per le
finestre consumer da 256 configurare 256. Non è un limite token o un contesto.

Ogni quota può produrre più HTTP. Il packing usa il minimo dei budget token
configurati e del batch nativo osservato sul processo caricato, distinto da
contesto e microbatch. Un input sopra un budget noto non viene inviato da solo
come eccezione. Non viene troncato né spezzato dalla libreria.

```python
prepared = embedding.prepare_texts(texts, input_type="document", timeout=120)
for item in prepared:
    print(item.token_count, item.context_token_limit, item.batch_token_limit,
          item.max_input_tokens, item.fits_input)
# Il consumer spezza lossless gli oversize e prepara soltanto i nuovi frammenti.
vectors = embedding.embed_prepared(prepared, timeout=120)
query = embedding.prepare_query("come funziona?", timeout=30)
query_vector = embedding.embed_prepared([query], timeout=30)[0]
```

`PreparedEmbeddingInput` è pubblico e immutabile. Contiene il testo già
preparato e un conteggio esatto con `add_special=True, parse_special=True`;
`embed_prepared` non applica nuovamente il prefisso. `prepare_query("")` misura
anche prefisso e speciali. Liste vuote non effettuano I/O. La preparazione non
ammette l'input all'inferenza: restituisce anche i conteggi sopra limite.

I limiti esposti sono snapshot conservativi sulle repliche utilizzabili. `None`
significa ignoto; `fits_context`, `fits_request` e `fits_input` sono distinti.
L'ammissione verifica nuovamente identità e limiti caricati. Conteggi emessi dallo
stesso adapter si riusano se identità dei pesi, semantica e policy del tokenizer
sono verificate; un record alterato/estraneo viene rifiutato. Un fallback senza
attestazione o una build tokenizer ignota riconteggia. Il record non mantiene lease o processi attivi.

Preparazione e inferenza condividono scheduler e limiti di ammissione.
`token_count_parallelism` è un tetto globale per adapter. Il timeout è una
deadline complessiva della chiamata, incluse coda e preparazione; ogni HTTP usa
il tempo residuo. Cleanup e chiusura restano drenati e possono attendere il
trasporto. Nessuna garanzia sull'arresto immediato del calcolo remoto.

Firme e criteri completi: [contratti embedding](EMBEDDING-CONTRACTS.md).

## Embedding: pool esplicito e compatibilità precedente

```python
from runtime_llama import build_embedding_adapter

embedding = build_embedding_adapter({
    "role_id": "retrieval",
    "base_url": "http://localhost:8080",
    "model": "nomic-embed-text-v2-moe_0",
    "dimension": 768,
    "normalize": True,
    "embedding_prefixes": {
        "query": "search_query: ",
        "document": "search_document: ",
    },
    "batch_size": 8,
    "max_input_tokens": 512,
    "batch_token_limit": 480,
    "token_count_parallelism": 1,
    "embedding_request_parallelism": 4,
    "embedding_queue_size": 16,
    "embedding_replicas": [
        {"model": "nomic-embed-text-v2-moe_0", "max_in_flight": 2},
        {"model": "nomic-embed-text-v2-moe_1", "max_in_flight": 2},
    ],
})
try:
    vectors = embedding.embed_texts(["documento uno", "documento due"])
    query_vector = embedding.embed_query("dove trovo la configurazione?")
    limits = embedding.health_status()
finally:
    embedding.close()
```

Gli alias sono identificatori HTTP espliciti; il suffisso non prova l'identità
dei pesi o del dispositivo. Verificare equivalenza e configurazione prima di
usarli come repliche intercambiabili. Non si configurano GPU tramite il client.

Il numero di slot server resta indipendente dai tetti HTTP del client:
l'esempio consente quattro lavori globali e due per alias; il server può
eseguirli con una capacità diversa. Slot, contesto, batch, microbatch, backend
hardware e configurazione del processo possono influire sull'equivalenza
numerica osservata. Verificare lo stesso corpus su ogni replica e sul pool con
una tolleranza dichiarata; l'identità dei pesi non garantisce da sola risultati
numericamente equivalenti in qualsiasi configurazione. Non attribuire una
differenza a un kernel specifico quando cambiano insieme più parametri.
Le impostazioni del processo restano al gestore e non sono tutte attestabili
dai soli endpoint HTTP.

`embedding_request_parallelism` limita globalmente i lavori attivi nello stesso
adapter, anche fra più chiamanti. Il lavoro mantiene il proprio posto durante
osservazione `/props`, tokenizzazione, packing e richieste embedding. I tetti
per alias si applicano all'interno di questo limite. Il routing sceglie a turno
alias distinti per finestra e conserva l'affinità delle quote; non trasferisce automaticamente su un'altra replica
un lavoro fallito. Adapter distinti hanno scheduler distinti.

`embedding_queue_size` limita i lavori ammessi in attesa; zero applica attesa
diretta prima dell'ammissione. Una coda piena applica backpressure al chiamante.
Il default mantiene una sola richiesta attiva, senza introdurre concorrenza
nelle configurazioni esistenti. `token_count_parallelism` è un controllo
separato: non è il numero di richieste embedding né di slot GPU.

## Embedding: prefissi per modello e intento

`embedding_prefixes` appartiene al profilo del modello. Il consumer passa il
testo e dichiara l'intento; tutte le repliche dello stesso adapter ereditano
la preparazione comune. Modelli diversi usano profili/adapter distinti.

```python
# Configurazione per EmbeddingGemma 2:
embedding = build_embedding_adapter({
    "model": "embeddinggemma-2_0",
    "base_url": "http://localhost:8080",
    "max_input_tokens": 4096,
    "batch_token_limit": 4096,
    "embedding_prefixes": {
        "query": "task: search result | query: ",
        "document": "title: none | text: ",
    },
})
try:
    documents = embedding.embed_texts(["manuale del progetto", "istruzioni build"])
    query = embedding.embed_query("come si compila?")
    queries = embedding.embed_texts(["come si compila?", "quali GPU?"], input_type="query")
    prepared = embedding.embed_texts(
        ["title: Runtime | text: istruzioni build"], input_type="raw"
    )
finally:
    embedding.close()
```

| Chiamata | Preparazione |
| --- | --- |
| `embed_query(text)` | `query + text`, un solo vettore |
| `embed_texts(texts)` oppure `input_type="document"` | `document + text`, un vettore per input |
| `embed_texts(texts, input_type="query")` | `query + text`, batch di query |
| `embed_texts(texts, input_type="raw")` | Testi già preparati, senza prefisso aggiunto |

La concatenazione conserva esattamente spazi, newline e separatori configurati;
non aggiunge spazi e non interpreta placeholder. Non riconosce prefissi già
presenti: con un profilo abilitato, i testi preparati dal consumer richiedono
`raw`. Una query vuota mantiene il risultato `[]`; una lista vuota restituisce
`[]`. `input_type` non valido causa `ValueError` locale, anche con lista vuota.

I prefissi predefiniti sono vuoti: le configurazioni esistenti conservano
gli input precedenti. È disponibile anche il tipo pubblico immutabile
`EmbeddingPrefixes(query="...", document="...")`, utilizzabile al posto del
dizionario. Valori non stringa e chiavi diverse da `query`/`document` vengono
rifiutati. L'override del ruolo sostituisce l'intera coppia dei defaults:
un campo omesso torna vuoto. Prefissi non vuoti richiedono un profilo embedding.

La libreria non sceglie i prefissi dal nome dell'alias. La configurazione deve
seguire il modello e il task: [Nomic v2 MoE](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)
richiede `search_query: ` / `search_document: `; per ricerca documentale
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) usa
`task: search result | query: ` / `title: none | text: ` quando manca il titolo.
Titoli variabili o altri formati possono essere preparati esplicitamente con
`raw`; questo contratto configura prefissi letterali, senza template dinamici.

La preparazione avviene una sola volta, prima di tokenizzazione, packing e
scheduler. Il prefisso e i token speciali rientrano in `max_input_tokens` e nel
budget del batch: 8192 token di contesto non significano 8192 token di testo
più prefisso. Split adattivo e repliche riusano i testi completi già preparati.
Non viene troncato alcun testo e nessun parametro di caricamento viene modificato.

Cambiare i prefissi cambia la rappresentazione dei testi. Per un indice
esistente occorre riallineare la preparazione di query e documenti ed eventualmente
ricalcolare i vettori; `physical_identity` identifica il server/modello/contesto,
senza certificare l'identità semantica dell'indice o dei prefissi.

## Embedding: limiti e integrità

`batch_size` è il massimo totale di testi per finestra; `batch_token_limit`
limita i token complessivi per richiesta HTTP, insieme al batch nativo noto.
Ogni input resta indivisibile ed è confrontato con il minimo dei limiti di
contesto e dei budget richiesta noti. Un input che supera da solo un budget
noto viene rifiutato prima dell'invio, senza eccezioni singleton.
Una replica può dichiarare propri `max_input_tokens` e `batch_token_limit`,
che restringono i limiti comuni. Gli slot osservati non diventano una capacità
del batch. Nessun testo viene troncato implicitamente.

La risposta completa conserva ordine, cardinalità e dimensione; vettori
malformati o non finiti causano un errore. La dimensione appresa si pubblica
solo dopo la validazione dell'intera chiamata. Un errore o la cancellazione
arresta l'ammissione di nuovi lavori di quella chiamata e ne drena la pulizia,
senza restituire un risultato parziale come completo. Gli altri chiamanti
mantengono il proprio stato di cancellazione.

Solo HTTP 400 con tipo strutturato `exceed_context_size_error` autorizza lo
split adattivo del batch fallito sulla stessa replica. Le parti riuscite non
vengono ripetute. Un singolo input eccessivo e gli altri errori propagano.

## Thinking diagnostico, esplicitamente abilitato

```python
from runtime_llama import build_adapter, ThinkingDiagnosticsConfig

records = []
llm = build_adapter({
    "role_id": "coder",
    "model": "chat-model",
    "base_url": "http://localhost:8080",
    "num_predict": 256,
    "reasoning": "auto",
})
try:
    result = llm.generate_with_metadata(
        [{"role": "user", "content": "Quanto fa 17 per 23?"}],
        thinking_diagnostics=ThinkingDiagnosticsConfig(enabled=True, max_chars=256),
        thinking_sink=records.append,
    )
    print(result.text)
    diagnostic = result.thinking_diagnostic
finally:
    llm.close()
```

Il default è disabilitato. Si può impostare `thinking_diagnostics` nel profilo o
per chiamata, anche come dizionario `{"enabled": True, "max_chars": 256}`.
Il tetto riguarda solo la copia diagnostica: non modifica sampling, budget,
output visibile, parsing o protezioni runaway.

La snapshot finale contiene `call_id`, ruolo, modello, `sources`, testo,
conteggi osservati/catturati, `truncated`, `status` e `terminal_reason`.
`sources` distingue `separated` e `inline`. Il thinking inline viene rimosso
dal testo visibile; `reasoning_text` conserva il precedente significato di
solo reasoning separato. Una chiamata senza reasoning produce una diagnostica
vuota, senza inventare contenuto.

La diagnostica è su `LLMCallResult.thinking_diagnostic`, sull'evento terminale
di `stream_with_metadata()` e, dopo avvio della chiamata, sull'errore tipizzato
come `exc.thinking_diagnostic`. La chiusura anticipata del generatore consegna
la snapshot al sink con stato `interrupted`. Il sink riceve una sola snapshot
finale; un suo errore non sostituisce risultato o errore della generazione.
Gli stati sono `completed`, `interrupted`, `cancelled`, `runaway`, `error`;
il motivo terminale conserva il dettaglio dell'esito.

La cattura è per invocazione, anche usando lo stesso adapter da più thread.
`get_last_usage()` e `get_last_terminal_reason()` descrivono l'ultima chiamata
e non vanno usati per correlare invocazioni concorrenti.

Non vengono raccolti prompt, header o payload di richiesta. Le credenziali note
e le forme comuni di credenziali nel testo diagnostico vengono redatte; questo
non è un filtro universale di dati sensibili generati dal modello. Il logging
ordinario conserva solo conteggi. Il consumer decide se e dove persistere il
testo; `runtime_llama.logging.log_thinking_diagnostic` è un helper esplicito.

## Errori, cancellazione e chiusura

`LLMBackendError`, `LLMTimeoutError`, `LLMCancelledError` e `LLMRunawayError`
mantengono esiti distinti. `ClientHTTPError` espone stato e codici strutturati
limitati, senza copiare messaggi o corpi remoti nell'errore.

Gli adapter accettano `cancellation_probe` per generazione e `cancel` per
embedding. Chi interrompe uno stream deve chiudere il generatore, ad esempio
con `contextlib.closing`. La cancellazione del client non prova lo stop GPU
del server remoto. Il timeout HTTP e il timeout di attesa degli eventi non
costituiscono una deadline dell'intera conversazione.
Le callback di cancellazione possono essere chiamate da più worker e devono
essere prive di effetti collaterali e sicure fra thread, ad esempio `Event.is_set`.

Il client possiede sessioni HTTP separate per richieste concorrenti e le
riutilizza dopo la pulizia. `close()` drena i lettori e chiude le risorse
possedute; può attendere il timeout di una lettura attiva. Un `LlamaCppClient`
iniettato nell'adapter resta del chiamante. Una `requests.Session` iniettata
nel client resta del chiamante e il dispatch è serializzato per proteggerla;
gli altri utilizzatori esterni devono coordinarne l'accesso.

## Integrazione e verifica dei consumer

Il progetto usa Apache-2.0, copyright 2026 Davide Mele, con attribuzioni upstream
conservate in [NOTICE](../NOTICE) e licenza MIT del client derivato. Il probe
`probes/qualify_replicas_thinking.py` richiede opt-in live e registra build,
alias, limiti, equivalenza numerica, ingest e carico misto.

Il consumer sceglie la versione della libreria, configura endpoint/alias e
scheduler condiviso, collega eventuali sink e propaga gli errori. Deve poi
qualificare il proprio flusso, inclusi tool/MCP e autorizzazioni. Le prove della
libreria non certificano quel flusso. I [requisiti](REQUIREMENTS.md) e il
[README](../README.md#verification) descrivono i confini della verifica pubblica;
questa release non dichiara una nuova qualifica live o acceptance del consumer.
