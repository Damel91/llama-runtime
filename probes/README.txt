Probe operativi per runtime-llama
===============================

Eseguire dalla root del checkout con Python 3.11+ e dipendenze di sviluppo:

  python -m pip install '.[dev]'

In alternativa usare PYTHONPATH=src per qualificare i sorgenti del checkout.
I probe sono strumenti del repository, non API distribuite nel wheel.
Repository: llama-runtime; distribuzione Python: runtime-llama; import: runtime_llama.
La versione 0.5.0 aggiunge discovery pubblica read-only e mantiene le API di
inferenza esistenti. Non dichiara una nuova qualifica live o acceptance del consumer.

Sostituire negli esempi SERVER, LLM_ALIAS, EMBEDDING_ALIAS e BUILD con i valori
esatti del proprio server. I probe live richiedono modelli gia' caricati e non
eseguono caricamento, unload, riavvii o modifiche alla configurazione. L'auto-load
resta una politica del server. Non sono previsti retry automatici del probe.
Gli URL devono essere HTTP(S), privi di credenziali, query e fragment; questi
probe non configurano autenticazione HTTP; qualify_auto_discovery.py
accetta --api-key-env. Conservare ogni esecuzione in una
directory distinta: un nuovo run non sostituisce le evidenze di un fallimento.

Conversazioni lunghe e 24 turni reali
-----------------------------------

  python probes/qualify_long_conversations.py \
    --base-url http://SERVER:8080 --model LLM_ALIAS \
    --expected-build BUILD --output /tmp/long-run-01 --acknowledge-live

Il wrapper richiede i due helper e gli oracle JSON presenti nella stessa cartella.
Genera history sintetiche con almeno 24 coppie user/assistant e tool correlati,
con prompt di circa 8K, 32K e 64K token. Per ciascuna verifica recall e follow-up,
payload sul filo, identificatori tool, testo SSE e usage. Poi esegue 24 richieste
consecutive sullo stesso adapter, accumulando le risposte reali nella history.
Le history sintetiche e i turni realmente generati sono riportati separatamente.
Il modello deve supportare template, strumenti storici e contesto sufficienti.
Output massimo: 128 token nei casi lunghi, 32 token nei turni reali. Timeout idle:
900 secondi; budget della campagna: 45 minuti. Non qualifica il contesto massimo.
Exit 0 richiede il passaggio di ogni caso lungo e di tutti i 24 turni reali.

Ingest di 10.000 chunk
--------------------

  python probes/qualify_ingest.py --prepare \
    --base-url http://SERVER:8080 --model EMBEDDING_ALIAS \
    --expected-build BUILD --output /tmp/ingest-run-01
  python probes/qualify_ingest.py --run --acknowledge-live \
    --base-url http://SERVER:8080 --model EMBEDDING_ALIAS \
    --expected-build BUILD --output /tmp/ingest-run-01

--prepare genera corpus/oracle offline. Dichiarare gli stessi parametri in entrambi
i comandi. I default definiscono il profilo richiesto dalla prova: contesto per slot 512,
4 slot, dimensione 768, batch adapter 32, budget provider 480 token e 4 worker
tokenizer. Sono modificabili tramite i flag mostrati da --help; devono coincidere
con l'oracle preparato e i metadata osservati sul server. --source consente di
selezionare un checkout al posto del pacchetto installato.

Verifica ordine, conteggio, proiezione esatta della stessa risposta provider,
dimensioni, valori finiti e normalizzazione. Registra hash, batch, token, RSS e
8 query campione. Il budget in token puo' produrre batch effettivi piu' piccoli
di quello configurato. I vettori float64 occupano circa 61,4 MB con 768 dimensioni,
oltre a corpus, manifest e log. Le misure di risorse live richiedono POSIX.
L'oracle memoria riguarda questo carico finito, non l'assenza universale di leak.
L'audit passivo del trasporto registra classi d'errore, errno, frame e percorso,
senza header o contenuti. Il corpus e' sintetico e viene conservato separatamente.
Su errore restano il prefisso pubblicato e result.json; resume e retry del job
appartengono al chiamante. Non pubblica i risultati intermedi di una call fallita.

Recupero da errori controllati
----------------------------

  python probes/qualify_recovery.py --mode fixture --output /tmp/recovery-local-01
  python probes/qualify_recovery.py --mode live_proxy --acknowledge-live \
    --base-url http://SERVER:8080 --llm-model LLM_ALIAS \
    --embedding-model EMBEDDING_ALIAS --embedding-dimension 768 \
    --expected-build BUILD --output /tmp/recovery-live-01

fixture usa soltanto risposte sintetiche su socket loopback. live_proxy inoltra
le risposte valide del server esterno e inietta guasti sintetici dichiarati.
I 17 casi verificano riduzione e risalita batch, 401/429/500 senza retry, payload
invalidi, EOF, SSE malformato, disconnessione, usage e timeout. Il recupero e' una
nuova call esplicita sullo stesso adapter. Il proxy bufferizza lo stream upstream:
non misura TTFT o interruzioni reali a meta' stream. Non simula un riavvio effettivo
di Noxi, crash, OOM o guasti VPN/TLS; non prova l'arresto remoto dopo un timeout.

Repliche embedding e thinking per invocazione
--------------------------------------------

  python probes/qualify_replicas_thinking.py --live \
    --base-url http://SERVER:8080 --llm-model LLM_ALIAS \
    --replica EMBEDDING_ALIAS_0 --replica EMBEDDING_ALIAS_1 \
    --expected-build BUILD --output /tmp/replicas-thinking-01.json

Richiede due alias embedding caricati, dimensione 768, contesto per slot 512 e
4 slot per replica. Usa lo stesso corpus sintetico di 10.000 documenti per quattro
fasi: replica 0, replica 1, pool e pool con tre richieste LLM sovrapposte. Il client
limita globalmente a 4 lavori, con tetto 2 per alias nelle fasi pool. Il tokenizer
usa 1 worker per lavoro; batch 8 e budget 480 token. --count e --window modificano
la taglia del carico, con default 10.000 e 256; dichiarare la taglia effettiva.

Confronta tutti i vettori con il riferimento float64 della replica 0: errore
assoluto massimo 1e-5 e coseno minimo 0.999999, oltre a ordine/cardinalita',
dimensioni, finitezza e norma. Tre richieste con un solo input ancorano indipendentemente
posizioni del riferimento. Conserva corpus, hash, riferimento e audit delle
richieste nella sottocartella <nomeoutput>-artifacts (riferimento circa 61 MB).
I tetti osservati riguardano richieste client sovrapposte, non computazione GPU.
Il tempo end-to-end include preflight, tokenizzazione e validazione.
La proiezione di ogni vettore viene verificata anche contro il raw della stessa
risposta HTTP, con tolleranza 1e-12: distingue l'integrita' dell'adapter dalla
variabilita' numerica fra inferenze. Scostamenti dalle soglie fra repliche o
nelle ancore vengono registrati e la campagna prosegue per raccogliere tutte
le misure; il risultato finale resta failed. Errori strutturali interrompono
la prova. Non si abbassano automaticamente le soglie e non si ripetono richieste.

Le quattro prove thinking combinano reasoning separato/inline e API bloccante/
streaming, con tetto diagnostico 256 caratteri. Il modello/template deve supportare
reasoning_format deepseek/none. Il probe fallisce esplicitamente se il canale
richiesto manca, senza dichiarare supporto. Non prova stati di guasto remoti:
cancel, runaway, errori e close anticipato sono coperti dai test deterministici.
Il catalogo e le proprieta' dei modelli vengono controllati prima e dopo.
Senza --live il probe termina prima di importare la libreria o fare rete.

Discovery automatica delle repliche
----------------------------------

  python probes/qualify_auto_discovery.py --allow-live \
    --base-url http://SERVER:8080 --model MODEL_SELECTOR \
    --max-input-tokens 4096 --require-verified \
    --output /tmp/auto-discovery-01.json

MODEL_SELECTOR puo' essere il basename pubblicato nel tag Noxi, un alias preciso
o sha256:<digest contenuto>. Discovery e inferenza usano lo stesso endpoint.
La modalita' --require-verified si ferma prima dell'inferenza se mancano i tag
di contenuto/semantica. Senza quel vincolo, un alias preciso privo di tag puo'
operare come singola istanza seriale non verificata; non forma un pool.
Il probe invia quattro documenti brevi e una query attraverso un unico adapter
condiviso, verifica cardinalita', dimensioni, finitezza e norma e conserva
diagnostica prima/dopo. Non carica modelli, non misura la memoria e non certifica
qualita' retrieval o equivalenza numerica fra GPU. Exit2 indica un prerequisito
non soddisfatto; exit1 un errore; exit0 le verifiche funzionali superate.
--api-key-env legge una key opzionale dall'ambiente senza copiarla nel report.

Interpretazione delle evidenze
-----------------------------

I check locali, il comportamento dei modelli live e l'acceptance umana sono
distinti. Tempi osservati sotto carico misto non sono benchmark isolati. Il probe
non esegue consumer applicativi o tool MCP. Ogni bundle conserva oracle, sorgenti
effettivamente eseguiti e hash. Associare i risultati soltanto alla revisione e
alla configurazione effettivamente esercitate; non attribuirli automaticamente
a revisioni successive della libreria.

Contratti embedding: budget totale e preparazione esatta
------------------------------------------------------

  python probes/qualify_embedding_contracts.py --allow-live \
    --base-url http://SERVER:8080 --model MODEL_SELECTOR \
    --install /tmp/installed-runtime-llama-wheel \
    --timeout 60 --wall-timeout 240 --output /tmp/contracts-live-01.json

--install sceglie un'installazione wheel esterna al checkout. Il profilo limitato
richiede due repliche caricate con identita' verificata, dimensione 768,
contesto 2048, batch nativo 2048, uno slot e Flash Attention on. Si ferma se il profilo
cambia; non lo amministra. Massimo 551 output sintetici: finestra 256, caller
concorrenti 255 + 3 query,32 riferimenti diretti, confine 2048/2049 e paritytask.
Registra tokenizzazioni, richieste embedding, metadata, payload e overlap HTTP,
preparazione/wall time e cleanup; non sono misure GPU pure. Riutilizzo dei conteggi,
ordine, rifiuto di input oversize, proiezione raw e soglie numeriche rimangono rigorosi.
La finestra 256 deve usare due POST da 128 testi con somma token superiore a
2048: il batch nativo non e' un tetto aggregato HTTP. Nessun batch_token_limit
esplicito viene configurato da questa prova. I limiti pubblici devono rifletterlo.
Il .http.jsonl conserva hash dei testi/vettori e metadata compatti, non bulk vettori.
Usare un nome output nuovo per ogni run e conservare manifest e hash wheel.

Motore persistente: qualifica funzionale di un wheel installato
-------------------------------------------------------------

  /path/to/clean-env/bin/python probes/qualify_queue_engine.py --allow-live \
    --base-url http://SERVER:8080 --model MODEL_SELECTOR \
    --output /tmp/persistent-queue-01.json

Per verificare anche il riuso dopo il run principale, aggiungere --qualify-reuse.

Il probe richiede un wheel installato, due repliche verificate e dimensione768.
Non imposta contesto, slot o caricamento. Esegue otto canary stringhe/ID per
replica, un conteggio oversize con rifiuto prima dell'inferenza e511 documenti
sintetici in otto finestre64x7+63, max_pending_windows4, P2/Q2, batch_size256.
Usa il default di attesa zero senza override e richiede almeno un POST con
input provenienti da finestre diverse. Correlazioni tokenizer/ID/raw e indici
di finestra verificano cardinalita' e ordine sia interno sia fra finestre.
Soglie: maxabs1e-5, cosine0.999999, norma1e-6,
proiezione della stessa risposta1e-12. Registra richieste/payload/concorrenza,
preparazione e tempi client/HTTP, senza attribuirli al solo calcolo GPU.
I primi otto output della pipeline sono confrontati anche con i canary piccoli,
alle stesse soglie fra inferenze. La pipeline deve inviare esattamente511
tokenizzazioni per i testi nuovi e meno di50metadata, il conteggio della
baseline0.10.1; i contatori di pipeline e delle altre fasi restano separati.
"Freddo" significa511testi non presenti in cache, non modello o GPU freddi.
I canary nativi e il conteggio oversize di un testo diverso precedono comunque
la pipeline, come nella baseline; l'oversize puo' occupare una entry distinta.
Il report e il sidecar vettori riguardano il singolo candidato; nessun confronto
di throughput fra versioni. Uno LLM caricato e attivo e' ammesso e il suo stato
is_processing viene registrato quando leggibile. Limiti finiti per run:
180secondi,600HTTP embedding,
550tokenizzazioni e1200metadata; nessun retry o comando di amministrazione.
I fingerprint prima/dopo escludono soltanto il timestamp volatile created
del catalogo, conservando identita', argomenti, build e profilo caricati.

Con --qualify-reuse la prova511 resta per prima. Dopo, un altro adapter nuovo
prepara otto testi brevi una volta (otto tokenize), esegue due embed_prepared
e due embed_texts sugli stessi testi; tutte e quattro le inferenze devono
aggiungere zero tokenize. Le mappe raw sono separate per fase e verificano
ordine, ID, proiezione e confronto numerico fra i quattro passaggi. Sono
registrati occupancy/hit/miss della cache, drain su close e wall delle API e
dei hook di preparazione/issue/validazione dei record pubblici. Le somme dei
wall dei worker possono sovrapporsi e non sono elapsed, tempo CPU o tempo GPU.
Il riuso non e' un benchmark a cache fredda. Il totale atteso e'536tokenize
con riuso,528senza; restano i tetti globali550/600/1200e180secondi. La prova non
installa wheel, migra consumer o pubblica una release.
