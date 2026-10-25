# Piano di ottimizzazione embedding — 25 ottobre 2026

**Stato: progettazione, senza implementazione o nuove prove.** Piano concordato
fra la chat runtime e la chat CodingCastle: runtime possiede il filone scheduler;
CC ha revisionato limiti operativi, lifecycle e overhead dal lato consumer.
Flower MCP è l'autorità canonica del progetto `llama-runtime`; questo documento
ne è una proiezione leggibile. CC non deve usare Flower per contribuire alla
revisione: la registrazione del piano è responsabilità della chat runtime.

Registrati `REQ-000046`–`REQ-000060`, scenari `GOAL-000028`–`GOAL-000030` e
milestone `MILE-000029`–`MILE-000035`, tutte pianificate. L'architettura
**revisione 5 è proposta**, base 4; la revisione 4 resta adottata. Fingerprint
della proposta:
`d907fd6fce274664d04df22834d0c94bca72183d87e909a7490745130f725ab4`.
Nessun packet/run di implementazione è stato aperto per questo piano.

La rilettura `project_progress` conferma milestone `planned`, zero criteri
esecutivi e nessuna accettazione inferita. Segnala ancora `scope_complete=false`
e scope requisiti/criteri non pianificato nei packet: è il lavoro di dettaglio
esecutivo da aprire nel prossimo ciclo, non una dichiarazione di implementazione
completa. Requisiti, dipendenze ed exit policy delle milestone sono registrati.

L'autorizzazione attuale riguarda soltanto la progettazione coordinata. Il
server di inferenza è offline: non sono stati eseguiti test, probe o inferenze,
né prodotti nuovi wheel. Nessuna modifica a consumer, modelli, database, job,
servizi, configurazioni o installazioni. Nessun push o rilascio fa parte del
piano. Per l'esecuzione servirà un ciclo governato distinto.

## Baseline e diagnosi

Baseline repository `6443f2c1069a9cf6c9c0e275866b191fe5283bbc`; codice candidato
`4ad8c22111946bfb8236b474a74704323e1e79ca`, versione sorgente `0.13.1.dev2`.
Il wheel dev2 già verificato ha SHA-256
`c889feda2649ce8b9206f589923c702676d9b7c048538df7fe3a46baceb3b378`.
Questi identificano la baseline, non una nuova consegna.

| Evidenza precedente / lettura statica | Conseguenza progettuale |
| --- | --- |
| Dev2 ha rimosso il blocco riprodotto fra finestre dello stesso stream; la condivisione del catalogo per coorte esiste già | Preservare queste correzioni; non riproporle come lavoro nuovo |
| Nei profili precedenti rimangono intervalli fra POST di circa 150–210 ms; controlli diretti e SDK circa 60 ms ciascuno | Esaminare i confini di controllo senza attribuire tutto il tempo a rete, server o GPU |
| In un profilo split-GPU: 1,883 replica-secondi di attesa con metadati pendenti; CPU dispatch circa 0,03–0,05 s | Attese sovrapposte, non risparmio wall time promesso né prova di collo di bottiglia GIL |
| `P` costruisce pool HTTP, capacità del motore e vari limiti di preparazione/controllo; le props intervengono sul cap per alias | Togliere soltanto il controllo slot non elimina gli altri limiti accoppiati |
| Registrazioni sync e copie delle finestre possono precedere l'ammissione byte dei record | Portare la backpressure prima della ritenzione; è un rischio di copertura, non un leak misurato |
| I normali health/catalog/props omettono il limite risposta del client | Estendere il limite pre-decode a tutti i JSON di controllo |
| Pool working, risultati e cache sono distinti; esistono ulteriori transitori | Dichiarare unità e ownership, senza promettere un unico limite RSS |

La campagna precedente contiene 120 percorsi misurati riusciti; 36 hanno audit
esplicito sia assoluto sia coseno. Gli altri 84 non acquisiscono retroattivamente
l'audit mancante. Il controllo alternato non dimostra un miglioramento generale
stabile; `FIND-000007` resta aperto. Dettagli e limiti nella
[qualifica dev2](verification/STREAM-WINDOW-0.13.1.dev2.md).

## Filone A: scheduler basato sulle repliche

`N` conta destinazioni loaded canoniche, indipendentemente indirizzabili, della
stessa generazione **locale** di topologia verificata. Identità del contenuto
e semantica devono concordare. Sinonimi dello stesso elemento catalogo non
contano due volte; nome, suffisso, path o dimensione del vettore non provano
equivalenza o collocazione su GPU/processi diversi. Anche una lista esplicita
multi-alias richiede verifica. Un singolo alias legacy non verificabile resta
isolato, con capacità uno; una lista eterogenea/non verificabile fallisce prima
dell'ammissione. `N=0` significa indisponibilità.

La capacità potenziale originale diventa:

```text
automatico:             C = N
guardia positiva G:     C = min(N, G)
per replica:            una lease originale
concorrenza osservata:  <= C, subordinata a lavoro pronto e crediti risorse
```

`total_slots`, `n_batch` e `n_ubatch` restano osservazioni native: non
moltiplicano le POST ammesse. Una POST può già fornire più sequenze al server.
La libreria non assegna slot GPU e non amministra caricamenti/processi.

Le quote usano `N`, indipendentemente dalla guardia `G`. Con budget totale
`B=batch_size`, il cap testi condiviso è `ceil(B/min(B,N))`; una finestra con
meno testi usa al massimo `min(numero_testi,N)` destinazioni non vuote. I resti
ruotano e i crediti appartengono sempre alla finestra originaria. Le code
continuano ad accumulare record compatibili fra finestre senza un timer
obbligatorio. Limiti di byte/token espliciti possono richiedere ulteriori POST.

| Testi / budget totale | Repliche N | Guardia | Piano pieno di riferimento, in assenza di altri limiti |
| --- | --- | --- | --- |
| 256 / 256 | 2 | auto | Fino a 128 testi per replica, picco 2 POST |
| 256 / 256 | 4 | auto | Fino a 64 testi per replica, picco 4 POST |
| 2048 / 256 | 2 | auto | 16 POST originali da 128, picco 2 |
| 2048 / 256 | 2 | 1 | Le stesse 16 POST originali da 128, picco 1 e turni fra alias |

Il piano quando `G<N` cambia intenzionalmente rispetto al codice attuale.
L'ordine pubblico, la cardinalità e la correlazione dei risultati restano
invariati. La selezione equa deve coprire osservazione, ammissione e dispatch
degli alias: un ordine fisso del dizionario non deve affamare il secondo alias.
Il cursore di dispatch avanza soltanto dopo un invio ammesso ed è distinto
dalla rotazione delle quote e dai turni dei consumer. La fairness dev2 fra
finestre dello stesso stream resta un vincolo.

Una lease appartiene al job originale per tutto l'albero di isolamento:
root e discendenti sono seriali sul medesimo alias; gli altri alias possono
avanzare. Non si rilascia e riacquisisce il permesso per ogni figlio. Più POST
cumulative non significano maggiore concorrenza.

I worker HTTP seguono la capacità della generazione corrente. Sostituzione
topologia/executor alla frontiera di quiete, dopo drain dei trasporti e dei
riferimenti della generazione precedente. Una riduzione blocca nuovi ingressi,
senza revocare lease o risultati. Controllo e tokenizer conservano pool
limitati propri; discovery non ridimensiona il pool in cui sta eseguendo.
Nessun secondo scheduler o copia concorrente delle code ready.

### Configurazione e compatibilità previste

| Campo / contratto | Decisione del piano |
| --- | --- |
| `request_parallelism` automatico / `embedding_request_parallelism` dell'adapter | Assente o `None`: automatico; intero positivo: guardia G. I valori positivi persistiti non diventano implicitamente automatici |
| `EmbeddingReplicaConfig.max_in_flight` | Valore positivo ancora parsabile; effettivo uno, con richiesto/effettivo/motivo visibili. Compatibilità sintattica, cambiamento comportamentale anche per adapter espliciti |
| `batch_size` | Budget totale testi distribuito su N, non limite per GPU né contesto |
| Nuovo minimo `embedding_max_pending_owners` | Tetto globale H positivo. Auto-default risolto una volta da `(G configurata positiva, altrimenti seed compatibile 2) + queue_size`, minimo uno; mai dal numero scoperto di repliche |
| `max_pending_windows` | Sottotetto per stream che consuma H; non aggiunge un altro budget globale |
| Lookahead/preparazione | Target `min(H,max(1,C+queue_size))`; shrink senza revoca dei crediti esistenti |

Il seed 2 di H non descrive repliche o capacità HTTP. H resta lo stesso fra
generazioni locali, senza creare budget sovrapposti. Anche `N>H` è valido:
un owner può alimentare più alias; i diversi crediti limitano le risorse
effettive. I default proposti non sono ancora implementati.

## Filone B: limiti operativi e overhead, revisionato con CC

### Ownership, backpressure e memoria

| Famiglia | Ammissione / unità | Rilascio reale |
| --- | --- | --- |
| Owner sync/async e pull in corso | H globale prima di registrazione/pull; sottotetto stream | Consegna/trasferimento o cleanup effettivo, non il solo `Future.done()` |
| Snapshot input e record working | Crediti byte/riferimenti prima della copia; budget token preparati distinto | Fine dell'ownership del record, compreso il drain quando inviato |
| Risultati completati | Pool globale distinto; prenotazioni ordinate | Consegna, trasferimento o distruzione |
| Cache di preparazioni esatte | Pool LRU già limitato per entry/byte e identità/build/policy | Evizione/chiusura; non autorizza nuove inferenze |
| Handoff request/response | Pool globale separato, derivato numericamente dal budget byte esistente; prima del POST/read | Trasferimento atomico ai crediti risultato o distruzione, oltre la fine HTTP se necessario |
| JSON di controllo/tokenizer/embedding | Limite finito per body, anche decompresso, prima del decode | Drain e rilascio del body/oggetto posseduto |

Non si sottrae il credito risposta dal working pool già saturo: i record pronti
potrebbero richiedere l'invio proprio per liberarlo, creando un deadlock.
Gli snapshot devono lasciare headroom di progresso oppure fallire come input
impossibile. Cleanup non richiede nuovi crediti o un worker bloccato in admission.
L'ordine di acquisizione e una riserva di progresso per controllo/recovery
devono impedire che tutte le handoff embedding blocchino le osservazioni
necessarie a terminare il lavoro che trattiene quelle stesse risorse.
Questo vincolo è esplicito in `REQ-000051`, revisione 2: verificare anche
risposte di owner successivi che occupano handoff mentre un owner precedente
attende un'osservazione fresca. Nessuna attesa di nuovi crediti può trattenere
le risorse necessarie a produrli; è ammessa una politica equivalente se ne
viene dimostrata l'assenza di cicli.
Il bootstrap a dimensione vettore ignota usa un envelope conservativo finito e
può ridurre la concorrenza. Più repliche non aumentano automaticamente i pool.

La libreria acquisisce il count credit prima del pull async, senza un pull
speculativo per conoscere la dimensione. Valida incrementalmente una sequenza
finita con una regola di snapshot limitato, prima di una copia illimitata.
Non rilegge una sorgente mutabile già ammessa. Pressione aggregata: attesa equa,
cancellabile, sotto deadline; input indivisibile impossibile: errore tipizzato
con risorsa, limite, richiesto quando noto e rimedio. Memoria della sorgente o
del chiamante e iterazione non cooperativa restano esterne alla garanzia.

Byte sul filo, body decompresso, oggetti Python, cache e RSS non sono la stessa
unità. Il piano richiede un envelope conservativo di materializzazione e
contatori separati, non una promessa che un body di 8 MiB occupi 8 MiB di RAM.

### Input, payload, timeout e recupero

Il controllo del payload usa la serializzazione effettiva, inclusi Unicode,
escape, prefisso una volta, nome modello ed envelope. Riusare encoding e
conteggi locali immutabili evita lavoro duplicato, senza stime caratteri/token
presentate come conteggi esatti. Il percorso automatico diretto resta senza
`/tokenize` ordinario o calibrazione implicita.

Restano distinti quota testi, contesto per input, budget HTTP token esplicito,
byte JSON, prenotazioni di coda e limiti nativi. **Non cambia in questo piano
l'allowance diretta attuale basata su `min(context,n_ubatch)`**: rivalutarne la
correttezza richiede una decisione e prove separate, senza collegarla al cap
HTTP. Margini categoria 80% / mixed 90% e raccomandazione calibrata persistita
restano invariati. Una prenotazione non è consumo token misurato.

Il 500 conserva l'isolamento limitato a sottobatch e l'individuazione dei
chunk da far dividere al consumer, senza ritokenizzare l'intera ingestione,
riprodurre i successi o cambiare la media. Il solo 500 non prova un overflow
né che il server non abbia lavorato. Nessun retry generalizzato di POST ambigue.

Deadline monotona originale attraverso tutte le fasi, senza reset a ogni GET
o figlio. Readiness e timeout della singola finestra restano distinti dal
tempo complessivo dell'ingestione. Deadline del trasporto condiviso fissata
al dispatch sui partecipanti iniziali; scadenza/Stop di uno ritira soltanto
quell'owner. La chiusura di tutti o dell'adapter ferma e drena il lavoro reale.
Stato `draining` e crediti trattenuti finché la pulizia termina; preservare
l'eccezione primaria anche con interruzioni ripetute. Non promettere arresto
GPU remoto o close riuscita entro un tempo rigido per client/sorgenti arbitrari
che ignorano la cancellazione.

### Chiamate di controllo native

Tutti gli health/catalog/props, error body e controlli di recovery devono avere
un limite finito pre-decode, preferendo `embedding_max_response_bytes` anziché
un altro parametro ridondante. Client iniettati: capacità di lettura e ownership
esplicite, senza retry `TypeError` che rimuova silenziosamente il limite.

Restano le verifiche fresche di intero pool, identità, build e limiti dopo
attese e ai confini POST/recovery. Il riuso locale richiede provenienza
endpoint/alias/scope/autorità e stessa coorte congelata. Una risposta tardiva
non può riabilitare una generazione revocata. Nessuna cache TTL, ereditarietà
per owner arrivati dopo o digest dei pesi usato come attestazione del contesto.

Ottimizzazione nativa concreta: in un confine **singleton statusless eleggibile**,
la stessa risposta root props può soddisfare anche selected props:
catalogo + due props diventano catalogo + un props. Una GET risparmiata per
confine eleggibile; zero controlli necessari rimossi su router o casi diversi.
Non si attribuisce questo duplicato alle 19 o 35–36 GET dei profili precedenti.
I controlli restano fuori dai lock e non devono ricominciare continuamente
per arrivi che possono essere ammessi in una coorte successiva.

La policy `reuse` esiste già: eventuale qualifica separata, nessun cambio
implicito del default `close` e nessuna patch obbligatoria a llama.cpp.

### Opzione separata: osservazione coerente e generazione remota

`REQ-000060` / `MILE-000035` sono opzionali e non bloccano il percorso nativo.
Uno snapshot completo potrebbe sostituire catalogo + R props con una risposta,
solo con capability/versione e copertura esplicite. Uno snapshot GET non è
una lease atomica sul successivo POST. Per riuso fra POST serve un contratto
più forte, validato **dal server all'ammissione**: expected loading generation
o lease, rifiuto stale prima dell'inferenza, restart/ABA, alias reuse, build,
configurazione, scadenza, revoca, scope auth e drain definiti.

Assenza della capability prima dell'uso: percorso nativo fresco. Esito POST
ambiguo: nessun reinvio come fallback. Il client nativo da solo non chiude
il race fra osservazione e POST e non può garantire assenza assoluta di
autoload se il server lo applica dopo un unload concorrente. Politica di
caricamento al server/operator. Nessun nuovo endpoint o modifica Noxi è
richiesto o autorizzato ora.

## Requisiti canonici

Gli statement completi e la provenienza della richiesta sono nel ledger Flower.

| Filone | Requisito | Obbligo osservabile |
| --- | --- | --- |
| A1 | REQ-000046 | Pool loaded omogeneo verificato, compresi alias espliciti; niente capacità da sinonimi |
| A2 | REQ-000047 | Una lease originale per replica; auto N / guardia positiva; recovery e drain posseduti |
| A3 | REQ-000048 | Quote e cap testi da N indipendenti da G, resti/ordine/crediti invariati |
| A4 | REQ-000049 | Worker coerenti con capacità corrente e cambi generazione drenati; pool controllo separato |
| A5 | REQ-000050 | Fairness alias anche con G<N e conservazione del progresso fra finestre dev2 |
| B1 | REQ-000051 | Ledger risorse globale per famiglia, handoff separato, nessun deadlock o promessa RSS |
| B2 | REQ-000052 | Intake globale sync/async e snapshot limitati prima della ritenzione |
| B3 | REQ-000053 | Fit, payload, prenotazioni e limiti nativi separati; calibrazione/recovery conservate |
| B4 | REQ-000054 | Ogni JSON di controllo/errori limitato prima del decode/decompressione materializzata |
| B5 | REQ-000055 | Deadline monotone, Stop isolato per owner, chiusura e rilascio realmente drenati |
| B6 | REQ-000056 | Riduzione dei soli duplicati nativi provati con provenienza e freschezza preservate |
| B7 | REQ-000057 | Diagnostica limitata per fasi, risorse, lease originali e POST discendenti |
| Q1 | REQ-000058 | Matrice deterministica, verifiche repo e wheel riproducibile prima della qualifica live |
| Q2 | REQ-000059 | Misure live attribuibili e successiva accettazione CC in ciclo separato |
| C1 opzionale | REQ-000060 | Contratto snapshot/generazione imposto dal server, nativo sempre supportato |

## Milestone e dipendenze

Ogni requisito ha una sola milestone realizzativa. I prerequisiti registrati
sono requisiti di milestone precedenti, disgiunti dal proprio scope; la loro
registrazione non ne prova completamento o accettazione.

| Milestone | Scope posseduto | Dipendenze dirette | Criterio di uscita futuro |
| --- | --- | --- | --- |
| MILE-000029 — fondazioni | A1, B1, B3 | Nessuna nuova milestone | Tabelle identità/ownership/risorse e migrazione concordate, politiche input/calibrazione congelate |
| MILE-000030 — scheduler | A2–A5 | MILE-000029 | Capacità e quote corrette, fairness e generazioni drenate provate deterministicamente |
| MILE-000031 — intake e JSON | B2, B4 | B1/B3 di MILE-000029 | Crediti prima della ritenzione e tutti i JSON posseduti limitati; casi avversi coperti |
| MILE-000032 — join lifecycle/controlli | B5, B6 | MILE-000030 e MILE-000031 | Race Stop/deadline/drain coperte e call-count nativi ridotti solo nei confini eleggibili |
| MILE-000033 — diagnostica e candidato offline | B7, Q1 | MILE-000032 | Suite/check completi, metriche verificabili, wheel/versione/hash e smoke in ambiente pulito |
| MILE-000034 — live e handoff | Q2 | MILE-000033 | Server disponibile e autorizzazione futura; qualifica interleaved, residuali e contratto consumer consegnati |
| MILE-000035 — capability server opzionale | C1 | B6 di MILE-000032 e B7 di MILE-000033 | Autorizzazione separata del proprietario server; contratto versionato e fallback sicuro verificati |

MILE-000030 e MILE-000031 possono procedere in parallelo dopo le fondazioni;
MILE-000032 è il punto di ricongiungimento. MILE-000035 non è prerequisito
di MILE-000033/000034. Nessuna fase è accettata automaticamente dal piano.

## Piano di verifica futura

| Area | Casi deterministici minimi / asserzioni |
| --- | --- |
| Capacità | N=0/1/2/4 e >4, slot variabili/ignoti; max una lease per alias e C globale, nessun cap nascosto dal vecchio default |
| Quote e fairness | Guardia 1 con N>1, resti dispari, meno testi che repliche, molte finestre/consumer; nessuna starvation, ordine/cardinalità, nessuna duplicazione |
| Identità/topologia | Alias sinonimi, lista esplicita discordante, reload stessi hash con contesto/build diversi, peer cambiato, risposte tardive, shrink/growth con risultati trattenuti |
| Intake | Sync concorrenti e più stream, queue zero, consumer in pausa, early break, cancel in attesa, finestra eccessiva/sequenza mutabile; nessun pull extra o copia non ammessa |
| Memoria | Budget identici con N crescente, vettori grandi/ignoti, cache, risultati trattenuti, saturazione working/handoff, osservazioni del primo owner mentre gli owner successivi trattengono handoff, nessun ciclo di crediti o rilascio precoce |
| Payload | Byte limite/limite+1, Unicode/escape/prefissi/envelope, input indivisibile; conteggio uguale ai byte effettivamente inviati |
| JSON/trasporto | Errori/body grandi, gzip/chunked/slow-drip, framing troncato, malformed; limite pre-materializzazione e cleanup, client iniettato senza fallback non limitato |
| Recovery | Albero limitato, discendenti seriali, cancel parziale, successi conservati, errori non-500/ambigui non riprodotti |
| Deadline/chiusura | Ogni fase, scadenze owner diverse, uno/tutti cancellati, KeyboardInterrupt ripetuti, close durante refresh/recovery; lease e riferimenti fino al drain |
| Controlli | Call-count per fase, coorti nuove/congelate, singleton eleggibile/non eleggibile/router, alias lento; nessuna verifica fresca necessaria rimossa |
| Regressioni | Test dev2 su prefetch diverso, finestra più vecchia, accumulatori incrementali, token-ID/preparazioni sigillate e percorso diretto senza tokenizer |

Poi: controlli completi prescritti da `AGENTS.md`, build e install-smoke del
wheel fuori dall'albero sorgente. Nessuno di questi è stato eseguito ora.

La futura campagna live userà corpus congelato privato senza copiarlo nel
repository/distribuzione, identità e profili caricati invariati, soglie
`max_abs <= 1e-5` e `cosine >= 0.999999`. Repliche disponibili e loro collocazione
saranno dichiarate dall'operatore, senza ricaricarle tramite libreria.
Ripetizioni e regola di giudizio latenze saranno fissate prima di misurare;
versioni alternate e trace individuali, non soltanto mediane favorevoli.
Separare scheduler, limiti, osservazioni e policy connessione per attribuire
gli effetti. Se il piano cambia, il replay nativo usa ciascun payload SDK reale.

Misure: POST originali/figlie, GET per endpoint e motivo, byte, testi per POST,
capacità richiesta/effettiva, usato/limite/picco e causa dell'attesa. Tempi di
admission/preparazione/metadata/HTTP/output/drain, unioni wall time, CPU e
replica-secondi separati. Non chiamarli puro tempo GPU e non promettere una
riduzione del numero totale di POST quando il nuovo contratto cambia le quote.
Diagnostica limitata, senza testi, array token, credenziali o percorsi privati.

## Confine con CodingCastle e consegna futura

Il runtime possiede discovery, repliche, batching, richieste HTTP, code,
risorse e cleanup. CC conserva identità e intervalli dei frammenti, splitting
lossless, AST/grafo, budget caratteri persistito e provenienza, scritture e
deadline/cancellazione del job. Nessun numero di repliche o slot da passare
al consumer; un'eventuale guardia positiva già configurata resta rispettata.

Al termine dell'implementazione futura: firme API/config aggiornate, nota sui
cambiamenti intenzionali, esempio consumer, semantica batch, wheel/versione/
SHA-256, risultati deterministici e live con residuali. L'integrazione e
l'accettazione dell'ingestione CC restano nella sua chat e nel suo ciclo,
con autorizzazione propria; readiness strutturale e disponibilità semantica
restano distinte. Nessuna re-ingestione pulita è autorizzata da questo piano.

Riferimenti: [contratto embedding](EMBEDDING-CONTRACTS.md),
[concorrenza e diagnostica](CONCURRENCY-AND-DIAGNOSTICS.md),
[finestre async](ASYNC-EMBEDDING-WINDOWS.md),
[discovery repliche](AUTOMATIC-EMBEDDING-REPLICAS.md),
[HTTP nativo](NATIVE-HTTP.md). Le pagine contrattuali descrivono il codice
esistente; questo documento descrive la proposta futura.
