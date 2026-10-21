# Progetto del motore embedding a code persistenti

Proposta del 21 ottobre 2026 sulla baseline `f94734f`, candidata locale 0.9.0.
Il progetto sostituisce il coordinamento per finestra con un motore persistente
dell'adapter: preparazione concorrente, code per replica e composizione dei
batch quando esiste capacità di esecuzione. Le API Python restano compatibili;
llama.cpp continua a possedere tokenizzazione nativa e inferenza.

Questa è una proposta di architettura, requisiti e verifiche, non una nuova
implementazione o qualifica. L'acquisizione nel progetto Flower `llama-runtime`
resta pendente: i tool non sono esposti nella sessione e l'endpoint configurato
ha rifiutato la connessione. Non sono assegnati identificativi Flower locali.

## Diagnosi della pipeline attuale

La preparazione è già progressiva: `_prepare_job` pubblica ciascun record
appena pronto. Gli executor HTTP, preparazione e tokenizer hanno già thread
persistenti; la bridge async aggiunge coordinatori per le finestre ammesse.

Il limite è l'ownership del lavoro. `_execute` conserva quote e record pronti
per chiamata; `_ready_indices` sceglie gli indici prima di `try_submit`.
Lo scheduler può quindi accodare un batch già fissato mentre la replica è
occupata. I record che diventano pronti durante quell'attesa non possono più
entrarvi. Il timer di coalescing compensa in parte questa organizzazione.

Riferimenti: [adapter](../src/runtime_llama/adapters/embedding.py),
[scheduler](../src/runtime_llama/embedding_scheduler.py),
[bridge async](../src/runtime_llama/embedding_stream.py),
[contratto attuale](PROGRESSIVE-EMBEDDING.md).

## Principio ripreso da TEI

TEI separa la preparazione dei singoli input dalla coda di inferenza; il suo
batcher attende capacità nel canale del backend prima di estrarre il batch.
La coda applica budget di token e richieste. Questi principi sono adattabili
al client senza introdurre il server TEI o copiarne il backend.

Fonti consultate il 21 ottobre 2026:
[batcher TEI](https://github.com/huggingface/text-embeddings-inference/blob/main/core/src/infer.rs),
[coda TEI](https://github.com/huggingface/text-embeddings-inference/blob/main/core/src/queue.rs).

La differenza è il confine HTTP: la libreria compone richieste remote, mentre
TEI compone batch del proprio backend. La libreria non può trasformare uno
slot llama.cpp in più slot GPU. Una POST con molti testi riduce overhead HTTP;
il server decide come eseguire i task corrispondenti. Slot configurati e
concorrenza HTTP non dimostrano simultaneità o saturazione GPU.

## Motore e thread

Un solo motore possiede ammissione, record, code, crediti e lease HTTP per
l'intera vita dell'adapter. Le finestre delimitano i risultati pubblici;
non creano nuovi scheduler o code di durata limitata alla chiamata.

| Risorsa | Organizzazione proposta |
| --- | --- |
| Coordinamento | Un thread a eventi con `Condition`; nessuna scansione periodica per ogni finestra |
| Preparazione | Pool persistente di worker per le chiamate al tokenizer remoto, limitato da `token_count_parallelism` |
| Repliche | Una coda persistente per ogni istanza remota verificata, con sottocode dei proprietari delle richieste |
| Inferenza HTTP | Pool persistente limitato da `embedding_request_parallelism`; i worker eseguono lease assegnati dal solo scheduler |
| Metadati | Lavoro di controllo limitato, fuori dai lock delle code; nessun thread di controllo per input |
| API sincrona | Registra una richiesta e attende il suo risultato |
| API asincrona | Registra nello stesso motore e collega il risultato all'event loop, senza un coordinatore bloccante per finestra |

Una coda per replica è una struttura dati con capacità osservata, non un
executor indipendente. Con due repliche a uno slot e parallelismo globale due,
due worker possono attendere due POST distinti contemporaneamente. Con più
repliche dei worker, il motore le serve a rotazione entro il limite globale.

I thread sono adatti al lavoro prevalente di attesa di rete. Il GIL del normale
CPython limita invece il parallelismo del codice Python che usa intensamente
la CPU: serializzazione JSON e gestione dei record non diventano parallelismo
CPU illimitato. Non è richiesto un interprete free-threaded. Un eventuale
intervento nativo sul lavoro CPU richiede prima un profilo misurato.
[Documentazione Python](https://docs.python.org/3/library/threading.html#gil-and-performance-considerations).

Il reader cancellabile del trasporto esistente resta valido. Il totale dei
thread comprende anche quei reader, limitati dalle HTTP realmente attive;
non coincide soltanto con la dimensione del pool embedding. Worker persistenti
non implicano connessioni persistenti: resta la policy HTTP nativa senza patch.

## Proprietà dei record e distribuzione

Ogni record interno porta proprietario, indice di finestra e di input, task,
deadline, generazione del pool, destinazioni ammesse, identità dei pesi e
semantica, build tokenizer, policy speciali/prefissi, conteggio e ID esatti.
Un solo componente ne possiede lo stato a ogni transizione:

`ammesso -> in preparazione -> pronto -> riservato -> inviato -> completato`

Cancellazione o fallimento sono terminali alternativi. Un record già inviato
non ritorna nella coda dei pronti; ogni risultato viene associato una sola volta
all'indice originale. L'ordine di completamento può essere diverso da quello
di consegna. La consegna pubblica conserva ordine, cardinalità e finestre
complete, senza risultati parziali presentati come successi.

`batch_size` resta il budget totale di testi di una finestra di distribuzione.
Si conserva la scelta di alias utilizzabili e la divisione con `divmod` dello
scheduler attuale: 256 su due destinazioni assegna 128 crediti ciascuna; 256
su quattro ne assegna 64. I resti ruotano. Il numero selezionato resta limitato
da testi, alias utilizzabili e limite globale come nel contratto corrente.

I crediti sono limiti di distribuzione, non batch HTTP già composti. Le code
persistenti possono contenere lavoro di più finestre ammesse. L'assegnazione
dei record non ancora preparati può avvenire quando serve alimentare una
replica, rispettando i crediti residui. Il riuso di una preparazione su un'altra
replica richiede identità, semantica, build/tokenizer e policy compatibili.
Nessun trasferimento avviene dopo l'invio.

Ogni record consuma i crediti della propria finestra d'origine. Il tetto di
testi di una POST deriva da una singola quota del piano selezionato per quel
turno: aggregare code di finestre successive non somma i loro tetti. Nel caso
256 su due repliche, il tetto derivato rimane quindi 128 testi per POST, anche
quando il backlog contiene più finestre. Limiti di token/byte possono ridurlo.

Questo progetto conserva le quote bilanciate: non introduce una redistribuzione
silenziosa che assegni 200 dei 256 testi alla GPU più veloce. Un riequilibrio
può utilizzare soltanto crediti ancora disponibili e record non inviati.
Una replica che ha esaurito la propria quota può quindi terminare prima;
garantire occupazione continua richiederebbe un diverso contratto di distribuzione.

## Composizione al momento del dispatch

Il motore considera soltanto repliche con dati pronti, capacità locale e
capacità globale disponibili. Le osservazioni remote necessarie avvengono
fuori dal lock e non trattengono un permesso embedding durante attese di
preparazione o discovery. Dopo attese viene rivalidato lo snapshot richiesto
dal contratto di identità e limiti.

Il coordinatore invia i GET a worker di controllo limitati e riprende su evento:
non esegue HTTP né attende Future sincrone. Un `/props` lento sospende soltanto
le selezioni che dipendono da quell'osservazione. La capacità di ogni replica è
il minimo tra slot osservati, limite esplicito per replica e tetto globale;
slot ignoti o non validi autorizzano soltanto una richiesta.

Sotto un lock breve il motore sceglie una sottocoda equamente, estrae i record
pronti compatibili e riserva atomicamente capacità locale/globale. Il batch è
composto in quel momento. Serializzazione del payload, HTTP, callback utente,
completamento delle Future e `join` avvengono fuori dal lock. Un lease
corrisponde a una singola POST; un limite ridotto prima dell'invio richiede
nuova selezione o suddivisione dei soli record non inviati.

Mentre una replica è occupata, preparazione e pubblicazione continuano entro
i budget. Al suo rilascio il batch successivo comprende anche i record pronti
arrivati nel frattempo. Non vengono accodate closure contenenti indici già
congelati. Le rotazioni tra repliche e tra proprietari pronti impediscono a
un'unica ingestione di ottenere tutti i turni; un POST già attivo non è
preemptibile per favorire una query sopraggiunta.

Il dispatch ordinario non impone attese per riempire il batch: se una replica
è libera esegue il lavoro pronto. Questa policy può produrre batch piccoli
quando la preparazione è più lenta dell'inferenza. Una coda persistente non
garantisce contemporaneamente batch pieni e assenza di attese con input sporadici.

La proposta estende `embedding_batch_wait_sec` ad accettare zero per questa
policy e propone zero come nuovo default, con nota di migrazione. Nella 0.9.0
il campo accetta solo valori positivi e il default è 0.5: zero non è ancora
configurabile. Valori positivi espliciti resterebbero una policy opzionale del
medesimo motore, senza percorso legacy separato. Deadline e timeout finiti
rimangono indipendenti dall'attesa facoltativa di coalescing.

## Limiti separati e pressione sul server

Il packing applica separatamente numero di testi, limite individuale effettivo,
budget aggregato HTTP esplicitamente configurato e tetto del payload.
`n_batch` non diventa il budget aggregato HTTP. `n_ubatch` è un vincolo per
input soltanto quando il contratto osservato del backend/modello lo giustifica;
non viene dedotto genericamente per ogni architettura.

Preparazione esatta e limiti conservano il prefisso una volta sola e gli stessi
token speciali dell'embedding. Le preparazioni pubbliche rimangono immutabili
e sigillate; il percorso interno conserva record più leggeri. Gli ID verificati
alimentano il percorso embedding nativo già qualificato, senza una seconda
tokenizzazione delle stringhe. Nessuna stima caratteri/token autorizza l'inferenza.

Il tokenizer remoto usato oggi richiede chiamate per i singoli testi distinti:
questa rifattorizzazione non inventa un endpoint bulk e non elimina quei POST.
Può sovrapporli e limitare il lavoro pendente. Duplicati condividono il risultato
solo in un ambito limitato con identità e policy verificate; nessuna cache
globale dei contenuti. Il parallelismo tokenizer va misurato perché compete
con le altre richieste sul server.

Il limite globale embedding continua a riguardare le POST embedding. Tokenizer
e metadati hanno limiti propri, visibili in diagnostica. Il motore deve inoltre
avere un tetto esplicito delle HTTP totali; la sua configurazione pubblica sarà
definita nel packet di implementazione. Un'unica autorità assegna tutti questi
budget. Il controllo conserva possibilità di avanzare quando l'inferenza e la
preparazione sono sature, evitando deadlock per permessi annidati.

## Memoria e backpressure

I limiti coprono input pendenti, byte delle copie di testo possedute dalla
libreria, token/ID preparati, payload e risposte temporanee, più risultati
asincroni completati ma non consegnati. La capacità della coda non viene
affidata soltanto al numero di finestre. Le tuple di interi Python non valgono
quattro byte per token: l'accounting deve includere contenitori e rappresentazioni
temporanee con margini conservativi.

Prima della tokenizzazione si riserva il budget raw e il massimo buffer
temporaneo ammesso per quel lavoro. La pubblicazione trasferisce atomicamente
il record al budget preparato; una fase non trattiene un permesso aspettando
indefinitamente un'altra fase che necessita dello stesso permesso. Risposte
tokenizer/JSON e singoli input hanno tetti di risorse espliciti e fallimenti
tipizzati quando li superano.

Un input oltre il contesto rimane misurabile con `prepare_*`, entro quei tetti
di risorse, ma non entra nella coda di inferenza. La memoria dei dati originali
del chiamante e la lista completa dei vettori restituita dall'API sincrona
restano proporzionali al suo input. La backpressure limita i buffer intermedi
posseduti dal motore; non promette memoria costante per qualsiasi chiamata.
Le finestre async conservano anche il loro limite di intake e di risultati
non consumati, prima di leggere altre finestre dalla sorgente.

## Cancellazione e cambio di topologia

Le sottocode distinguono domini con cancellazione e deadline indipendenti.
Nel primo progetto una POST non mescola domini indipendenti. Due finestre
pubbliche async hanno deadline proprie anche nello stesso stream; non vengono
fuse automaticamente. Finestre interne della stessa chiamata possono invece
essere aggregate soltanto rispettando crediti e budget del batch selezionato.
Questo conserva Stop isolato senza introdurre un nuovo contratto di trasporto
condiviso tra chiamanti.

Un lease conserva capacità e ownership fino al ritorno e al drain del
trasporto, anche dopo la cancellazione del waiter. Un fallimento arresta e
drena il proprio dominio, preservando il primo errore e le altre chiamate.
Non viene ripresentata una POST dall'esito ambiguo: consegna locale una volta
sola non significa esecuzione remota exactly-once.

Record, snapshot e lease appartengono a una generazione del pool. Una modifica
di identità o tokenizer invalida i record pendenti incompatibili. Un cambio
dei limiti rivalida quelli non inviati. La sostituzione della topologia avviene
al confine drenato esistente; non viene eseguito hot-swap di alias nei batch
attivi. I GET restano osservazioni, senza garanzia atomica con la successiva POST.

La chiusura ferma ammissione e intake, segnala cancellazione, sveglia gli
attendenti, drena preparazione/HTTP e chiude infine worker e client posseduti.
Nessuna attesa di chiusura avviene sotto il lock delle code o del discovery.
`close` e `aclose` conservano proprietà e differenze di cleanup delle API attuali.

## Piano di implementazione e criteri di verifica

Le fasi seguenti sono unità proposte per l'acquisizione in Flower. Il codice
nuovo sostituisce il percorso attuale, senza scheduler paralleli o modalità legacy.

| Fase | Risultato e criterio di completamento |
| --- | --- |
| Modello di stato e budget | Record, generazioni, crediti, lease, memoria e invarianti verificati con eventi controllati |
| Motore persistente | Scheduler unico, code per replica e selezione al dispatch; rimozione di `_Quota`, `_ready_indices` e del coordinamento bloccante per finestra |
| Facciate e lifecycle | API sync/async e preparazione sul motore comune, metadati, cancellazione, topologia e close drenato compatibili |
| Qualifica e consegna | Test deterministici, lint/build/wheel pulito e confronto live limitato autorizzato; documentazione/versione/hash separati dall'accettazione consumer |

I test deterministici usano eventi/barriere e clock controllato, senza affidare
le asserzioni a ritardi arbitrari. Devono dimostrare:

- Arrivi durante una POST inclusi nel batch successivo; nessun batch congelato
  mentre aspetta capacità; nessuna attesa artificiale con policy zero.
- Sovrapposizione reale di preparazione e due POST su connessioni distinte;
  una replica lenta non blocca il lavoro eleggibile dell'altra.
- Quote su due/quattro repliche, resti dispari, meno testi che repliche,
  coda zero, limiti globali inferiori e più chiamanti concorrenti.
- Ordine e cardinalità con completamenti invertiti, fairness e nessuna
  duplicazione di invio o consegna.
- Backpressure di input/token/byte e consumer fermo; numero di thread limitato
  indipendentemente dal numero di finestre.
- Conteggio esatto, Unicode, prefisso/speciali, oversize misurabile, ID riusabili
  soltanto con identità/policy compatibili e nessuna seconda tokenizzazione nativa.
- Cancellazione e timeout in ogni transizione, fallimenti parziali, POST ambigua,
  cambio identità/tokenizer/limiti/topologia e chiusura durante tutte le fasi.

Il confronto live mantiene corpus, ordine, modello/pesi, prefissi, build,
profili, soglie numeriche e carico GPU comparabili. Copre ingestione continua,
finestre 256 + 255, input sporadici e più chiamanti. Misura richieste per endpoint,
testi/token/byte per POST, connessioni, tempo di preparazione, metadati, code,
HTTP e totale, primo risultato, latenze, memoria e thread. Registra anche il
tempo con capacità libera e backlog eleggibile; l'assenza di dati pronti è una
causa diversa dall'inefficienza dello scheduler.

Prima della qualifica si fissano ripetizioni, ordine alternato dei confronti e
criterio di regressione; mediana e variabilità accompagnano i risultati. Meno
HTTP o più concorrenza da sole non certificano un miglioramento. I tempi HTTP
possono sovrapporsi e non vengono presentati come puro tempo GPU. Non si
modificano modelli, runtime o consumer per rendere favorevole il confronto.

La promessa progettuale è eliminare attese e congelamenti introdotti dal
client, mantenendo i contratti. L'entità del guadagno dipenderà anche dal costo
della tokenizzazione remota e dalla capacità effettiva dei server caricati.
