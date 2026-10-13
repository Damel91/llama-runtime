# IMPL001 — Distribuzione pubblica e sviluppo privato

## Obiettivo

Pubblicare `llama-runtime` come repository autonoma, con storia Git nuova e
senza dati personali o configurazioni operative. Conservare separatamente la
versione precedente e le sue evidenze per lo sviluppo privato delle connessioni
remote/VPN. Il copyright pubblico resta `2026 Davide Mele`, su scelta esplicita
del titolare; la licenza del progetto pubblico è Apache-2.0, come Flower MCP.
Le attribuzioni e la licenza MIT del client derivato rimangono incluse.

## Confini

- La repository pubblica non contiene branch, oggetti Git, report originali,
  cache o artefatti provenienti dalla storia privata.
- Il nome della repository è `llama-runtime`; la distribuzione Python rimane
  `runtime-llama` e l'import rimane `runtime_llama`, per compatibilità.
- La versione pubblica 0.4.1 distingue questa distribuzione dalla precedente
  0.4.0. Non cambia il comportamento di inferenza o embedding.
- Endpoint ed esempi pubblici usano localhost o domini riservati agli esempi.
  Credenziali e configurazioni reali rimangono esterne alla repository.
- Build, modelli, server, consumer e rete non sono amministrati da questo lavoro.

## Realizzazione

1. Conservare la baseline privata nel branch `dev/vpn`, senza remote pubblico.
2. Esportare soltanto sorgenti, test, probe e documentazione tecnica selezionati.
   Escludere la storia Git e i resoconti contenenti configurazioni reali.
3. Conservare i contratti Python, discovery, scheduler e tokenizzazione. Cambiare
   solo versione, documentazione pubblica e metadati di distribuzione necessari.
4. Usare Apache-2.0 per il progetto con NOTICE e attribuzioni derivate intatte.
5. Rendere la costruzione sdist indipendente dall'identità dell'host: UID/GID
   zero, nomi proprietario/gruppo vuoti e timestamp controllati; nessun percorso
   locale nell'header gzip o nei metadati dell'archivio.
6. Verificare codice, pacchetti, storia pubblica e collegamenti documentali.
   Pubblicare soltanto i riferimenti della nuova repository dopo tali controlli.

## Verifiche

- Suite deterministica completa, Ruff e build senza isolamento.
- Installazione della wheel fuori checkout: import, versione, API e dipendenze.
- Confronto del payload runtime con la baseline: identico salvo la costante
  di versione; nessuna nuova qualifica live attribuita alla nuova release.
- Ispezione di wheel/sdist: contenuti attesi, Apache-2.0, NOTICE e MIT upstream;
  assenza di dati della macchina e metadati TAR/gzip normalizzati.
- Audit dei nuovi commit e tag: identità pubblica noreply, nessuna ascendenza
  privata, nessun branch di sviluppo pubblicato accidentalmente.

La documentazione tecnica non sostituisce il lifecycle esterno. Test software,
qualifica di una specifica installazione e acceptance restano distinti.
