# Progetto 1: MCP Agentico per la Selezione Contestuale di Pittogrammi

I sistemi CAA attuali basati su ARASAAC richiedono spesso che il caregiver navighi manualmente tra grandi directory di pittogrammi per avviare la conversazione. Questo progetto mira a sviluppare una pipeline agentica MCP che consenta di avviare una comunicazione partendo da una semplice descrizione testuale del contesto, come ad esempio:
* “Siamo in biblioteca”
* “È mattina e stiamo andando a scuola”

Il sistema deve arricchire automaticamente questa descrizione utilizzando informazioni contestuali (tempo, calendario, posizione simbolica) e interrogare ARASAAC per proporre un insieme iniziale di pittogrammi rilevanti.

### Componenti della Pipeline
La pipeline può includere i seguenti componenti:
* **ARASAAC API** (retrieval pittogrammi + metadata)
* **get_time** (ora e momento della giornata)
* **get_schedule** (calendario personale / agenda)

### Flusso della Pipeline
Idealmente, il flusso della pipeline dovrebbe essere:
1.  **Input**: breve descrizione testuale fornita dal caregiver.
2.  **Arricchimento del contesto**:
    * Se la descrizione è vaga $\rightarrow$ interrogare calendario e tempo.
    * Se è dettagliata $\rightarrow$ usarla direttamente.
3.  **Retrieval iniziale** di un pool di pittogrammi tramite ARASAAC API.
4.  **Filtro tramite LLM** (prompt-based - no fine-tuning), che seleziona i pittogrammi più adatti al contesto.