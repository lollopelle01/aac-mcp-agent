# Open questions / doubts on the project

## 1. Original dataset (CommonGen)

### 1.1 Ho usato solo CommonGen in quanto il dataset con i pittrogrammi e i metadata deve essere uno specchio dei risultati delle API ARASAAC e quindi non del primo dataset fornito con HF, giusto?

### 1.2 In CommonGen ci sono 54k righe in cui sono espressi circa 6k id, però, in rapporto ai dataset che ho scaricato da ARASAAC, ci sono 1000 id che non esistono tra i 13k pittogrammi di ARASAAC. Inoltre questi 1000 id sporcano circa metà delle righe quindi il drop non è così scontato come in altri casi (ad esempio id == None).

### 1.3 La frase presente nel datset (attributo "sentence") è quella intesa dal bambino giusto? Poi è stata divisa in concetti e successivamente nel miglior id per quel concetto. Quindi di fatto devo guardare solo gli id giusto?

La frase e i concetti serviranno solo per motivare gli id, quindi nel mio caso servono solo come supporto all'annptazione del dataset per aggiungere i mock per il tool per la valutazione?

---

## 2. Annotated dataset

### 2.1 Come definiamo vago e chiaro?

Visto che mi sembrava chiaro dal testo che con "chiaro" non devi chiamare i tool e con "vago" li devi chiamare allora ho pensato che:

- chiaro = senso_frase_verboso + ora + senso_impegni
- vago = senso_frase

### 2.2 Annotazione tramite modello (Mistral) è corretta?

Maggiore comprensione semantica, vedi note di valutazione sulla generazione. Non sapendo bene quali sono i criteri di un input del caregiver in questo modo c'è una variabilità più umana. Richiede circa 10 ore sul cluster per tutto il dataset, si potrebbero provare altri parametri (magari temperature più alta) o altri modelli (tipo qwen).

### 2.3 Distrattori generici per assicurarsi che il modello usi l'ora per fetchare l'impegno giusto?

### 2.4 È corretto scegliere momenti della giornata e poi sorteggiare un orario in quel range?

---

## 3. Valutazione

### 3.1 È corretto assumere il teacher forcing?

Io assumerei che se il pittogramma corretto è presente nella finestra allora verrà selezionato e che se non c'è allora verrà cercato manualmente e poi selezionato?

### 3.2 Come accennato in annotazione, vengono generati impegni e orari/momenti della giornata per ogni impegno e quelli saranno dati ai tool che li restituiranno all'agente, simulando il funzionamento in fase di test, rendendolo deterministico.

### 3.3 Se anche droppassimo le righe con id non esistenti avremmo 25k righe, una lunga valutazione in locale. Si può fare una valutazione in generale, magari su vari modelli, con gpu (solo HF e no ollama) e poi fare solo qualche test di prova in locale con ollama?

### 3.4 Dobbiamo effettivamente valutare quando il modello chiama i tool e se era necessario?

### 3.5 Quale latenza è accettabile per l'utente? Quale dimensione della finestra è utile?

### 3.6 Nel dataset si usa un best_id per il concetto, si possono considerare anche i non-best_id ma sempre per quel concetto come validi ma un po'peggiori, tipo un top-k?

### 3.7 Come metriche vanno bene:

Sì, le metriche implementate sono appropriate. Ecco una spiegazione completa di ciascuna e di come viene usata nell'eval (`run_eval_hf.py`):

#### Metriche primarie di retrieval

**`gold_in_candidates`** (bool)
Verifica se il pittogramma gold è presente nel *pool di candidati* prodotto da `_search_candidates()` + `_expand_pool_by_synset()`, **prima** del ranking e del taglio alla finestra finale. Misura la qualità del retrieval a monte: se il gold non è mai nel pool, l'agente non ha speranza di mostrarlo. Corrisponde al "recall del retrieval".

**`hit` / `gold_in_window`** (bool) — *metrica primaria*
Verifica se il gold ID è nella finestra finale di `max_results` pittogrammi restituita all'utente. È la metrica più importante: rappresenta se il sistema ha effettivamente proposto il pittogramma giusto. Si misura separatamente per split `clear` e `vague`.

**`n_candidates`** (int)
Dimensione del pool prima del ranking. Utile per capire se il retrieval è troppo stretto (pool piccolo → hit basso per forza) o troppo largo (pool enorme → il ranking diventa critico).

**`window_len`** (int)
Dimensione effettiva della finestra restituita (≤ `max_results`). Può essere inferiore se ci sono pochi candidati fresh.

**`fresh_count`** (int)
Numero di pittogrammi "fresh" (non già presentati in turni recenti) nella finestra finale. Un valore basso indica che il sistema sta facendo molto padding con pittogrammi stale — da monitorare in sessioni lunghe.

#### Metriche di overlap semantico

**`overlap_level`** (str | None)
Anche quando il gold non è nella finestra (`hit=False`), misura quanto "vicini" semanticamente sono i pittogrammi proposti. I livelli, in ordine decrescente di qualità:

| Livello      | Significato                                                                  |
| ------------ | ---------------------------------------------------------------------------- |
| `synset`   | Almeno un pittogramma nella finestra condivide un synset WordNet con il gold |
| `category` | Almeno uno condivide una categoria ARASAAC con il gold                       |
| `keyword`  | Almeno uno condivide una keyword con il gold                                 |
| `tag`      | Almeno uno condivide un tag con il gold                                      |
| `None`     | Nessuna sovrapposizione trovata                                              |

È calcolato confrontando i metadata di ogni pittogramma nella finestra con quelli del gold. Permette di distinguere un fallimento "vicino" (es. synset overlap) da uno "lontano" (nessun overlap).

---

## 4. Architettura / Pipeline

### 4.1 Il modello deve chiamare tutto o si limita a intervenire sul testo? Perchè l'accesso alle informazioni può seguire meccanismi deterministici tipo sql per poi farlo ragionare sul risultato, che sarebbe sicuramente più veloce ma sicuramente ci sarebbero delle perdite semantiche: il modello non può conoscere tutte le keyword, o memorizzare bene tutti i metadati come se fossero degli embedding.

### 4.2 Concettualmente l'idea è una simmetria online/offline. Ogni operazione sui dati viene fatta tramite la stessa operazione che in base a una flag o alla necessità (no internet) guarda localmente oppure online.

### 4.4 La consegna prevedeva un secondo step LLM di filtraggio dopo il retrieval. L'implementazione attuale lo ha rimosso in favore di un ranking deterministico (concept_order + quality_score). È accettabile eliminare del tutto il secondo LLM call, o va motivato esplicitamente?

### 4.5 Il planner LLM decide autonomamente se chiamare `get_time`/`get_schedule` (call_tools). Alternativa: la decisione potrebbe essere hardcoded (vago = sempre chiama i tool). Quale approccio è più corretto per un sistema MCP agentico?

### 4.6 Il sistema usa una sola chiamata LLM per turno (solo il planner). Questo è sufficiente per la tesi o è necessario giustificare perché non si usa un approccio ReAct/chain-of-thought multi-step?

### 4.7 La dimensione della finestra è fissa. Andrebbe resa adattiva in base al numero di candidati trovati, o la fissità è una scelta progettuale difendibile?

### 4.8 Il sistema non ha un segnale esplicito di "frase completata": il caregiver svuota l'input manualmente dopo aver selezionato tutti i pittogrammi di una frase. È il comportamento atteso per un sistema AAC reale?
