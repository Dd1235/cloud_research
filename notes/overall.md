# As of 4th September

```
LLM serving simulator
│
├── Workload / execution model
│   ├── arrivals + queues
│   ├── prefill
│   ├── decode
│   ├── continuous batching
│   └── chunked prefill
│
├── Metrics
│   ├── TTFT
│   ├── TPOT
│   ├── TBT / TBT p99
│   ├── cache hit rate
│   ├── queue/load balance
│   └── routing/cache-view errors
│
├── Routing policies
│   │
│   ├── cache-blind
│   │   ├── random
│   │   ├── round robin
│   │   └── P2C
│   │
│   ├── hash/affinity constrained
│   │   ├── session hash
│   │   └── DualMap
│   │
│   └── explicit cache-aware
│       ├── longest-prefix ranker
│       ├── hybrid scorer
│       ├── LMETRIC product
│       └── Dynamo-style additive cost
│
├── Router's cache knowledge
│   ├── perfect view
│   ├── periodic snapshot
│   ├── shadow / record-insert
│   └── delayed events      ← not built yet
│
├── Staleness failure analysis
│   ├── false positives
│   ├── false negatives
│   ├── routing regret
│   └── execution false positives
│
├── Theory
│   ├── cache turnover / characteristic time
│   ├── age-of-information normalization
│   ├── block survival model
│   ├── radix-tree dependence
│   └── per-worker re-reference rates
│
└── Repairs
    ├── TTL / survival discount       ← stale promises
    ├── dispatch overlay              ← blind spot
    ├── load term / threshold
    └── hash cap such as DualMap

```

-  we start with an ideal m/m/1 server, that gives latency $1 / (\mu - \lambda)$
- deterministic round robin does better than random, or p2c in a homogeneous setup becaues it regularizes each workres arrival stream.
- `queue_cv = how evenly time-varying outstanding queues were distributed` which is different from load_cv that's about total work
- tpot hides stalls so we also include tbt 99, this becomes especiall important for chunked prefill
- cache-blind policies are good for load balancing especially with heterogeneous workers
- single-home affinity, prefix/session hash -> server
- dual map has two home affinity
- then we make routers cache-knowledge a first class citizen, worker.cache is different from worker.view
    - here in shadow, worker knows the inserts but not the eviction
- staleness 
    - routing regret can be usefull even with perfect view situation, because its possible that we have traded cache locality against queuing. A may match 1000 blocks, B may match 800, and we pick B based on our algorithm
    - execution fp is : dispatch type cache said present, by execution time it got evicted.

- shadow can appear outperform even perfect view?! because of free in-flight KV sharing. we need a kv_available_at = prefill_done type metric to negate this. but do we want to though?but this seems to be an artefact of simulation solved specifically for simulation! for a typical conversion within-session traffic is serial and strongly causally ordered.

- in terms of views, a "5 sec old view" is not as informative as "0.5 cache lifetimes old"
- boom theoretical cousin, che's approximation
- behave as though each cached object remains resident for a characteristic time \(T_C\) after its latest access.
- and our measured cache turnover is approximately $ T_c $ 
- `view_age / T_c`
- essentially an age of information
- essentially instead of a boolean stale cache, we have a confidence decaying as the age increases.
- but in radix tree like stucture, leaves can disappear fast but interior blocks can live much longer.
- if I never saw a block i might ask F(idle age), but if a snapshot saw the block alive given its already survived until now, what is the probability it survives the snapshots age? (oh wow reminds me of garbage collection, like a generational bias)
- we also cant assume correlated elements are independent. if block 12 exists so does its ancestors
- also relevant rate is a function of worker and block, not fleet and block. because if a prefix gets spread over 8 machines, only this workers references will be refreshed.
- but real traces we got a different problem, we conviniently had block 0 - meaningful prefix identity but real traces don't. The conversation trace had essentially one distinct first block across ~6000 requests because system prompts get shared. 
    - The intended DualMap approach is even better: adaptively extend the hashed prefix when a key becomes too hot.
- dualmap performed well on both traces because its only among {A,B} that we determine based on cache. You can gain robustness not merely by making metadata fresher, but by reducing how much damage bad metadata is allowed to cause.
- orderinal vs cardinal use of cache state
    - Cardinal policies are much more sensitive to false cache promises because an exaggerated overlap directly distorts the magnitude of the routing score.

    - but pure rankers have their own pathology. shadow metadata + lpm were very poor on real traces. there can be herding, that tiny prefix advatage at the beginning can cause much larger prefix advantage
    - self-observation changes the system you're observing.
- stale promises - reduce confidence with age. false negative/blind spot - snapshot + dispatches since snapshot


- also apart from shared boilerplate most deep prefix sharing is probably per session
- roughly tc = cache_capacity/eviction rate
- also given the assumption of initial few nodes being cache for boiler plate like system prompt that is present is almost every llm query and the rest being conversations, that would rarely branch and are unique to a conversation, we can make sure keep these almost static things in cache for sure? even when theres overload, so that these are not thrashed? but actually this problem maybe self solving, because genuinely every turn everything above will be repeated?

- but one global \(T_C\) is probably too crude for a radix tree? also very eminiscent of generations gc?
    - young objects die quickly, and objects that survive repeatedly
    - for kv, unique tail has low reuse, but systm/tool prefix are constantly touched

- rescue rate: how frequently new accesses on that same worker refresh a cached block before it would otherwise disappear.



