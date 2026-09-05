 
- Random idea 1 :  
    - it is possible that we go back to a session from yesterday
    - even more so with those daily reminder chats in gpt and claude
    - these have that temporal locality
    - it is possible that the already calculated kv cache contents have been evicted
    - and a few days later, going back to that chat, that entire session with all turns kv cache gets recomputed again
    - what if, i store the kv cache on disk (at least if there is a recurring reminder in that chat), and say my router is able to have "disc cache" view also, and i can tell the machine to bring up that content to memory?
    - or what if as a service provider, all the "recurring" chats go / get transferred to a "special" server, where eviction policy is aware of the recurrent nature of these sessions? but then again. You may have an investor who asks claude to update her on her portfolio every morning? Or a founder who has it take care of all the meeting overlaps every morning?

- what if we extend dual map idea, to say triple map? what if this is adaptive based on how many workers you have? 

- i think 

```
                 PREFIX REUSE
                     │
          ┌──────────┴──────────┐
          │                     │
   within-session          cross-session
          │                     │
 sequential extension      concurrent possible
          │                     │
 causally ordered          system/tool/template
          │                shared prefixes
 huge deep reuse                 │
                            shallower reuse
```