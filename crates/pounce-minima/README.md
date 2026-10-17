# pounce-minima

Building blocks of the [POUNCE](https://github.com/jkitchin/pounce)
multistart / find-minima global search: the deduplicating archive of found
minima, the start sampler (seedable ChaCha8 and Owen-scrambled Sobol'), the
penalty and tunnel TNLP wrappers used by the flooding, deflation and tunneling
strategies, and the search options.

The search itself is exposed today through the `pounce` CLI
(`pounce model.nl --minima <method>`) and, from Python, `pounce.find_minima`.
See the [find-minima documentation](https://jkitchin.github.io/pounce/find-minima-cli.html).

Licensed under EPL-2.0.
