(This project is currently incomplete.)

Does `cudaMalloc()` slow you down? 
Database query processing requires temporary memory for intermediate results and data structures, and invocations of `cudaMalloc()` mid-query to provision the required memory can significantly degrade performance.
The GPU database programmer is thus impelled to invoke `cudaMalloc()` once at database startup and manage the memory manually.
We provide in this library a stack allocator, where `cudaMalloc()` is invoked once and memory is allocated and freed off the stack by simply moving a pointer to the memory pool.
Of course, with this design, fragmentation becomes a problem, as allocations with unused memory can become 'trapped' by active allocations higher up on the stack.
This library implements efficient `memmove()` and an extension to a large number of source memory regions, allowing the database programmer to compress fragmented allocations in order to free space for the stack allocator.
Thus, the two components of this library work together to provide a query execution environment unencumbered by `cudaMalloc()`,
allowing maximal GPU utilization without recourse to streams and maximal memory utilization.

Currently, you will only find `MemMove` implemented, and the stack allocator is absent. There are no examples.
I expect to have this finished by the end of the week, as I have already largely implemented these ideas elsewhere in more fragmented form.