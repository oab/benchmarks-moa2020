# Benchmarking padding and dimension Lifting

Used to produce timings for "Padding Stencil 
Computations in the Mathematics of Arrays Formalism"(Table I&II).

To adapt parameters for your hardware adjust the 
makefiles and run the benchmark with  "make run".

The parameter SIZE is will determine the 3d-array size 
(SIZE\*SIZE\*SIZE). For the dimension lifting benchmarking 
there are 2 more parameters; KSPLIT which determines how to 
split the last axis and TSIZE which determines how to split all 
axes or tile. For the padding benchmark the PAD parameter 
determines the amount of padding or replication to do.
