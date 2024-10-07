Some stuff for BabyLM Challenge 2024
====================================
Here lies some ideas I had for [BabyLM Challenge 2024](https://babylm.github.io/) that
didn't quite work in the end. Experiments used [muP](https://github.com/microsoft/mup)
where possible to ensure the models were well-behaved and scalable. The default gpt2
model from huggingface, but adapted to use muP, was used as a benchmark.

Use LLM to search for linear-time attention functions
-----------------------------------------------------
Rather than hand-crafting a novel attention function this searches functions using LLM prompting.

 * Generate code via LLM to implement an attention function `attention(keys, values, queries)`
 which is used in a transformer model set up to use an arbitrary attention function.
 * Prompt with a history of previous implementations with their validation metrics/error messages.
 * Test that the function doesn't leak future information by training a next-token linear model
 on its output when applied to a random sequence and checking if it exceeds random chance performance.

Holographic reduced representations key/value attention
-------------------------------------------------------
I've been interested in making this work for a while but previous attempts have failed.
This round was no exception as the performance never beat the gpt2 benchmark and was
not much better than a simple embeddeding + linear output model.
 * I suspect the input and output transformations are doing most of the work and the HRR attention as I've set up isn't being useful. Maybe this can be improved still?
 * using 'ortho' norm in fft ops was necessary for better-than-random output

Complexity-based curriculum learning
------------------------------------
Sort dataset examples based on their complexity (in this case, compressed size). Prioritize learning on training examples
which were difficult.

 * Didn't significantly improve model performance