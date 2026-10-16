The SMS Spam Collection used in chapter 15 contains 5574 text messages, of which 747 are labelled spam. Only 5171 of the messages are distinct, so 403 rows are exact copies of other rows.

Chapter 15 computes sentence embeddings with the encoder all-MiniLM-L6-v2. Its weights are released under the Apache-2.0 licence, and its outputs have 384 dimensions.

The daily bike-rental data of chapter 16 cover 731 days, from 1 January 2011 to 31 December 2012. They come from the Capital Bikeshare system and are licensed CC BY 4.0.

The quietest day in the bike-rental data is 29 October 2012, when only 22 bicycles were rented.

The temporal convolutional network of chapter 16 uses kernel size 3 and dilations 1, 2, 4, 8 and 16, which gives a receptive field of 63 days.

In the chapter 16 backtest, Holt-Winters reaches a mean absolute scaled error of 1.204, and the seasonal naive forecast reaches 1.272.

The cart-pole task of chapter 19 counts as solved when the mean return over 100 consecutive episodes reaches 475. Episodes are cut off after 500 steps.

The hand-written cart-pole controller pushes the cart towards the side to which the pole is falling, judged by the pole angle plus half its angular velocity.

In the grid world of chapter 19 the discount factor is 0.9, and the optimal value of the start state is 0.729.

With seed 1, the REINFORCE agent of chapter 19 first solves the cart-pole task at episode 549.

The deep Q-network of chapter 19 copies its online weights into the target network every 1000 steps and stores up to 50000 transitions in its replay buffer.

Over the ten held-out digit classes of chapter 20, the autoencoder reaches a mean AUC of 0.912 and PCA reaches 0.880.

The diffusion model of chapter 20 uses 100 noising steps, with the noise variance rising linearly from 0.0001 to 0.2.

GPT-2 assigns the original test sentence of chapter 21 a perplexity of 48.0; with its words shuffled the perplexity rises to 6051.7.

The instruction-tuned model used for retrieval-augmented generation in chapter 21 is Qwen2.5-0.5B-Instruct, released under the Apache-2.0 licence.

With the rounded published constants of the parametric loss fit, the compute-optimal model for the Gopher budget has 30.3 billion parameters.

The companion repository of this book is published at github.com/abzzall/neural_network_book and its code is released under the MIT licence.

The book is typeset in Times New Roman at 14 points on A4 paper and built with XeLaTeX.

Every random experiment in the companion code is repeated for at least three seeds when a single seed could be misleading.

Review questions in every chapter mix comparison, scenario, figure interpretation, design evaluation and risk questions.
