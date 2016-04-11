# Scott Linderman's Thesis, April 2016

Neuroscience is entering an exciting new age.  Modern recording
technologies enable us to simultaneously measure the activity of
thousands of neurons in organisms performing complex behaviors.  Such
recordings offer an unprecedented opportunity to glean insight into
the mechanistic underpinnings of intelligence, but they also present
extraordinary statistical and computational challenges: how do we make
sense of these large scale recordings and turn data into
understanding? This thesis develops a suite of tools that instantiate
hypotheses about neural computation in the form of
probabilistic models and a corresponding set of Bayesian inference
algorithms that efficiently fit these models to neural spike trains.
From the posterior distribution of model parameters and variables,
we seek to advance our understanding of how the brain works. 

## Building

First, install the included fonts. Then run
`./scripts/build`. You'll need `xelatex` in order to compile with the
non-standard fonts.

To compile a single chapter, run
    xelatex --jobname=<chapternumber> singlechater.tex
    bibtex <chapternumber>
    xelatex --jobname=<chapternumber> singlechater.tex
    xelatex --jobname=<chapternumber> singlechater.tex



## Credits
Built with the [Dissertate](https://github.com/suchow/Dissertate) package. 

