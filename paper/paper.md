---
title: 'Neural Network Lab: Hands-on Jupyter lessons that build classic neural networks from scratch'
tags:
  - Python
  - Jupyter
  - neural networks
  - perceptron
  - backpropagation
  - associative memory
  - fuzzy sets
  - soft computing
authors:
  - name: Anup Bagale
    # orcid: 0000-0000-0000-0000   # TODO: add your ORCID iD (strongly recommended by JOSE)
    affiliation: 1
affiliations:
  - name: TODO — Department, University, City, Country
    index: 1
date: 15 October 2026
bibliography: paper.bib
---

# Summary

*Neural Network Lab* is an open learning module of eight executable Jupyter [@kluyver2016]
lessons that take learners from a single McCulloch–Pitts threshold neuron [@mcculloch1943] to a
multi-layer network trained by backpropagation [@rumelhart1986], and then to two classic
neighbouring topics: bidirectional associative memory [@kosko1988] and fuzzy sets [@zadeh1965].
Each lesson derives one idea, asks the learner to work a small example by hand, reproduces that
example in a few lines of NumPy [@harris2020], and visualises the result with Matplotlib
[@hunter2007]. The algorithms live in `nnlab`, a small Python package written to be read alongside
the equations, and an automated test suite checks every implementation against the hand-worked
results. Lessons run in the browser through Google Colab with no installation.

# Statement of need

Many undergraduate courses on neural networks or soft computing still teach the classical learning
rules — Hebbian, perceptron and delta rules, backpropagation, associative memories and fuzzy sets —
following textbooks such as @zurada1992 and @haykin2009. Students meet these rules as hand
calculations, but rarely see them run, and the step from a worked example to working code is left
to them. Excellent open resources exist for modern deep learning [@nielsen2015; @goodfellow2016;
@zhang2023], but they typically begin at the multi-layer perceptron and quickly move to high-level
frameworks, so the earlier ideas that motivate them — why a single neuron cannot compute XOR
[@minsky1969], how the delta rule [@widrow1960] becomes backpropagation — are covered briefly or not
at all.

This module fills that gap. It is aimed at (i) instructors who need ready-to-use laboratory
material for the classical part of a neural networks or soft computing syllabus, and
(ii) self-learners who want to understand what happens inside `model.fit()`. Because the worked
examples follow the notation and data of @zurada1992, the lessons can be adopted alongside that
textbook or used on their own. All content is openly licensed (code MIT, text and figures CC BY 4.0),
so instructors can reuse, remix and translate individual lessons.

# Learning objectives and content

After completing the module, learners should be able to

1. explain how a weighted sum and a threshold implement logic gates, and why non-linearly-separable
   functions such as XOR require a hidden layer;
2. compare step, sigmoid and ReLU activations, including the role of the steepness parameter $\lambda$
   and the link between saturation and vanishing gradients;
3. apply the Hebbian, perceptron and delta learning rules by hand and in code, and contrast
   unsupervised, error-correcting and gradient-based learning;
4. derive backpropagation for a network with one hidden layer, verify it numerically, and interpret
   the hidden layer as a learned change of representation;
5. store and recall pattern pairs in a bidirectional associative memory and reason about its capacity;
6. compute fuzzy-set operations and identify which laws of classical set algebra fail for fuzzy sets.

The eight lessons are: (1) McCulloch–Pitts neurons and logic gates; (2) activation functions;
(3) Hebbian learning [@hebb1949]; (4) the perceptron learning rule [@rosenblatt1958]; (5) the delta
rule and gradient descent; (6) backpropagation on XOR; (7) bidirectional associative memory; and
(8) fuzzy sets. Lessons 1–6 form a core sequence; Lesson 7 builds on Lesson 3, and Lesson 8 can be
taken at any point after Lesson 2. Each lesson takes about 25–40 minutes.

# Instructional design

Every notebook follows the same, explicitly signposted structure: learning objectives and
prerequisites; the idea and its mathematics; a small example to **work by hand first**; the code;
a figure that makes the behaviour visible; “try it yourself” exercises; and self-check questions with
collapsible answers. Asking learners to predict a result on paper before running the code turns each
code cell into immediate feedback on their own reasoning, and the library prints textbook-style
update tables so that the two can be compared line by line.

The supporting package `nnlab` deliberately favours clarity over speed: each line of, for example,
`TwoLayerNet.train_step` corresponds to one equation in Lesson 6. Learners can therefore read the
implementation as a second explanation, and instructors can set exercises that modify it. Lessons
also treat failure as content: the perceptron cycling forever on XOR, a learning rate large enough to
stop backpropagation converging, and BAM recall breaking down as more pairs are stored are all shown
and explained rather than hidden. The test suite, including a finite-difference gradient check of
backpropagation, and continuous integration that executes every notebook keep the material correct
as it evolves. All figures share a single colour-blind-checked palette and use marker shape as well
as colour to distinguish classes.

# Experience of use

<!--
TODO (required by JOSE — please write this yourself and keep it factual).
Describe how the material has actually been used, e.g.:
  - in which course / semester / institution the original lab notebook was used, and by how many students;
  - what worked well and what students found difficult;
  - any feedback received, and changes you made because of it.
If it has not yet been used in a class, say so honestly and describe the self-study use and how you
plan to use or evaluate it. Do not overstate.
-->

# Story of the project

<!--
TODO (write in your own words). For example: the lab began as the author's own collection of
exercises written while taking/teaching a neural networks course at <institution>, originally as a
single Colab notebook; it was then restructured into eight lessons with a tested helper package
so that others could adopt it.
-->

# AI usage disclosure

Generative AI was used in preparing this submission. Anthropic's Claude (model Claude Opus 5.5) was
used to restructure the author's original single notebook into the `nnlab` package and eight lesson
notebooks; to generate the test suite scaffolding; to draft lesson narrative, exercises and
self-check answers; to draft the README and community documentation; and to produce a first draft of
this paper. The original algorithms, worked examples and choice of topics are the author's.
<!-- TODO: before submitting, review every AI-generated part, then keep (and make true) the sentence
below, adding anything else you changed. -->
The author reviewed, edited and validated all AI-assisted content, verified that every
implementation reproduces the original hand-worked results, and made the final decisions on the
module's scope, structure and pedagogy.

# Acknowledgements

The worked examples in Lessons 3–5 follow the classic examples in @zurada1992.

# References
