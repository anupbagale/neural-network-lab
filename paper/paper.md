---
title: 'Neural Network Lab: Jupyter lessons that implement classic neural network algorithms from scratch'
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
    # orcid: 0000-0000-0000-0000   # TODO: add your ORCID iD
    affiliation: 1
affiliations:
  - name: TODO Department, University, City, Country
    index: 1
date: 15 October 2026
bibliography: paper.bib
---

# Summary

Neural Network Lab is a set of eight Jupyter [@kluyver2016] notebooks that implement the classic
neural network algorithms in NumPy [@harris2020]. The lessons start with the McCulloch-Pitts
threshold neuron [@mcculloch1943], go through the Hebbian, perceptron and delta learning rules to
backpropagation [@rumelhart1986], and end with bidirectional associative memory [@kosko1988] and
fuzzy sets [@zadeh1965]. In each lesson the learner works a small example by hand, runs the same
example in code and looks at the result in a Matplotlib [@hunter2007] figure. The algorithms are
in a small Python package, `nnlab`, and a test suite checks them against the hand-worked results.
The notebooks run in Google Colab without any installation.

# Statement of need

Many undergraduate courses on neural networks and soft computing still teach the classical learning
rules, usually from textbooks such as @zurada1992 and @haykin2009. Students practise these rules as
hand calculations but rarely see them run, and turning a worked example into working code is left
to them. There are good open resources for modern deep learning [@nielsen2015; @goodfellow2016;
@zhang2023], but they generally start at the multi-layer perceptron and move on to high-level
frameworks. Earlier material, such as why one neuron cannot compute XOR [@minsky1969] or how the
delta rule [@widrow1960] leads to backpropagation, gets little space.

These notebooks are meant for instructors who need lab material for the classical part of such a
course, and for students who want to see what happens inside a training loop. The worked examples
use the notation and data of @zurada1992, so the notebooks can be used next to that book or on
their own. The code is released under the MIT license and the text and figures under CC BY 4.0, so
lessons can be reused, changed and translated.

# Learning objectives and content

After working through the notebooks, a learner should be able to:

1. explain how a weighted sum and a threshold implement logic gates, and why XOR needs a hidden layer;
2. compare step, sigmoid and ReLU activations, including the effect of the steepness parameter
   $\lambda$ and the link between saturation and vanishing gradients;
3. apply the Hebbian, perceptron and delta rules by hand and in code;
4. derive backpropagation for a network with one hidden layer, check it numerically, and interpret
   the hidden layer as a new representation of the input;
5. store and recall pattern pairs in a bidirectional associative memory and reason about its capacity;
6. compute fuzzy set operations and say which laws of ordinary set algebra fail for fuzzy sets.

The lessons are (1) McCulloch-Pitts neurons and logic gates, (2) activation functions,
(3) Hebbian learning [@hebb1949], (4) the perceptron learning rule [@rosenblatt1958], (5) the delta
rule, (6) backpropagation on XOR, (7) bidirectional associative memory and (8) fuzzy sets.
Lessons 1 to 6 build on each other. Lesson 7 needs only lesson 3, and lesson 8 can be taken any
time after lesson 2. Each lesson takes 25 to 40 minutes.

# Instructional design

All notebooks have the same layout: objectives and prerequisites, the idea and its mathematics, a
small example to work by hand, the code, a figure, exercises, and questions with answers that are
hidden until expanded. Learners are asked to work out the first steps of each example on paper
before running the code. The code then prints the same steps as a table, so learners can check
their own working line by line.

The `nnlab` package is written for reading rather than speed. For example, each line of
`TwoLayerNet.train_step` corresponds to one equation in lesson 6, so the implementation can be read
as a second explanation of the algorithm, and exercises can ask students to modify it. The lessons
also show cases where the methods fail: the perceptron never settles on XOR, a large learning rate
stops backpropagation from converging, and recall in an associative memory breaks down as more
pairs are stored. The test suite includes a finite-difference check of the backpropagation
gradients, and continuous integration runs every notebook on each change. All figures use one
colour palette chosen to be readable with common forms of colour blindness, and classes are also
told apart by marker shape.

# Experience of use

<!--
TODO (required by JOSE; please write this yourself and keep it factual).
Describe how the material has been used, for example:
  - the course, semester and institution where the original lab notebook was used, and roughly how many students;
  - what worked and what students found hard;
  - any feedback and the changes you made because of it.
If it has not been used in a class yet, say so and describe how you plan to use or evaluate it.
-->

# Story of the project

<!--
TODO (write this in your own words). For example, where the original notebook came from and why
you turned it into separate lessons with a tested package.
-->

# AI usage disclosure

Generative AI was used in preparing this submission. Anthropic's Claude (Claude Opus 5.5) was used
to restructure the original single notebook into the `nnlab` package and eight lesson notebooks, to
write the test suite, to draft the lesson text, exercises and answers, to draft the README and
contributor documentation, and to produce a first draft of this paper. The original algorithms,
worked examples and choice of topics are the author's.
<!-- TODO: review every AI-generated part before submitting, then keep (and make true) the
sentence below, adding anything else you changed. -->
The author reviewed, edited and validated all AI-assisted content, checked that every
implementation reproduces the original hand-worked results, and made the final decisions on scope,
structure and teaching approach.

# Acknowledgements

The worked examples in lessons 3 to 5 follow examples in @zurada1992.

# References
