# Contributing

Thanks for your interest in improving these notebooks. Fixes for typos and mistakes are just as
welcome as new exercises or lessons.

## Asking for help

If something in a lesson is unclear or a notebook doesn't run, please
[open an issue](https://github.com/anupbagale/Neural_Network_lab/issues/new/choose) and pick
"Question". Say which lesson and cell you are on and where you are running it (Colab, Jupyter,
VS Code and so on).

## Reporting a problem

Use "Bug report" for code that crashes or gives a wrong result, and "Lesson feedback" for mistakes
or confusing parts of the text. It helps a lot if you include:

- the lesson and cell, or the `nnlab` function involved
- what you expected and what happened instead
- your Python, NumPy and Matplotlib versions, for example from
  `python -c "import sys, numpy, matplotlib; print(sys.version, numpy.__version__, matplotlib.__version__)"`

## Making a change

1. Fork the repository and create a branch.
2. Install the package with the development tools: `pip install -e ".[dev]"`
3. Make your change. A few things to keep in mind:
   - Code in `nnlab` should stay easy to read. Each line should match an equation in a lesson.
   - Lessons follow the same order: objectives, the idea, a worked example, code, plots,
     exercises.
   - Plots should call `nnlab.style.apply_style()` and use the colours in `PALETTE`, in order.
4. Run the tests:
   ```bash
   pytest
   pytest --nbmake notebooks/
   ```
5. If you edited a notebook, run it from top to bottom before committing so the saved output
   matches the code.
6. Open a pull request that explains what you changed and why.

Some things that would be useful: more exercises, translations, new lessons that fit the sequence
(for example Hopfield networks or Kohonen maps), and notes on how you used the notebooks in a
course.

Code contributions are released under the [MIT License](LICENSE) and text under
[CC BY 4.0](LICENSE-CONTENT). Please follow the [code of conduct](CODE_OF_CONDUCT.md).
