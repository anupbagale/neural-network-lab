# Contributing to Neural Network Lab

Thank you for helping to improve this lab! Contributions from students, instructors and
self-learners are all welcome — from fixing a typo to adding a new lesson.

## 🆘 Getting help

If something in a lesson is unclear, or a notebook does not run for you,
[open an issue](https://github.com/anupbagale/Neural_Network_lab/issues/new/choose) and choose
**Question / help**. Please say which lesson and cell you are on, and where you are running it
(Colab, Jupyter, VS Code…).

## 🐞 Reporting a problem

Use the **Bug report** template for code that crashes or gives wrong results, and
**Lesson feedback** for mistakes or confusing passages in the teaching material. A good report
includes:

- the lesson (and cell) or the `nnlab` function involved;
- what you expected and what happened instead;
- your Python, NumPy and Matplotlib versions (`python -c "import numpy, matplotlib, sys; print(sys.version, numpy.__version__, matplotlib.__version__)"`).

## 🛠️ Proposing a change

1. Fork the repository and create a branch: `git checkout -b my-change`.
2. Install the development tools: `python -m pip install -e ".[dev]"`.
3. Make your change. Please keep the existing style:
   - `nnlab` code should be **readable first** — each line should map onto an equation in a lesson;
   - lessons follow the *objectives → idea → by hand → code → see it → try it → check* structure;
   - figures use `nnlab.style.apply_style()` and the `PALETTE` colours, in order.
4. Run the checks:
   ```bash
   python -m pytest
   python -m pytest --nbmake notebooks/
   ```
5. If you changed a notebook, re-run it top to bottom so the saved outputs match the code.
6. Open a pull request describing **what** you changed and **why**.

## 💡 Ideas for contributions

- Additional exercises, or translations of existing lessons.
- New lessons that fit the sequence (e.g. Hopfield networks, Kohonen maps, radial basis functions).
- Reports of how you used the lab in a course — these are very valuable to the project.

By contributing you agree that code contributions are released under the [MIT License](LICENSE)
and teaching content under [CC BY 4.0](LICENSE-CONTENT), and that you will follow the
[Code of Conduct](CODE_OF_CONDUCT.md).
