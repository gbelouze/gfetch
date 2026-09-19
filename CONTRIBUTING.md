# Contributing guidelines

## Contributing code

1. (optional) Set up a Python development environment (advice: use
   [uv](https://docs.astral.sh/uv/))
2. Clone the repository and install `gfetch`
   ```bash
   git clone https://github.com/gbelouze/gfetch.git
   cd gfetch
   uv sync
   ```
3. Start a new branch off the main branch: `git switch -c my-new-branch main`
4. Make your code changes
5. Install the `pre-commit` hooks so formatting/lint/type checks run automatically before
   each commit
   ```bash
   uv run pre-commit install
   ```
   You can then run the checks by hand, or all at once with
   ```bash
   uv run pre-commit run --all-files
   ```
6. Commit, push, and open a pull request!
   ```bash
   git add file1 file2 file3  # add the modified files
   git commit -m "Short message to explain your changes"  # commit your changes
   git push -u origin my-new-branch  # change the branch name to the one you created in step 3.
   ```
   Use the link in the output, which should look something like
   `https://github.com/gbelouze/gfetch/compare/my-new-branch`, and create a *pull request*.
   Someone else will review your code and merge it to the repository!
