# Default recipe: run all checks
default: check

# Run all quality checks
check:
    @echo "\n══════════════════════════════════════"
    @echo "  CHECK"
    @echo "══════════════════════════════════════\n"
    @uv run prek run --all-files

# Run tests
test *ARGS:
    @echo "\n══════════════════════════════════════"
    @echo "  TEST"
    @echo "══════════════════════════════════════\n"
    @uv run pytest tests/ {{ARGS}}

# Score predicted output against expected
score EXPECTED PREDICTED:
    @echo "\n══════════════════════════════════════"
    @echo "  SCORE"
    @echo "══════════════════════════════════════\n"
    @uv run src/cli.py score --expected {{EXPECTED}} --predicted {{PREDICTED}}

# Score an experiment folder (uses the expected.tsv written alongside extracted.tsv)
score-expt FOLDER:
    @echo "\n══════════════════════════════════════"
    @echo "  SCORE EXPERIMENT {{FOLDER}}"
    @echo "══════════════════════════════════════\n"
    @uv run src/cli.py score --expected expts/{{FOLDER}}/expected.tsv --predicted expts/{{FOLDER}}/extracted.tsv --input data/playgroup_dev_in.tsv

# Compare all experiments
compare:
    @echo "\n══════════════════════════════════════"
    @echo "  COMPARE"
    @echo "══════════════════════════════════════\n"
    @uv run src/cli.py compare

# Run CLI, this will trigger the pipeline
extract *ARGS:
    @echo "\n══════════════════════════════════════"
    @echo "  EXTRACT"
    @echo "══════════════════════════════════════\n"
    @uv run src/cli.py -v extract {{ARGS}}

# Extract, score, and compare in one go
run *ARGS:
    @echo "\n══════════════════════════════════════"
    @echo "  EXTRACT"
    @echo "══════════════════════════════════════\n"
    @uv run src/cli.py -v extract {{ARGS}}
    @echo "\n══════════════════════════════════════"
    @echo "  SCORE"
    @echo "══════════════════════════════════════\n"
    @folder=$(ls -t expts/ | head -1) && uv run src/cli.py score --expected expts/$folder/expected.tsv --predicted expts/$folder/extracted.tsv --input data/playgroup_dev_in.tsv
    @echo "\n══════════════════════════════════════"
    @echo "  COMPARE"
    @echo "══════════════════════════════════════\n"
    @uv run src/cli.py compare
