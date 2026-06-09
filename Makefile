.PHONY: geometry build dry-run-cases clean

geometry:
	python3 scripts/generate_geometry.py

build:
	scripts/build_solver.sh

dry-run-cases:
	python3 scripts/run_all_cases.py --dry-run

clean:
	$(MAKE) -C solver/sim_tcpc clean
