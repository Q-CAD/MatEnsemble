# === TODO === 
- [x] Create strategy base class 
- [x] Implement the stategies 
- [x] Create strategy base class for processing futures
- [x] Implement strategies for processing futures
- [x] Refactor matflux.py and matfluxGen.py to be more modular and in one manager.py 

- [x] Test matflux/matfluxGen refactor make sure it works before doing anything else
- [x] Fix problems and test again 
    * Problems: ~/problems.txt
    - [x] Use *ONE* executor in the manager super loop instead of spawning new ones each time
    - [x] Make sure future objects have proper fields appended at creation (task_ or task + job_spec)
    - [x] Move writing of restart files into the FutureProcessingStrategy implementations
    - [x] Make sure you remove the finished future rather than popleft in FutureProcessingStrategy implementations

## NOTE: Refactored code runs way slower 

- [x] Fix problems causing slowdown and test again
    **--- More Problems ---**
    - [x] Make tests consistent so that we have an apples to apples comparison
    - [x] Remove extra logging and RPC calls to limit traffic 
    - [x] Update resources calls to update in place in submit_until_ooresources()

- [x] Test matensemble again until it is working as before

  **--- Got it working as before ---**
- [x] Update logging to be more industry standard 
- [x] Refactor Fluxlet to remove global side effects
- [x] Add type annotations back to strategies
- [x] Document all of the code vigorously 
    - [x] Document manager.py
    - [x] Document fluxlet.py
    - [x] Document strategies/*
- [x] Remove all TODOs and HACKs

- [x] Update the documentation and make sure it has all of the strategies
- [x] Make a script to build the documentation 
- [x] Remove all artifacts from the repository 

**--- Add Testing ---**
- [x] Make sure the simple hello world tests work
- [x] Figure out what is going on with the GPU tasks
- [x] Make some tests that have failures to make sure the failed tasks get logged appropriately 
- [ ] NOTE: Come back later -- Unit tests | Integration Tests --

### Find Solution For Distribution 

- [ ] Turn Matensemble into a uv project
- [ ] Build the initial Apptainer container 
- [ ] Test the apptainer container 

- [ ] Setup GitHub Actions to automatically build and register the container on code pushes to main branch 
- [ ] Test the changes 


- [ ] Allow tasks to have other tasks as dependencies

- [ ] Give the README a complete overhaul
