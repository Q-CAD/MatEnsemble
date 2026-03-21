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
- [x] NOTE: Come back later -- Unit tests | Integration Tests --

### Find Solution For Distribution 

- [x] Turn Matensemble into a uv project
- [x] Build the initial Apptainer container 
    - [x] Create the matensemble.def file
    **--- Def File Spec ---**
    * The file should be off of a frontier base image use rocky linux version
    * Install build dependecies of flux-core
    * Build flux-core from source 
    * Install build dependencies for flux-sched
    * Build flux-sched from source 
    * Export all variables 
    * Install matensemble
- [x] Test the apptainer container 

**--- Build Base Images ---**
- [x] Build Base Image for Baseline
- [x] Build Base Image for Frontier
- [x] Build Base Image for Perlmutter
- [x] Push images to GitHub Container Registry
- [x] Build MatEnsemble Images with each base image
- [x] Test Images on each respective system
- [ ] NOTE: Come back later -- Test Perlmutter Image

**--- Setup GitHub Actions ---**
- [x] Setup Matrix build action to build MatEnsemble images for baseline, frontier, and perlmutter and push them to ghcr
- [x] Setup action to build with uv and publish with uv 
- [x] Setup action to build docs and publish them

**--- Test CI/CD ---**
- [x] Make small change to MatEnsemble and Docs
- [x] Push to main see if dev builds succeed
- [x] Run release.sh script to see if releases happen properly

**--- Updated/Better UX ---**
- [x] Refactor to be built around Task/Job Objects
- [x] Allow users to decorate python functions to create TaskSpec's 
    - [x] Allow functions to depend on other functions 
    - [x] Topologically sort all of the Jobs based on dependencies
- [x] Write worker runtime that flux can target and call user defined functions
- [x] Write the Job objects specification to a file in their direcotry 

**--- Status Dashboard ---**
- [ ] Update the Pipeline.run() method to have a dashboard flag 
- [ ] Add logic to launch the dashboard when the user runs the workflow 

**--- Polish Everything ---**
- [ ] Update all the documentation 
- [x] Update the example workflows 
- [ ] Provide tutorials for how to run the example workflows 

**--- Science Example ---**
- [x] Test the real science example that Soumendu provided 
- [ ] Update version of LAMMPS

**--- Model Context Protocol ---**
- [ ] MCP implementation 
