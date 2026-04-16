# === TODO === 
- [x] Create strategy base class 
- [x] Implement the stategies 
- [x] Create strategy base class for processing futures
- [x] Implement strategies for processing futures
- [x] Refactor matflux.py and matfluxGen.py to be more modular and in one manager.py 

- [x] Test matflux/matfluxGen refactor make sure it works before doing anything else
- [x] Fix problems and test again 
    ### --- Problems ---
    - [x] Use *ONE* executor in the manager super loop instead of spawning new ones each time
    - [x] Make sure future objects have proper fields appended at creation (task_ or task + job_spec)
    - [x] Move writing of restart files into the FutureProcessingStrategy implementations
    - [x] Make sure you remove the finished future rather than popleft in FutureProcessingStrategy implementations

## NOTE: Refactored code runs way slower 

- [x] Fix problems causing slowdown and test again
    ### --- More Problems ---
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
    ### --- Def File Spec ---
    - The file should be off of a frontier base image use rocky linux version
    - Install build dependecies of flux-core
    - Build flux-core from source 
    - Install build dependencies for flux-sched
    - Build flux-sched from source 
    - Export all variables 
    - Install matensemble
- [x] Test the apptainer container 

## --- Build Base Images ---
- [x] Build Base Image for Baseline
- [x] Build Base Image for Frontier
- [x] Build Base Image for Perlmutter
- [x] Push images to GitHub Container Registry
- [x] Build MatEnsemble Images with each base image
- [x] Test Images on each respective system
- [ ] NOTE: Come back later -- Test Perlmutter Image
- [ ] Create Perlmutter image from new base image [Neil's Containerfiles](https://github.com/namehta4/Containerfiles/blob/main/Base/GPU/Dockerfile)

## --- Setup GitHub Actions ---
- [x] Setup Matrix build action to build MatEnsemble images for baseline, frontier, and perlmutter and push them to ghcr
- [x] Setup action to build with uv and publish with uv 
- [x] Setup action to build docs and publish them

## --- Test CI/CD ---
- [x] Make small change to MatEnsemble and Docs
- [x] Push to main see if dev builds succeed
- [x] Run release.sh script to see if releases happen properly

## --- Updated/Better UX ---
- [x] Refactor to be built around Task/Job Objects
- [x] Allow users to decorate python functions to create TaskSpec's 
    - [x] Allow functions to depend on other functions 
    - [x] Topologically sort all of the Jobs based on dependencies
- [x] Write worker runtime that flux can target and call user defined functions
- [x] Write the Job objects specification to a file in their direcotry 

## --- Status Dashboard ---
- [x] Update the Pipeline.run() method to have a dashboard flag 
- [x] Add logic to launch the dashboard when the user runs the workflow 

## --- Science Example ---
- [x] Test the science example that Soumendu provided 
- [x] Update version of LAMMPS
- [x] Fix Bug with jobspec.env -> jobspec.environment 
- [ ] Test it again 

## --- Polish Everything ---
- [x] Update all the documentation 
- [x] Update the example workflows 
- [x] Provide tutorials for how to run the example workflows 
- [x] Change name of 'Job' to 'Chore'
- [ ] Change name of 'Pipeline' to something else
- [ ] Make ChoreType.PYTHON have the ability to be defined in the runner script

## --- Fix Containers ---
- [x] Install latest version of lammps in frontier images 
- [x] Test MPI problem with MatEnsemble in Frontier Images
- [ ] Test Science Example with latest MPICH install 
- [ ] Ping neil to ask about MPICH in image 

## --- Finish presentation ---
- [ ] Conda environment might make this very simple 
- [ ] Create a jupyter notebook and screen record it 
- [ ] Place that at the end of the presentation

## --- Test Perlmutter Container ---
- [ ] Give them a test with the current command that you have been running 
- [ ] If that doesn't work break it down into smaller pieces 
  ### Smaller pieces 
    - [ ] Maybe start with an nvidia image rather than Neil's image 
    - [ ] Make sure that flux works in the container 

    - [ ] Create a container that just has flux and have some different tests for that
    - [ ] Create a container that has just MPI and test that make sure it works 

    - [ ] Combine flux and MPI and see if that works 
    - [ ] Create a container that has lammps and make sure that that is working
    - [ ] Combine all the pieces 

## --- Test Frontier Apptainer container ---
- [ ] Need lots more help here 
- [ ] Make something that is very small first 

## --- Create first draft for JOSS ---
- [ ] Read some example papers 
- [ ] Create draft and show Dr. Bagchi 
- [ ] Polish the repository to be ready for review 
- [ ] Make sure that the tests work 
- [ ] Make sure that the example workflows work correctly 
- [ ] Make sure that they can easily test the code and 
- [ ] Create a conda package that they can easily test the code without having 
      to compile flux and flux-sched themselves 

## AFTER EVERYTHING ABOVE IS DONE AND STABLE 

## --- Model Context Protocol ---
- [ ] MCP implementation 
- [ ] Map out the Tool and Resources
    ### Implement the Resources 
    - [ ] Resource to Fetch ALL Docs
    - [ ] Resource to Fetch Relavant Source Code
    - [ ] Resource to Fetch Examples General or system dependent 
    ### Implement the Tools 
    - [ ] Tool to create a directory for the workflow 
    - [ ] Tool to write a file in that directory 
    - [ ] Tool to delete a file in that directory 
    - [ ] Tool to create a workflow 
    - [ ] Tool to verify a workflow 
    - [ ] Tool to create a batch script 
    - [ ] Tool to setup container env
    - [ ] Tool to submit a batch script 
    ### Implement the Prompts 
    - [ ] ???
- [ ] Test the server locally 
- [ ] Test the server on an HPC cluster 
- [ ] Create documentation for setting it up 


## --- Reading List ---
- [ ] [Agentic Orchestration of HPC Applications](https://vsoch.github.io/assets/posts/agentic-orchestration-hpc-workloads-cloud-sochat-milroy.pdf)
- [x] [Container Training Slides](https://drive.google.com/drive/folders/1_mTBBc98TEX3XFpNp0rqoqj1VjN9TKoO)
- [ ] [Containers as Jupyter Kernels](https://docs.nersc.gov/services/jupyter/how-to-guides/#how-to-use-a-container-to-run-a-jupyter-kernel)
- [ ] [Using SPIN to Run Persistent Containers](https://docs.nersc.gov/services/spin/)
- [ ] [Using uv to package lammps and flux into pip install???](https://sgoel.dev/posts/building-cython-or-c-extensions-using-uv/)

- [ ] Ping Neil about MPICH 
- [x] Convert Scaffold into PowerPoint Presentation 
- [x] Email coordinatior about length of presentation and audience


