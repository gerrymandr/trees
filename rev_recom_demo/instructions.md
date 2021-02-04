## Instructions to try out the new RevRecom release of GerryChain

All of these commands are on the command line.

1. Clone the GerryChain repo.
    `git clone https://github.com/mggg/GerryChain.git`

2. Start a new Conda environment and activate it. Assuming you have conda already installed:
   `conda create --name new_chain`
   `conda activate new_chain`

3. Enter the directory of the `GerryChain` repo you just cloned.
   `cd GerryChain`

4. Copy the `install_script.sh` file in this Github directory (not the one you just entered!)
   The file is at https://github.com/gerrymandr/trees/blob/main/rev_recom_demo/install_script.sh .
   Put this file in the `GerryChain` repository. This can be done by entering
   `wget https://github.com/gerrymandr/trees/blob/main/rev_recom_demo/install_script.sh`

5. Run the script. This installs all the dependencies for the new release.
   `sh install_script.sh`

This should set you up. Go run your scripts and let Bhushan know if anything broke!
