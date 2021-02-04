
conda create --name new_chain
conda activate new_chain

git clone https://github.com/mggg/GerryChain.git
cd GerryChain

wget https://github.com/gerrymandr/blob/main/rev_recom_demo/install_script.sh
sh install_script.sh
