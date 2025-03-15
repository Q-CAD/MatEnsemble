### To install in Frontier (HPE-Cray synstem with AMD GPUS)
This will attempt to build all the dependencies from scratch, so 
1) make sure that you are at a location which has enough storage 
2) expect to take this a while. Recommend to use `nohup` type processes launching. 
```bash
git clone https://github.com/Q-CAD/MatEnsemble.git
cd matensemble/scripts/frontier
chmod +x install_frontier.sh
./install_frontier.sh
```