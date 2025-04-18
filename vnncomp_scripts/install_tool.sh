sudo sed -i "/#\$nrconf{restart} = 'i';/s/.*/\$nrconf{restart} = 'a';/" /etc/needrestart/needrestart.conf

apt-get update -y
apt-get install sudo -y
sudo apt-get update -y
sudo apt-get install wget -y
sudo wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.3-linux-x86_64.tar.gz
sudo tar -xvzf julia-1.9.3-linux-x86_64.tar.gz
sudo cp -r julia-1.9.3 /opt/
sudo ln -s /opt/julia-1.9.3/bin/julia /usr/local/bin/julia
sudo apt-get install build-essential -y
sudo apt-get install git -y
sudo apt-get install python3 -y
sudo apt-get install python3-pip -y
sudo apt-get install psmisc -y
sudo apt-get install aria2 -y
source ~/.bashrc

# # Install NVIDIA driver

sudo apt-get upgrade -y linux-aws
cat << EOF | sudo tee --append /etc/modprobe.d/blacklist.conf
blacklist vga16fb
blacklist nouveau
blacklist rivafb
blacklist nvidiafb
blacklist rivatv
EOF
GRUB_CMDLINE_LINUX="rdblacklist=nouveau"
sudo update-grub
sudo apt  install awscli 
aws s3 cp --recursive s3://ec2-linux-nvidia-drivers/latest/ .
chmod +x NVIDIA-Linux-x86_64*.run
sudo /bin/sh ./NVIDIA-Linux-x86_64*.run
nvidia-smi -q | head
sudo reboot


git clone git@github.com:intelligent-control-lab/ModelVerification.jl.git

script_name=$0
script_path=$(dirname "$0")
project_path=$(dirname "$script_path")

cd $project_path
echo '
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.dev(".")
Pkg.dev("./onnx_parser/NaiveNASflux")
Pkg.dev("./onnx_parser/ONNXNaiveNASflux")
Pkg.add("LazySets")
Pkg.add("PyCall")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("CUDA")
using CUDA
CUDA.set_runtime_version!(v"12.2")
' | julia

script_name=$0
script_path=$(dirname "$0")
chmod +x ${script_path}/*.sh
pip3 install -r "${script_path}/NNet/test_requirements.txt"
cd ~/vnncomp2023_benchmarks
tmux new-session -d -s MV
tmux send-keys -t MV:0 "julia" C-m
tmux send-keys -t MV:0 "include(\"/home/ubuntu/ModelVerification.jl/vnncomp_scripts/vnn_funcs.jl\")" C-m
tmux send-keys -t MV:0 "warmup(\"/home/ubuntu/vnncomp2023_benchmarks/benchmarks/acasxu/\")" C-m
sleep 120