TOOL_NAME=ModelVerification
VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4

echo "Preparing $TOOL_NAME for benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE' and vnnlib file '$VNNLIB_FILE'"

# if [ $CATEGORY != "nn4sys" ]; then
# 	echo "Only support nn4sys"
# 	exit 1
# fi

# kill any zombie processes

yes n | case $ONNX_FILE in *.gz) gzip -kd $ONNX_FILE;; esac

script_name=$0
script_path=$(dirname "$0")
project_path=$(dirname "$script_path")
# julia --project="${project_path}" "${script_path}/prepare_instance.jl"  "$ONNX_FILE"

tmux send-keys -t MV:0 "using include(\"ModelVerification.jl/vnncomp_scripts/run_instance.jl\")" C-m

# script returns a 0 exit code if successful. If you want to skip a benchmark category you can return non-zero.
exit 0