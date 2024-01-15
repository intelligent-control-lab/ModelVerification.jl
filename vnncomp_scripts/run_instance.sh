VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4
RESULTS_FILE=$5
TIMEOUT=$6

echo "Running benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE', vnnlib file '$VNNLIB_FILE', results file $RESULTS_FILE, and timeout $TIMEOUT"

# run the tool to produce the results file
script_name=$0
script_path=$(dirname "$0")
project_path=$(dirname "$script_path")
tmux send-keys -t MV:0 "vnn_verify(\"$CATEGORY\", \"$CATEGORY\", \"$ONNX_FILE\", \"$VNNLIB_FILE\", \"$RESULTS_FILE\", \"$TIMEOUT\")" C-m
exit 0